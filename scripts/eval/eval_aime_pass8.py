"""
Evaluate models on AIME dataset with pass@k metric.
Generates multiple samples per problem and calculates pass@k.
"""

import argparse
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
from vllm import LLM, SamplingParams

try:
    from math_verify import verify, parse as math_parse
    _MATH_VERIFY_AVAILABLE = True
except ImportError:
    _MATH_VERIFY_AVAILABLE = False


def extract_boxed_answer(text: str) -> str | None:
    """Extract the final answer from \\boxed{}, handling nested braces.
    Tries from the last match backwards until a fully closed one is found.
    """
    pattern = r"\\boxed\{"
    matches = list(re.finditer(pattern, text))
    if not matches:
        return None
    # Try from last to first, return first fully closed boxed
    for match in reversed(matches):
        start = match.end()
        depth = 1
        i = start
        while i < len(text) and depth > 0:
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
            i += 1
        if depth == 0:
            return text[start:i-1].strip()
    return None


def normalize_answer(ans: str) -> str:
    """Normalize answer for comparison."""
    ans = ans.strip()
    # Normalize \dfrac to \frac
    ans = ans.replace('\\dfrac', '\\frac')
    # Remove trailing .0 for integer answers
    if re.match(r'^\d+\.0$', ans):
        ans = ans[:-2]
    # Remove spacing commands
    ans = ans.replace('\\,', '').replace('\\!', '').replace('\\;', '')
    ans = ans.replace('\\quad', '').replace('\\qquad', '')
    # Normalize spaces
    ans = re.sub(r'\s+', ' ', ans).strip()
    return ans


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--data-path", type=str, default="/home/nvidia/peirongcan/negative-sd/data/aime/all.parquet")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--max-tokens", type=int, default=32000)
    parser.add_argument("--max-model-len", type=int, default=32000)
    parser.add_argument("--gpu-memory-util", type=float, default=0.85)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--use-chat-template", action="store_true",
                        help="Apply tokenizer chat template (needed for Instruct models)")
    parser.add_argument("--enable-thinking", action="store_true",
                        help="Enable Qwen3 thinking mode (passes enable_thinking=True to apply_chat_template)")
    parser.add_argument("--disable-thinking", action="store_true",
                        help="Disable Qwen3 thinking mode (passes enable_thinking=False to apply_chat_template)")
    parser.add_argument("--use-math-verify", action="store_true",
                        help="Use math_verify for answer checking (needed for non-integer answers like HMMT)")
    parser.add_argument("--run-name", default=None, help="Override output filename stem (default: derived from model-path)")
    args = parser.parse_args()

    # Load AIME data
    df = pd.read_parquet(args.data_path)
    df = df[df['Year'] == args.year].copy()
    df = df.reset_index(drop=True)

    print(f"Loaded {len(df)} AIME {args.year} samples from {args.data_path}")

    # Extract prompts and ground truths
    questions = []
    ground_truths = []
    ids = []

    for _, row in df.iterrows():
        part = f"-{row['Part']}" if 'Part' in row and pd.notna(row['Part']) and str(row['Part']).strip() not in ('', 'nan') else ""
        ids.append(f"{row['Year']}{part}-{row['Problem Number']}")
        question = row['Question']
        answer = str(row['Answer']).strip()

        questions.append(question)
        ground_truths.append(answer)

    # Build vLLM
    print(f"Loading model from {args.model_path} ...")
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_util,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        disable_custom_all_reduce=True,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        max_tokens=args.max_tokens,
        repetition_penalty=args.repetition_penalty,
        stop=None if args.use_chat_template else ["\n\n\n", "##"],
    )

    # Load tokenizer for chat template if needed
    tokenizer = None
    if args.use_chat_template:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        print(f"Loaded tokenizer for chat template from {args.model_path}")

    # Expand prompts for parallel processing: each problem gets N independent requests
    # This allows vLLM to process all 960 requests (30 problems x 32 samples) concurrently
    expanded_prompts = []
    prompt_to_problem = []  # Maps each request back to its problem index

    for problem_idx, question in enumerate(questions):
        for sample_idx in range(args.num_samples):
            if args.use_chat_template:
                user_msg = f"{question}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
                messages = [{"role": "user", "content": user_msg}]
                template_kwargs = dict(tokenize=False, add_generation_prompt=True)
                if args.enable_thinking:
                    template_kwargs["enable_thinking"] = True
                elif args.disable_thinking:
                    template_kwargs["enable_thinking"] = False
                prompt_text = tokenizer.apply_chat_template(messages, **template_kwargs)
            else:
                prompt_text = f"Solve the following problem. Put your final answer within \\boxed{{}}.\n\n{question}\n\nSolution:"
            expanded_prompts.append(prompt_text)
            prompt_to_problem.append(problem_idx)

    print(f"Total concurrent requests: {len(expanded_prompts)} ({len(df)} problems x {args.num_samples} samples)")
    print("Starting generation with full concurrency...")

    # Generate all requests concurrently
    t0 = time.time()
    outputs = llm.generate(expanded_prompts, sampling_params)
    elapsed = time.time() - t0
    print(f"Generation done in {elapsed:.1f}s ({len(expanded_prompts)/elapsed:.2f} samples/s)")

    # Process results: group samples by problem
    results = []
    correct_count = 0
    per_problem_correct = []

    # Group outputs by problem (each problem has args.num_samples responses)
    for problem_idx in range(len(questions)):
        question_id = ids[problem_idx]
        question = questions[problem_idx]
        gt = ground_truths[problem_idx]
        gt_norm = normalize_answer(gt)

        # Collect all N samples for this problem
        samples_correct = []
        all_responses = []
        all_answers = []

        for sample_idx in range(args.num_samples):
            # Calculate the index in the expanded outputs array
            output_idx = problem_idx * args.num_samples + sample_idx
            if output_idx < len(outputs):
                sample_output = outputs[output_idx]
                # With n=1 in sampling_params, outputs[0] contains the single generation
                response = sample_output.outputs[0].text
                pred_answer = extract_boxed_answer(response)
                pred_norm = normalize_answer(pred_answer) if pred_answer else None

                if args.use_math_verify and _MATH_VERIFY_AVAILABLE and pred_answer is not None:
                    try:
                        is_correct = bool(verify(math_parse(pred_answer), math_parse(gt)))
                    except Exception:
                        is_correct = (pred_norm is not None) and (pred_norm == gt_norm)
                else:
                    is_correct = (pred_norm is not None) and (pred_norm == gt_norm)
                samples_correct.append(is_correct)
                all_responses.append(response)
                all_answers.append(pred_answer)

        # Problem is correct if at least one sample is correct
        problem_correct = any(samples_correct)
        per_problem_correct.append(problem_correct)
        if problem_correct:
            correct_count += 1

        # Count how many samples were correct
        num_correct_samples = sum(samples_correct)

        result = {
            "id": question_id,
            "question": question,
            "ground_truth": gt,
            "ground_truth_normalized": gt_norm,
            "correct": problem_correct,
            "num_correct_samples": num_correct_samples,
            "total_samples": args.num_samples,
            "all_responses": all_responses,
            "all_answers": all_answers,
        }
        results.append(result)

    # Calculate metrics
    total = len(df)
    pass_at_k = correct_count / total * 100
    avg_at_k = sum(r['num_correct_samples'] for r in results) / (total * args.num_samples) * 100
    pass_at_1 = sum(1 for r in results if r['num_correct_samples'] >= 1) / total * 100

    print(f"\n{'='*60}")
    print(f"Model: {args.model_path}")
    print(f"Year: {args.year}")
    print(f"Total samples: {total}")
    print(f"Samples per problem: {args.num_samples}")
    print(f"avg@{args.num_samples}: {avg_at_k:.2f}%")
    print(f"pass@1: {pass_at_1:.2f}%")
    print(f"pass@{args.num_samples}: {pass_at_k:.2f}%")
    print(f"{'='*60}")

    # Extract model name for filename
    if args.run_name:
        model_name = args.run_name
    elif "global_step" in args.model_path:
        model_name = f"step_{args.model_path.split('global_step_')[-1].replace('/', '')}"
    elif "hf_step" in args.model_path:
        model_name = args.model_path.split('/')[-1]
    elif "/" in args.model_path:
        model_name = args.model_path.split('/')[-1].replace('/', '_')
    else:
        model_name = "baseline"

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    output_path = os.path.join(args.output_dir, f"eval_aime_y{args.year}_{model_name}_pass{args.num_samples}.parquet")
    results_df = pd.DataFrame(results)
    results_df.to_parquet(output_path, index=False)
    print(f"Saved raw results to {output_path}")

    # Save summary
    summary = {
        "model": args.model_path,
        "model_name": model_name,
        "year": args.year,
        "total_samples": total,
        "num_samples_per_problem": args.num_samples,
        "correct": correct_count,
        "avg_at_k": f"{avg_at_k:.2f}%",
        "pass_at_1": f"{pass_at_1:.2f}%",
        "pass_at_k": f"{pass_at_k:.2f}%",
        "elapsed_sec": elapsed,
    }

    summary_path = os.path.join(args.output_dir, f"eval_aime_y{args.year}_{model_name}_pass{args.num_samples}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")

    return summary


if __name__ == "__main__":
    main()

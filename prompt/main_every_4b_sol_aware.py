"""
Per-sample NSD dataset generation — solution-aware attack prompts.

For each training sample:
  1. Qwen3-4B (instruct mode) generates a rollout on the clean problem
  2. The rollout is used to generate a solution-aware "Targeted Attack Prompt"
  3. Dataset columns match math_nsd_every_4b_instruct format

Uses Qwen3-4B (instruct mode, enable_thinking=False).
Attack prompt file: prompt/attack_prompt/with_solution.txt
Output: data/math_nsd_every_4b_sol_aware/
"""

import argparse
import itertools
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from openai import OpenAI

CACHE_DIR = Path(__file__).parent / "cache_every_4b_sol_aware"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

VLLM_ENDPOINTS = [f"http://localhost:{8000+i}/v1" for i in range(6)]
clients = [OpenAI(api_key="EMPTY", base_url=url) for url in VLLM_ENDPOINTS]
client_pool = itertools.cycle(clients)

_PROMPT_FILE = Path(__file__).parent / "attack_prompt" / "with_solution.txt"
ATTACK_PROMPT_TEMPLATE = _PROMPT_FILE.read_text()

STUDENT_SUFFIX = r"Let's think step by step and output the final answer within \boxed{}."


@dataclass
class Config:
    model_path: str = "Qwen3-4B"
    input_path: str = "data/math/train.parquet"
    test_path: str = "data/math/test.parquet"
    output_dir: str = "data/math_nsd_every_4b_sol_aware"
    seed: int = 42
    max_workers: int = 30
    resume: bool = False


def call_model(client, messages, max_tokens=4096, temperature=0.0, model="Qwen3-4B"):
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return resp.choices[0].message.content.strip()


def solve(client, problem, model):
    content = f"Problem: {problem}\n\n{STUDENT_SUFFIX}"
    return call_model(client, [{"role": "user", "content": content}], temperature=0.0, model=model)


def gen_attack(client, problem, solution, model):
    prompt = ATTACK_PROMPT_TEMPLATE.format(problem=problem, solution=solution)
    return call_model(client, [{"role": "user", "content": prompt}],
                      max_tokens=400, temperature=0.7, model=model)


def clean_attack_prompt(raw: str) -> Optional[str]:
    raw = raw.strip()
    if raw.startswith('"') and raw.endswith('"'):
        raw = raw[1:-1].strip()

    think_pattern = re.compile(r'<\\?think\s*/?>', re.IGNORECASE)
    match = think_pattern.search(raw)
    if match:
        raw = raw[match.end():].strip()

    if '\n\n' in raw:
        paragraphs = [p.strip() for p in raw.split('\n\n') if p.strip()]
        if paragraphs:
            raw = paragraphs[-1]

    lines = [l.strip() for l in raw.split('\n') if l.strip()]
    if len(lines) > 1 and not raw.startswith('You are'):
        for line in reversed(lines):
            if line.startswith('You are'):
                raw = line
                break

    if len(raw) < 30:
        return None
    if not raw.startswith('You are'):
        return None
    return raw


def extract_boxed_answer(solution: str) -> str:
    matches = re.findall(r"\\boxed\{([^}]*(?:\{[^}]*\}[^}]*)*)\}", str(solution))
    return matches[-1].strip() if matches else ""


def generate_attack_for_sample(index: int, problem: str, config: Config) -> Optional[tuple[int, str]]:
    client = next(client_pool)
    cache_file = CACHE_DIR / f"attack_{index:05d}.txt"

    if config.resume and cache_file.exists():
        cached = cache_file.read_text().strip()
        if cached:
            return (index, cached)

    try:
        # Step 1: rollout
        solution = solve(client, problem, config.model_path)

        # Step 2: generate targeted attack
        raw_attack = gen_attack(client, problem, solution, config.model_path)
        attack_prompt = clean_attack_prompt(raw_attack)

        if attack_prompt:
            cache_file.write_text(attack_prompt)
            return (index, attack_prompt)
        else:
            print(f"  [Warning] Failed to clean attack for index {index}")
            return None

    except Exception as e:
        print(f"  [Warning] Failed for index {index}: {e}")
        return None


def build_student_prompt(question: str) -> list[dict]:
    content = f"Problem: {question}\n\n{STUDENT_SUFFIX}"
    return [{"role": "user", "content": content}]


def build_attack_teacher_prompt(question: str, attack_prompt: str) -> list[dict]:
    content = f"Problem: {question}\n\n{attack_prompt}\n\nNow solve the problem following this instruction:\n\n{STUDENT_SUFFIX}"
    return [{"role": "user", "content": content}]


def build_reference_teacher_prompt(question: str) -> list[dict]:
    content = f"Problem: {question}\n\n{STUDENT_SUFFIX}"
    return [{"role": "user", "content": content}]


def generate_train_dataset(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    output_path = Path(config.output_dir) / "train.parquet"

    if output_path.exists() and config.resume:
        print(f"  [cached] {output_path}")
        return pd.read_parquet(output_path)

    print(f"  Generating solution-aware attack prompts for {len(df)} training samples...")
    print(f"  Using {config.max_workers} workers...")

    attack_prompts = [None] * len(df)

    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        futures = {
            executor.submit(generate_attack_for_sample, i, row["problem"], config): i
            for i, row in df.iterrows()
        }
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            if result:
                index, attack = result
                attack_prompts[index] = attack
                completed += 1
                if completed % 100 == 0:
                    print(f"    Progress: {completed}/{len(df)}")

    missing = sum(1 for p in attack_prompts if p is None)
    if missing > 0:
        print(f"  [Warning] {missing} samples failed to generate")

    rows = []
    skipped = 0
    for i, (_, row) in enumerate(df.iterrows()):
        if attack_prompts[i] is None:
            skipped += 1
            continue
        gt = extract_boxed_answer(row["solution"])
        if not gt:
            skipped += 1
            continue
        question = row["problem"]
        rows.append({
            "data_source": "EleutherAI/hendrycks_math",
            "prompt": build_student_prompt(question),
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": gt},
            "extra_info": {
                "split": "train",
                "index": i,
                "answer": row["solution"],
                "question": question,
                "type": row.get("type", ""),
                "level": row.get("level", ""),
            },
            "teacher_prompt": build_reference_teacher_prompt(question),
            "attack_teacher_prompt": build_attack_teacher_prompt(question, attack_prompts[i]),
            "hint_type": f"sol_aware_{i:05d}",
        })

    result = pd.DataFrame(rows)
    result.to_parquet(output_path)
    print(f"  Saved {len(result)} rows (skipped {skipped}) -> {output_path}")
    return result


def generate_test_dataset(df: pd.DataFrame, config: Config, n_test: int = 200) -> pd.DataFrame:
    output_path = Path(config.output_dir) / f"test_{n_test}.parquet"

    if output_path.exists():
        print(f"  [cached] {output_path}")
        return pd.read_parquet(output_path)

    print(f"  Generating test dataset ({n_test} samples)...")
    rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        if i >= n_test:
            break
        gt = extract_boxed_answer(row["solution"])
        question = row["problem"]
        reference_prompt = build_reference_teacher_prompt(question)
        rows.append({
            "data_source": "EleutherAI/hendrycks_math",
            "prompt": build_student_prompt(question),
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": gt},
            "extra_info": {
                "split": "test",
                "index": i,
                "answer": row["solution"],
                "question": question,
                "type": row.get("type", ""),
                "level": row.get("level", ""),
            },
            "teacher_prompt": reference_prompt,
            "attack_teacher_prompt": reference_prompt,
            "hint_type": "none",
        })

    result = pd.DataFrame(rows)
    result.to_parquet(output_path)
    print(f"  Saved {len(result)} rows -> {output_path}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Generate per-sample NSD dataset with solution-aware attacks (Qwen3-4B)")
    parser.add_argument("--model_path", default="Qwen3-4B")
    parser.add_argument("--input_path", default="data/math/train.parquet")
    parser.add_argument("--test_path", default="data/math/test.parquet")
    parser.add_argument("--output_dir", default="data/math_nsd_every_4b_sol_aware")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_workers", type=int, default=30)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--n_test", type=int, default=200)
    args = parser.parse_args()

    config = Config(
        model_path=args.model_path,
        input_path=args.input_path,
        test_path=args.test_path,
        output_dir=args.output_dir,
        seed=args.seed,
        max_workers=args.max_workers,
        resume=args.resume,
    )

    project_root = Path(__file__).resolve().parent.parent
    config.input_path = str(project_root / config.input_path)
    config.test_path = str(project_root / config.test_path)
    config.output_dir = str(project_root / config.output_dir)
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Per-Sample NSD Dataset Generation (Qwen3-4B, solution-aware)")
    print("=" * 60)
    print(f"Model:       {config.model_path}")
    print(f"Output:      {config.output_dir}")
    print(f"Workers:     {config.max_workers}")
    print(f"Resume:      {config.resume}")
    print(f"Prompt file: {_PROMPT_FILE}")
    print()

    print("[Loading Data]")
    train_df = pd.read_parquet(config.input_path)
    test_df = pd.read_parquet(config.test_path)
    print(f"  Train: {len(train_df)} samples")
    print(f"  Test:  {len(test_df)} samples")
    print()

    print("[Generating Training Dataset]")
    train_result = generate_train_dataset(train_df, config)
    print()

    print("[Generating Test Dataset]")
    test_result = generate_test_dataset(test_df, config, n_test=args.n_test)
    print()

    print("=" * 60)
    print(f"Train: {len(train_result)} rows")
    print(f"Test:  {len(test_result)} rows")
    print(f"Output: {config.output_dir}")
    print(f"Cache:  {CACHE_DIR}")

    sample = train_result.iloc[0]
    print(f"\nSample attack_teacher_prompt:")
    print(f"  {sample['attack_teacher_prompt'][0]['content'][:300]}...")
    print("\nDone!")


if __name__ == "__main__":
    main()

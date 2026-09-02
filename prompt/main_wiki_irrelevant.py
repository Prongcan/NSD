"""
Wikipedia irrelevant context dataset generation.

For each training sample, randomly assigns one Wikipedia article as irrelevant context.
Uses the lure phrase "Below is some context you may find useful" to match the original
paper's setup (arxiv 2504.02111 / reasoning-perturbations-study).

Output format matches main_every_4b_instruct.py.
"""

import argparse
import random
import re
from pathlib import Path

import pandas as pd


WIKI_DIR = Path(__file__).resolve().parent.parent / \
    "scripts/case_study/reasoning-perturbations-study/data/irrelevant"

STUDENT_SUFFIX = r"Let's think step by step and output the final answer within \boxed{}."

MIN_WIKI_CHARS = 500  # filter out near-empty files


def load_wiki_texts(wiki_dir: Path) -> list[str]:
    texts = []
    for f in sorted(wiki_dir.glob("*.txt")):
        text = f.read_text().strip()
        if len(text) >= MIN_WIKI_CHARS:
            texts.append(text)
    print(f"  Loaded {len(texts)} Wikipedia articles from {wiki_dir}")
    return texts


def extract_boxed_answer(solution: str) -> str:
    matches = re.findall(r"\\boxed\{([^}]*(?:\{[^}]*\}[^}]*)*)\}", str(solution))
    return matches[-1].strip() if matches else ""


def build_student_prompt(question: str) -> list[dict]:
    content = f"Problem: {question}\n\n{STUDENT_SUFFIX}"
    return [{"role": "user", "content": content}]


def build_reference_teacher_prompt(question: str) -> list[dict]:
    content = f"Problem: {question}\n\n{STUDENT_SUFFIX}"
    return [{"role": "user", "content": content}]


def build_wiki_attack_prompt(question: str, wiki_text: str) -> list[dict]:
    content = (
        f"Problem: {question}\n\n"
        f"Below is some context you may find useful to answering the question above:\n\n"
        f"{wiki_text}\n\n"
        f"{STUDENT_SUFFIX}"
    )
    return [{"role": "user", "content": content}]


def generate_train_dataset(df: pd.DataFrame, wiki_texts: list[str], seed: int, output_dir: Path) -> pd.DataFrame:
    output_path = output_dir / "train.parquet"

    rng = random.Random(seed)
    rows = []
    skipped = 0

    for i, (_, row) in enumerate(df.iterrows()):
        gt = extract_boxed_answer(row["solution"])
        if not gt:
            skipped += 1
            continue

        question = row["problem"]
        wiki_text = rng.choice(wiki_texts)

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
            "attack_teacher_prompt": build_wiki_attack_prompt(question, wiki_text),
            "hint_type": f"wiki_irrelevant_{i:05d}",
        })

    result = pd.DataFrame(rows)
    result.to_parquet(output_path)
    print(f"  Saved {len(result)} rows (skipped {skipped}) -> {output_path}")
    return result


def generate_test_dataset(df: pd.DataFrame, output_dir: Path, n_test: int = 200) -> pd.DataFrame:
    output_path = output_dir / f"test_{n_test}.parquet"

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
    parser = argparse.ArgumentParser(description="Generate NSD dataset with Wikipedia irrelevant context attack")
    parser.add_argument("--input_path", default="data/math/train.parquet")
    parser.add_argument("--test_path", default="data/math/test.parquet")
    parser.add_argument("--output_dir", default="data/math_nsd_wiki_irrelevant")
    parser.add_argument("--wiki_dir", default=None, help="Override Wikipedia articles directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_test", type=int, default=200)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    input_path = project_root / args.input_path
    test_path = project_root / args.test_path
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    wiki_dir = Path(args.wiki_dir) if args.wiki_dir else WIKI_DIR

    print("=" * 60)
    print("Wikipedia Irrelevant Context NSD Dataset Generation")
    print("=" * 60)
    print(f"Input:      {input_path}")
    print(f"Output:     {output_dir}")
    print(f"Wiki dir:   {wiki_dir}")
    print(f"Seed:       {args.seed}")
    print()

    print("[Loading Wikipedia articles]")
    wiki_texts = load_wiki_texts(wiki_dir)
    print()

    print("[Loading Data]")
    train_df = pd.read_parquet(input_path)
    test_df = pd.read_parquet(test_path)
    print(f"  Train: {len(train_df)} samples")
    print(f"  Test:  {len(test_df)} samples")
    print()

    print("[Generating Training Dataset]")
    train_result = generate_train_dataset(train_df, wiki_texts, args.seed, output_dir)
    print()

    print("[Generating Test Dataset]")
    test_result = generate_test_dataset(test_df, output_dir, n_test=args.n_test)
    print()

    print("=" * 60)
    print(f"Train: {len(train_result)} rows")
    print(f"Test:  {len(test_result)} rows")
    print(f"Output: {output_dir}")

    sample = train_result.iloc[0]
    print(f"\nSample attack_teacher_prompt (first 300 chars):")
    print(f"  {sample['attack_teacher_prompt'][0]['content'][:300]}...")
    print("\nDone!")


if __name__ == "__main__":
    main()

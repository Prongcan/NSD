"""CANON: Consensus-Anchored Self-Distillation — Consensus Extraction Module.

Computes majority-vote consensus from N rollouts per prompt and builds
teacher prompts conditioned on the consensus solution.
"""
from collections import Counter
from typing import List, Optional, Tuple

import numpy as np
import torch

from verl.utils.reward_score.ttrl_math import extract_answer, simplify_expression_string


def extract_consensus(
    responses: List[str],
    log_probs: Optional[torch.Tensor] = None,
    selection: str = "highest_logprob",
) -> Tuple[Optional[str], Optional[str], float]:
    """Extract consensus answer and select best matching solution.

    Args:
        responses: N decoded response strings for one prompt.
        log_probs: [N] mean log-prob per response (used for solution selection).
        selection: "highest_logprob" or "random".

    Returns:
        (consensus_answer, best_solution, consensus_ratio)
        Returns (None, None, 0.0) if no valid answers found.
    """
    n = len(responses)
    answers = []
    for resp in responses:
        ans = extract_answer(resp)
        if ans is not None:
            try:
                ans = simplify_expression_string(ans)
            except Exception:
                pass
        answers.append(ans)

    valid_answers = [a for a in answers if a is not None]
    if not valid_answers:
        return None, None, 0.0

    counter = Counter(valid_answers)
    consensus_answer, consensus_count = counter.most_common(1)[0]
    consensus_ratio = consensus_count / n

    matching_indices = [
        i for i, a in enumerate(answers) if a == consensus_answer
    ]

    if not matching_indices:
        return consensus_answer, None, consensus_ratio

    if selection == "highest_logprob" and log_probs is not None:
        matching_lps = [(i, log_probs[i].item()) for i in matching_indices]
        best_idx = max(matching_lps, key=lambda x: x[1])[0]
    else:
        best_idx = matching_indices[np.random.randint(len(matching_indices))]

    return consensus_answer, responses[best_idx], consensus_ratio


def build_canon_teacher_prompt(problem: str, consensus_solution: str) -> List[dict]:
    """Build teacher prompt conditioned on the consensus solution."""
    content = (
        f"{problem}\n\n"
        f"Here is a verified correct solution for reference:\n"
        f"{consensus_solution}\n\n"
        f"Now solve this problem step by step, and put your final answer within \\boxed{{}}."
    )
    return [{"role": "user", "content": content}]


def batch_extract_consensus(
    all_responses: List[str],
    n: int,
    log_probs: Optional[torch.Tensor] = None,
    selection: str = "highest_logprob",
) -> Tuple[List[Optional[str]], List[Optional[str]], List[float]]:
    """Extract consensus for a batch of prompts, each with N rollouts.

    Args:
        all_responses: [num_prompts * n] decoded responses, grouped by prompt.
        n: number of rollouts per prompt.
        log_probs: [num_prompts * n] mean log-prob per response.
        selection: solution selection method.

    Returns:
        (consensus_answers, best_solutions, consensus_ratios) — each of length num_prompts.
    """
    assert len(all_responses) % n == 0
    num_prompts = len(all_responses) // n

    consensus_answers = []
    best_solutions = []
    consensus_ratios = []

    for i in range(num_prompts):
        start = i * n
        end = start + n
        group_responses = all_responses[start:end]
        group_lps = log_probs[start:end] if log_probs is not None else None

        ans, sol, ratio = extract_consensus(group_responses, group_lps, selection)
        consensus_answers.append(ans)
        best_solutions.append(sol)
        consensus_ratios.append(ratio)

    return consensus_answers, best_solutions, consensus_ratios

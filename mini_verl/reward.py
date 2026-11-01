"""Rule-based, verifiable rewards.

``default_compute_score`` mirrors ``verl/utils/reward_score/__init__.py``: dispatch on
``data_source`` to a task-specific ``compute_score(solution_str, ground_truth)`` (cf.
``verl/utils/reward_score/gsm8k.py``: strict format extraction, ``format_score`` vs ``score``).
``compute_reward`` mirrors ``verl/workers/reward_manager/naive.py``: decode each valid response,
score it, and put the scalar on the LAST valid response token of ``token_level_scores``.
"""

import re
from typing import Dict, List, Tuple

import torch

from .data import Tokenizer
from .protocol import DataProto


def extract_solution(solution_str: str, method: str = "strict"):
    """strict: the whole answer must be a bare integer (format check); flexible: last integer."""
    if method == "strict":
        return solution_str if re.fullmatch(r"\d+", solution_str) else None
    nums = re.findall(r"\d+", solution_str)
    return nums[-1] if nums else None


def compute_score(solution_str: str, ground_truth: str, method="strict", format_score=0.0, score=1.0) -> float:
    """``arithmetic/addition``: 1.0 if correct, format_score if well-formed but wrong, 0 otherwise."""
    answer = extract_solution(solution_str, method)
    if answer is None:
        return 0.0
    return score if answer == ground_truth else format_score


def compute_score_tool(solution_str: str, ground_truth: str, format_score=0.0, score=1.0) -> float:
    """``arithmetic/addition_tool``: the final answer is what follows the last tool response."""
    final = solution_str.split("</r>")[-1]
    return compute_score(final, ground_truth, "strict", format_score, score)


def default_compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs) -> float:
    if data_source == "arithmetic/addition":
        return compute_score(solution_str, ground_truth)
    if data_source == "arithmetic/addition_tool":
        return compute_score_tool(solution_str, ground_truth)
    raise NotImplementedError(f"Reward function is not implemented for {data_source=}")


def compute_reward(data: DataProto, tokenizer: Tokenizer, compute_score_fn=default_compute_score
                   ) -> Tuple[torch.Tensor, Dict[str, List]]:
    """NaiveRewardManager: returns (reward_tensor [bs, response_length], reward_extra_info)."""
    responses = data.batch["responses"]
    response_length = responses.size(1)
    valid_response_length = data.batch["attention_mask"][:, -response_length:].sum(-1)
    reward_tensor = torch.zeros_like(responses, dtype=torch.float32)
    scores = []
    for i in range(len(data)):
        n = int(valid_response_length[i])
        solution_str = tokenizer.decode(responses[i, :n].tolist(), skip_special_tokens=True)
        score = compute_score_fn(data.non_tensor_batch["data_source"][i], solution_str,
                                 data.non_tensor_batch["reward_model"][i]["ground_truth"])
        if n > 0:
            reward_tensor[i, n - 1] = score
        scores.append(score)
    return reward_tensor, {"acc": scores}

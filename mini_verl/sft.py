"""A tiny supervised warm-start (SFT) so that RL starts from a policy that is *sometimes* right,
the way real RL post-training starts from an instruct model.

Mirrors the loss of ``verl/workers/utils/losses.py`` ``sft_loss`` (token-mean NLL over response
tokens, ``loss_mask`` = 0 on prompt and on tool-response tokens) as used by
``verl/trainer/sft_trainer.py``. ``greedy_accuracy`` is the same metric as the trainer's
``_validate`` (greedy decoding + rule reward), computed without workers.
"""

import copy
from typing import List, Tuple

import torch

from .agent_loop import ToolAgentLoop, calculator
from .config import DataConfig, ModelConfig, RolloutConfig, SFTConfig
from .core_algos import masked_sum
from .data import RLHFDataset, Tokenizer, collate_fn, left_pad, make_addition_pairs
from .models import TinyPolicy, compute_position_id_with_mask, logprobs_from_logits
from .protocol import DataProto
from .reward import compute_reward
from .rollout import Rollout, right_pad


def sft_example(a: int, b: int, tokenizer: Tokenizer, use_tool: bool = False) -> Tuple[List[int], List[int], List[int]]:
    """(prompt_ids, response_ids, loss_mask). With use_tool the response is a full tool episode:
    ``<c>a+b</c><r>sum</r>sum<eos>`` with loss_mask 0 on the ``<r>...</r>`` observation."""
    prompt = tokenizer.encode(f"{a}+{b}=")
    if not use_tool:
        resp = tokenizer.encode(f"{a + b}<eos>")
        return prompt, resp, [1] * len(resp)
    call = tokenizer.encode(f"<c>{a}+{b}</c>")
    obs = tokenizer.encode(f"<r>{calculator(f'{a}+{b}')}</r>")
    final = tokenizer.encode(f"{a + b}<eos>")
    return prompt, call + obs + final, [1] * len(call) + [0] * len(obs) + [1] * len(final)


def build_sft_batch(examples, max_prompt_length: int, max_response_length: int, pad_id: int) -> dict:
    prompts, prompt_attn = left_pad([e[0] for e in examples], max_prompt_length, pad_id)
    responses = right_pad([e[1] for e in examples], max_response_length, pad_id)
    response_attn = right_pad([[1] * len(e[1]) for e in examples], max_response_length, 0)
    attention_mask = torch.cat([prompt_attn, response_attn], dim=1)
    return {"input_ids": torch.cat([prompts, responses], dim=1), "attention_mask": attention_mask,
            "position_ids": compute_position_id_with_mask(attention_mask), "responses": responses,
            "response_mask": right_pad([e[2] for e in examples], max_response_length, 0) * response_attn}


def sft_loss(policy, batch) -> torch.Tensor:
    R = batch["responses"].size(1)
    logits = policy(batch["input_ids"], batch["attention_mask"], batch["position_ids"])[:, -R - 1:-1]
    log_prob = logprobs_from_logits(logits, batch["responses"])
    return -masked_sum(log_prob, batch["response_mask"]) / batch["response_mask"].sum()


def run_sft(policy, pairs, tokenizer: Tokenizer, steps: int, batch_size: int, lr: float, seed: int,
            max_prompt_length: int, max_response_length: int, use_tool: bool = False, log_every: int = 0):
    """AdamW on random mini-batches of (a, b) pairs. Returns the list of losses."""
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=0.01)
    policy.train()
    losses = []
    for step in range(1, steps + 1):
        idx = torch.randint(len(pairs), (batch_size,), generator=g).tolist()
        batch = build_sft_batch([sft_example(*pairs[i], tokenizer, use_tool) for i in idx],
                                max_prompt_length, max_response_length, tokenizer.pad_token_id)
        loss = sft_loss(policy, batch)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        if log_every and step % log_every == 0:
            print(f"  sft step {step:>4} | loss {loss.item():.4f}")
    return losses


def greedy_accuracy(policy, pairs, tokenizer: Tokenizer, max_prompt_length: int, max_response_length: int,
                    data_source: str = "arithmetic/addition", multi_turn: bool = False) -> float:
    """Greedy-decode every pair with a throw-away rollout engine and return the mean rule reward."""
    cfg = RolloutConfig(response_length=max_response_length, do_sample=False)
    rollout = Rollout(copy.deepcopy(policy), cfg, tokenizer, seed=0)
    ds = RLHFDataset(pairs, tokenizer, max_prompt_length, data_source)
    batch = DataProto.from_single_dict(collate_fn([ds[i] for i in range(len(ds))]))
    gen = batch.pop(batch_keys=["input_ids", "attention_mask", "position_ids"])
    engine = ToolAgentLoop(rollout, tokenizer, cfg) if multi_turn else rollout
    batch.union(engine.generate_sequences(gen))
    _, extra = compute_reward(batch, tokenizer)
    return sum(extra["acc"]) / len(extra["acc"])


def warm_start_policy(model_config: ModelConfig, data_config: DataConfig, sft_config: SFTConfig,
                      use_tool: bool = False) -> TinyPolicy:
    """Fresh TinyPolicy + ``sft_config.steps`` SFT steps on the training pairs (never the val pairs)."""
    d = data_config
    train_pairs, _ = make_addition_pairs(d.operand_max, d.val_size, d.seed)
    policy = TinyPolicy(model_config)
    run_sft(policy, train_pairs, Tokenizer(), sft_config.steps, sft_config.batch_size, sft_config.lr, sft_config.seed,
            d.max_prompt_length, d.max_response_length, use_tool=use_tool)
    return policy

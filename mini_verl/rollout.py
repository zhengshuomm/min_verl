"""The rollout engine: an inference-only copy of the policy that generates responses.

Mirrors ``verl/workers/rollout/base.py`` (``BaseRollout.generate_sequences``) and
``verl/workers/rollout/hf_rollout.py`` (``HFRollout``); the output layout is the one produced by
``verl/experimental/agent_loop/agent_loop.py`` (``_postprocess``): LEFT-padded ``prompts``,
RIGHT-padded ``responses``, ``input_ids = cat(prompts, responses)``, ``attention_mask``,
``position_ids`` and ``response_mask``.

In verl the rollout is a separate inference engine (vLLM / SGLang) holding its OWN copy of the
weights. After every actor update the trainer calls ``update_weights`` (``engine_workers.py``,
``checkpoint_engine/base.py``) to push the new weights; until then the rollout samples from a
stale policy. ``Rollout.sync_weights_from`` is that step, and ``weight_version`` records which
training step the weights come from.
"""

from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F

from .config import RolloutConfig
from .data import Tokenizer, left_pad
from .models import compute_position_id_with_mask
from .protocol import DataProto


def right_pad(seqs: List[list], length: int, pad_value, dtype=torch.long) -> torch.Tensor:
    out = torch.full((len(seqs), length), pad_value, dtype=dtype)
    for i, s in enumerate(seqs):
        s = s[:length]
        if s:
            out[i, :len(s)] = torch.tensor(s, dtype=dtype)
    return out


def build_rollout_batch(prompt_ids, response_ids, response_masks, response_logprobs,
                        prompt_length: int, response_length: int, pad_token_id: int) -> DataProto:
    """Pad per-sample token lists into verl's rollout tensors (agent_loop.py ``_postprocess``)."""
    prompts, prompt_attn = left_pad(prompt_ids, prompt_length, pad_token_id)
    responses = right_pad(response_ids, response_length, pad_token_id)
    response_attn = right_pad([[1] * len(r) for r in response_ids], response_length, 0)
    response_mask = right_pad(response_masks, response_length, 0) * response_attn
    attention_mask = torch.cat([prompt_attn, response_attn], dim=1)
    return DataProto.from_dict(tensors={
        "prompts": prompts, "responses": responses,
        "input_ids": torch.cat([prompts, responses], dim=1),
        "attention_mask": attention_mask,
        "position_ids": compute_position_id_with_mask(attention_mask),
        "response_mask": response_mask,
        "rollout_log_probs": right_pad(response_logprobs, response_length, 0.0, dtype=torch.float32),
    })


def strip_left_padding(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> List[List[int]]:
    return [row[m.bool()].tolist() for row, m in zip(input_ids, attention_mask)]


class Rollout:
    """Inference engine with its own weights, its own RNG, and no autograd."""

    def __init__(self, module: torch.nn.Module, config: RolloutConfig, tokenizer: Tokenizer, seed: int):
        self.module = module.eval().requires_grad_(False)
        self.config, self.tokenizer = config, tokenizer
        self.generator = torch.Generator().manual_seed(seed)
        self.weight_version: Optional[int] = None  # None = never synced from the actor

    @torch.no_grad()
    def sync_weights_from(self, actor_module: torch.nn.Module, global_steps: Optional[int] = None):
        """Copy the actor's current parameters into the rollout's own parameter tensors."""
        for (n_r, p_r), (n_a, p_a) in zip(self.module.named_parameters(), actor_module.named_parameters()):
            assert n_r == n_a, f"parameter mismatch {n_r} vs {n_a}"
            p_r.copy_(p_a)
        self.weight_version = global_steps

    @torch.no_grad()
    def generate_tokens(self, prompts: List[List[int]], max_new_tokens: Sequence[int], stop_token_ids,
                        do_sample: bool, temperature: float, top_k: int = -1):
        """Batched decoding. Each row stops at a stop token or after its own budget.

        Returns (tokens, logprobs) per row; logprobs are log-softmax(logits / temperature) of the
        chosen token -- what vLLM returns as ``rollout_log_probs``.
        """
        B = len(prompts)
        ids, mask = left_pad(prompts, max(len(p) for p in prompts), self.tokenizer.pad_token_id)
        pos = compute_position_id_with_mask(mask)
        budget = torch.tensor(list(max_new_tokens))
        stop = torch.tensor(sorted(stop_token_ids))
        active = budget > 0
        tokens, logprobs = [[] for _ in range(B)], [[] for _ in range(B)]
        for t in range(int(budget.max()) if B else 0):
            active &= budget > t
            if not active.any():
                break
            logits = self.module(ids, mask, pos)[:, -1, :] / temperature
            if top_k > 0:
                kth = torch.topk(logits, top_k, dim=-1).values[:, -1:]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            logp = F.log_softmax(logits, dim=-1)
            if do_sample:
                tok = torch.multinomial(logp.exp(), 1, generator=self.generator).squeeze(-1)
            else:
                tok = logits.argmax(dim=-1)
            lp = logp.gather(-1, tok[:, None]).squeeze(-1)
            tok = torch.where(active, tok, torch.full_like(tok, self.tokenizer.pad_token_id))
            for i in active.nonzero().flatten().tolist():
                tokens[i].append(int(tok[i]))
                logprobs[i].append(float(lp[i]))
            ids = torch.cat([ids, tok[:, None]], dim=1)
            mask = torch.cat([mask, active.long()[:, None]], dim=1)
            pos = torch.cat([pos, pos[:, -1:] + 1], dim=1)
            active &= ~torch.isin(tok, stop)
        return tokens, logprobs

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """prompts: left-padded input_ids/attention_mask -> verl rollout batch (one sample per row).

        verl repeats each prompt ``rollout.n`` times in the trainer before calling this; passing
        ``meta_info["n"] > 1`` repeats here instead (the older vLLM-rollout behaviour).
        """
        n = prompts.meta_info.get("n", 1)
        if n > 1:
            prompts = prompts.repeat(n, interleave=True)
        cfg = self.config
        do_sample = prompts.meta_info.get("do_sample", cfg.do_sample)
        temperature = prompts.meta_info.get("temperature", cfg.temperature)
        prompt_ids = strip_left_padding(prompts.batch["input_ids"], prompts.batch["attention_mask"])
        responses, logprobs = self.generate_tokens(prompt_ids, [cfg.response_length] * len(prompt_ids),
                                                   {self.tokenizer.eos_token_id}, do_sample, temperature, cfg.top_k)
        out = build_rollout_batch(prompt_ids, responses, [[1] * len(r) for r in responses], logprobs,
                                  prompts.batch["input_ids"].size(1), cfg.response_length,
                                  self.tokenizer.pad_token_id)
        out.meta_info["weight_version"] = self.weight_version
        return out

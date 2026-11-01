"""A tiny decoder-only transformer policy, a critic that shares its trunk, and the log-prob /
entropy helpers used by the actor.

``logprobs_from_logits`` and ``entropy_from_logits`` reproduce the formulas of
``verl/utils/torch_functional.py``; ``compute_position_id_with_mask`` is ``verl/utils/model.py``.
The models stand in for the HuggingFace ``AutoModelForCausalLM`` (actor / rollout / ref) and
``AutoModelForTokenClassification(num_labels=1)`` (critic) that verl loads.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


def logprobs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """log p(label) = logit[label] - logsumexp(logits)  (``logprobs_from_logits_v2`` for fp32)."""
    logits_labels = torch.gather(logits, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return logits_labels - torch.logsumexp(logits, dim=-1)


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """H = logsumexp(logits) - sum(softmax(logits) * logits)."""
    pd = F.softmax(logits, dim=-1)
    return torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)


def compute_position_id_with_mask(mask: torch.Tensor) -> torch.Tensor:
    """Left padding -> positions start at 0 on the first real token (pads get 0 too)."""
    return torch.clip(torch.cumsum(mask, dim=-1) - 1, min=0)


class Block(nn.Module):
    """Pre-LayerNorm transformer block: x + Attn(LN(x)); x + MLP(LN(x))."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_ff), nn.GELU(), nn.Linear(cfg.d_ff, cfg.d_model))

    def forward(self, x, allowed):  # allowed: [B, 1, T, T] bool
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.n_heads, D // self.n_heads).permute(2, 0, 3, 1, 4)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(D // self.n_heads)
        att = scores.masked_fill(~allowed, float("-inf")).softmax(dim=-1)
        x = x + self.proj((att @ v).transpose(1, 2).reshape(B, T, D))
        return x + self.mlp(self.ln2(x))


class TinyTransformer(nn.Module):
    """``forward(input_ids, attention_mask, position_ids) -> hidden [B, T, d_model]``.

    Attention is causal AND ignores padding keys (attention_mask == 0); every position may always
    attend to itself so that left-pad rows never produce NaNs.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_position, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

    def trunk(self, input_ids, attention_mask=None, position_ids=None):
        B, T = input_ids.shape
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if position_ids is None:
            position_ids = compute_position_id_with_mask(attention_mask)
        causal = torch.ones(T, T, dtype=torch.bool).tril()
        allowed = causal & attention_mask.bool()[:, None, :]          # [B, T(query), T(key)]
        allowed = (allowed | torch.eye(T, dtype=torch.bool)).unsqueeze(1)
        x = self.tok_emb(input_ids) + self.pos_emb(position_ids)
        for blk in self.blocks:
            x = blk(x, allowed)
        return self.ln_f(x)


class TinyPolicy(TinyTransformer):
    """Causal LM: ``forward(...) -> logits [B, T, vocab]``."""

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        with torch.random.fork_rng():  # deterministic init that leaves the global RNG untouched
            torch.manual_seed(cfg.seed)
            self.apply(_init_weights)

    def forward(self, input_ids, attention_mask=None, position_ids=None):
        return self.lm_head(self.trunk(input_ids, attention_mask, position_ids))


class TinyCritic(TinyTransformer):
    """Same trunk + scalar value head: ``forward(...) -> values [B, T]``."""

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        self.value_head = nn.Linear(cfg.d_model, 1)
        with torch.random.fork_rng():
            torch.manual_seed(cfg.seed + 1)
            self.apply(_init_weights)
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, input_ids, attention_mask=None, position_ids=None):
        return self.value_head(self.trunk(input_ids, attention_mask, position_ids)).squeeze(-1)

    def load_trunk_from_policy(self, policy_state_dict):
        """verl initialises the critic from the actor checkpoint; the value head stays fresh."""
        trunk = {k: v for k, v in policy_state_dict.items() if not k.startswith("lm_head")}
        missing, _ = self.load_state_dict(trunk, strict=False)
        assert set(missing) == {"value_head.weight", "value_head.bias"}, missing


def _init_weights(m):
    if isinstance(m, (nn.Linear, nn.Embedding)):
        nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

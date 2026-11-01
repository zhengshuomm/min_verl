"""The RL math: advantage estimators, policy / value losses, loss aggregation, KL estimators,
KL controllers. Mirrors ``verl/trainer/ppo/core_algos.py`` (same names, signatures and formulas)
plus ``masked_*`` helpers from ``verl/utils/torch_functional.py``.

Shapes: everything token-level is ``(bs, response_length)``; ``response_mask`` is 1 on tokens the
model generated (0 on padding and on tool-response tokens).
"""

from collections import defaultdict
from enum import Enum
from typing import Optional

import torch


# ----------------------------------------------------------------------------- masked helpers
def masked_sum(values, mask, axis=None):
    valid = torch.where(mask.bool(), values, 0.0) * mask  # NaNs outside the mask cannot leak in
    return valid.sum() if axis is None else valid.sum(axis)


def masked_mean(values, mask, axis=None):
    s = masked_sum(values, mask, axis)
    return s / ((mask.sum(axis) if axis is not None else mask.sum()) + 1e-8)


def masked_var(values, mask, unbiased=True):
    mean = masked_mean(values, mask)
    variance = masked_mean((values - mean) ** 2, mask)
    if unbiased:
        mask_sum = mask.sum()
        assert mask_sum > 1, "masked_var needs at least two unmasked elements"
        variance = variance * mask_sum / (mask_sum - 1)
    return variance


def masked_whiten(values, mask, shift_mean=True):
    mean, var = masked_mean(values, mask), masked_var(values, mask)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    return whitened if shift_mean else whitened + mean


# ----------------------------------------------------------------------------- registry
class AdvantageEstimator(str, Enum):
    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    RLOO = "rloo"


ADV_ESTIMATOR_REGISTRY = {}


def register_adv_est(name_or_enum):
    def decorator(fn):
        name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
        ADV_ESTIMATOR_REGISTRY[name] = fn
        return fn
    return decorator


def get_adv_estimator_fn(name_or_enum):
    name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
    if name not in ADV_ESTIMATOR_REGISTRY:
        raise ValueError(f"Unknown advantage estimator simply: {name}")
    return ADV_ESTIMATOR_REGISTRY[name]


# ----------------------------------------------------------------------------- KL controllers
class AdaptiveKLController:
    """Ziegler et al. 2019: beta *= 1 + clip(KL/target - 1, -0.2, 0.2) * n_steps / horizon."""

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value, self.target, self.horizon = init_kl_coef, target_kl, horizon

    def update(self, current_kl, n_steps):
        proportional_error = max(-0.2, min(0.2, current_kl / self.target - 1))
        self.value *= 1 + proportional_error * n_steps / self.horizon


class FixedKLController:
    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def get_kl_controller(kl_ctrl):
    if kl_ctrl.type == "fixed":
        return FixedKLController(kl_coef=kl_ctrl.kl_coef)
    if kl_ctrl.type == "adaptive":
        assert kl_ctrl.horizon > 0, f"horizon must be larger than 0. Got {kl_ctrl.horizon}"
        return AdaptiveKLController(kl_ctrl.kl_coef, kl_ctrl.target_kl, kl_ctrl.horizon)
    raise NotImplementedError(kl_ctrl.type)


# ----------------------------------------------------------------------------- advantage estimators
@register_adv_est(AdvantageEstimator.GAE)
def compute_gae_advantage_return(token_level_rewards, values, response_mask, gamma, lam):
    """GAE, right to left. Masked (tool / pad) tokens are skipped: they neither produce a TD error
    nor break the chain. Advantages are whitened over the batch; returns = raw advantages + values."""
    with torch.no_grad():
        nextvalues, lastgaelam, advantages_reversed = 0, 0, []
        for t in reversed(range(token_level_rewards.shape[-1])):
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam_ = delta + gamma * lam * lastgaelam
            nextvalues = values[:, t] * response_mask[:, t] + (1 - response_mask[:, t]) * nextvalues
            lastgaelam = lastgaelam_ * response_mask[:, t] + (1 - response_mask[:, t]) * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        advantages = masked_whiten(advantages, response_mask)
    return advantages, returns


@register_adv_est(AdvantageEstimator.GRPO)
def compute_grpo_outcome_advantage(token_level_rewards, response_mask, index, epsilon: float = 1e-6,
                                   norm_adv_by_std_in_grpo: bool = True, config=None):
    """Group-relative: A_i = (r_i - mean_group) / (std_group + eps), std is torch.std (unbiased).
    norm_adv_by_std_in_grpo=False gives Dr.GRPO (no std division). Groups of size 1 get A = r."""
    scores = token_level_rewards.sum(dim=-1)
    id2score, id2mean, id2std = defaultdict(list), {}, {}
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx, group in id2score.items():
            if len(group) == 1:
                id2mean[idx], id2std[idx] = torch.tensor(0.0), torch.tensor(1.0)
            else:
                id2mean[idx], id2std[idx] = torch.mean(torch.stack(group)), torch.std(torch.stack(group))
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(-1) * response_mask
    return scores, scores


@register_adv_est(AdvantageEstimator.RLOO)
def compute_rloo_outcome_advantage(token_level_rewards, response_mask, index, epsilon: float = 1e-6,
                                   config=None, **kwargs):
    """Leave-one-out baseline: A_i = r_i - mean_{j != i} r_j = (r_i - mean) * n / (n - 1)."""
    scores = token_level_rewards.sum(dim=-1)
    id2score, id2mean = defaultdict(list), {}
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx, group in id2score.items():
            id2mean[idx] = torch.tensor(0.0) if len(group) == 1 else torch.mean(torch.stack(group))
        for i in range(bsz):
            n = len(id2score[index[i]])
            if n > 1:
                scores[i] = scores[i] * n / (n - 1) - id2mean[index[i]] * n / (n - 1)
        scores = scores.unsqueeze(-1) * response_mask
    return scores, scores


@register_adv_est(AdvantageEstimator.REINFORCE_PLUS_PLUS)
def compute_reinforce_plus_plus_outcome_advantage(token_level_rewards, response_mask, config=None, **kwargs):
    """Discounted return-to-go per token (masked tokens carry the return through), then whitened
    over the whole batch (no group baseline, no critic)."""
    gamma = config.gamma
    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0
        for t in reversed(range(token_level_rewards.shape[1])):
            new_running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = new_running_return * response_mask[:, t]
            running_return = new_running_return * response_mask[:, t] + running_return * (1 - response_mask[:, t])
        advantages = masked_whiten(returns, response_mask) * response_mask
    return advantages, returns


# ----------------------------------------------------------------------------- losses
def agg_loss(loss_mat, loss_mask, loss_agg_mode: str, dp_size: int = 1, batch_num_tokens: Optional[int] = None,
             global_batch_size: Optional[int] = None, loss_scale_factor: Optional[int] = None):
    """Reduce a (bs, T) loss matrix to a scalar.

    * token-mean               sum(loss * mask) / #tokens                (every token weighs the same)
    * seq-mean-token-sum       mean over sequences of the per-sequence token SUM
    * seq-mean-token-mean      mean over sequences of the per-sequence token MEAN (every sequence the same)
    * seq-mean-token-sum-norm  seq-mean-token-sum / loss_scale_factor (default T, a constant: Dr.GRPO)

    When ``batch_num_tokens`` / ``global_batch_size`` describe the GLOBAL mini-batch and the result
    is multiplied by ``dp_size``, summing over micro-batches and averaging gradients over DP ranks
    gives exactly the loss of the full mini-batch -- independent of how it was split.
    """
    if loss_agg_mode == "token-mean":
        if batch_num_tokens is None:
            assert dp_size == 1, "(global) batch_num_tokens is required when dp_size > 1"
            batch_num_tokens = loss_mask.sum()
        return masked_sum(loss_mat, loss_mask) / batch_num_tokens * dp_size
    if loss_agg_mode == "token-sum":
        return masked_sum(loss_mat, loss_mask) * dp_size
    if loss_agg_mode in ("seq-mean-token-sum", "seq-mean-token-sum-norm"):
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        seq_mask = (torch.sum(loss_mask, dim=-1) > 0).float()
        if global_batch_size is None:
            assert dp_size == 1, "global_batch_size is required when dp_size > 1"
            global_batch_size = seq_mask.sum()
        loss = masked_sum(seq_losses, seq_mask) / global_batch_size * dp_size
        if loss_agg_mode == "seq-mean-token-sum-norm":
            loss = loss / (loss_scale_factor if loss_scale_factor is not None else loss_mask.shape[-1])
        return loss
    if loss_agg_mode == "seq-mean-token-mean":
        seq_tokens = torch.sum(loss_mask, dim=-1)
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / (seq_tokens + 1e-8)
        seq_mask = (seq_tokens > 0).float()
        if global_batch_size is None:
            assert dp_size == 1, "global_batch_size is required when dp_size > 1"
            global_batch_size = seq_mask.sum()
        return masked_sum(seq_losses, seq_mask) / global_batch_size * dp_size
    raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")


def compute_policy_loss(old_log_prob, log_prob, advantages, response_mask, cliprange=None, cliprange_low=None,
                        cliprange_high=None, clip_ratio_c=3.0, loss_agg_mode: str = "token-mean", **global_batch_info):
    """PPO clipped surrogate with dual-clip (verl ``compute_policy_loss`` / ``compute_policy_loss_vanilla``).

    ratio = exp(log_prob - old_log_prob)
    A >= 0: max(-A*ratio, -A*clip(ratio, 1-eps_low, 1+eps_high))
    A <  0: min(that, -A*clip_ratio_c)        (dual clip: bounds the loss when ratio explodes)
    Returns pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower.
    """
    assert clip_ratio_c > 1.0, f"clip_ratio_c should be greater than 1.0, got {clip_ratio_c}"
    cliprange_low = cliprange if cliprange_low is None else cliprange_low
    cliprange_high = cliprange if cliprange_high is None else cliprange_high
    negative_approx_kl = torch.clamp(log_prob - old_log_prob, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = masked_mean(-negative_approx_kl, response_mask)
    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    pg_clipfrac = masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)
    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = masked_mean(torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask)
    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    pg_loss = agg_loss(pg_losses, response_mask, loss_agg_mode, **global_batch_info)
    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


def compute_value_loss(vpreds, returns, values, response_mask, cliprange_value: float,
                       loss_agg_mode: str = "token-mean", **global_batch_info):
    """Clipped value loss: 0.5 * agg(max((v - R)^2, (clip(v, v_old +- c) - R)^2))."""
    vpredclipped = torch.max(torch.min(vpreds, values + cliprange_value), values - cliprange_value)
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    vf_loss = 0.5 * agg_loss(torch.max(vf_losses1, vf_losses2), response_mask, loss_agg_mode, **global_batch_info)
    vf_clipfrac = masked_mean(torch.gt(vf_losses2, vf_losses1).float(), response_mask)
    return vf_loss, vf_clipfrac


def kl_penalty_forward(logprob, ref_logprob, kl_penalty):
    """Per-token KL estimators (http://joschu.net/blog/kl-approx.html), with x = log p - log p_ref:
    kl/k1 = x;  abs = |x|;  mse/k2 = x^2 / 2;  low_var_kl/k3 = exp(-x) + x - 1 (clamped to [-10, 10])."""
    if kl_penalty in ("kl", "k1"):
        return logprob - ref_logprob
    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()
    if kl_penalty in ("mse", "k2"):
        return 0.5 * (logprob - ref_logprob).square()
    if kl_penalty in ("low_var_kl", "k3"):
        kl = torch.clamp(ref_logprob - logprob, min=-20, max=20)
        kld = (torch.exp(kl) - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)
    raise NotImplementedError(kl_penalty)


def kl_penalty(logprob, ref_logprob, kl_penalty):
    """``"k3+"`` etc.: forward value of k3 but the (unbiased) gradient of k2 (straight-through)."""
    base = kl_penalty[:-1] if kl_penalty.endswith("+") else kl_penalty
    forward_score = kl_penalty_forward(logprob, ref_logprob, base)
    if not kl_penalty.endswith("+") or base in ("mse", "k2"):
        return forward_score
    backward_score = 0.5 * (logprob - ref_logprob).square()
    return backward_score - backward_score.detach() + forward_score.detach()

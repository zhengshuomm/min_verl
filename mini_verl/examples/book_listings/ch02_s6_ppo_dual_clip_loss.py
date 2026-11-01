#!/usr/bin/env python3
"""Standalone book listing — Section 2.6: Inside verl: A Line-by-Line Walkthrough of `compute_policy_loss_vanilla`
Pinned reference: verl v0.9.1 (commit 1876b06).
Expected stdout:
    [-1.2, -0.5, 0.8, 3.0]
    0.525 0.5 0.25 -0.157152
    1.025
"""

import torch

def vanilla_loss(old_log_prob, log_prob, advantages, response_mask,
                 cliprange_low=0.2, cliprange_high=0.2,
                 clip_ratio_c=3.0):
    negative_approx_kl = torch.clamp(
        log_prob - old_log_prob, min=-20.0, max=20.0
    )
    ratio = torch.exp(negative_approx_kl)
    n = response_mask.sum()
    ppo_kl = (-negative_approx_kl * response_mask).sum() / n
    pg_losses1 = -advantages * ratio
    clipped = torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
    pg_losses2 = -advantages * clipped
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    is_clipped = torch.gt(pg_losses2, pg_losses1).float()
    pg_clipfrac = (is_clipped * response_mask).sum() / n
    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    lower = (torch.gt(clip_pg_losses1, pg_losses3).float()
             * (advantages < 0).float())
    pg_clipfrac_lower = (lower * response_mask).sum() / n
    pg_losses = torch.where(
        advantages < 0, clip_pg_losses2, clip_pg_losses1
    )
    pg_loss = (pg_losses * response_mask).sum() / n  # token-mean
    return pg_losses, pg_loss, pg_clipfrac, pg_clipfrac_lower, ppo_kl

A = torch.tensor([[1.0, 1.0, -1.0, -1.0]])
r = torch.tensor([[1.5, 0.5, 0.5, 5.0]])
old, mask = torch.zeros(1, 4), torch.ones(1, 4)
per_tok, loss, cf, cf_low, kl = vanilla_loss(old, r.log(), A, mask)
no_dual = vanilla_loss(old, r.log(), A, mask, clip_ratio_c=1e9)[1]
toks = [round(v, 4) for v in per_tok[0].tolist()]
print(toks)
print(round(loss.item(), 6), cf.item(), cf_low.item(),
      round(kl.item(), 6))
print(round(no_dual.item(), 6))

# Verification assertions
assert toks == [-1.2, -0.5, 0.8, 3.0]
assert abs(loss.item() - 0.525) < 1e-5
assert cf.item() == 0.5
assert cf_low.item() == 0.25
assert abs(kl.item() + 0.157152) < 1e-5
assert abs(no_dual.item() - 1.025) < 1e-5

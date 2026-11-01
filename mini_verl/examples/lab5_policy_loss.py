"""Lab 5 -- The PPO policy loss by hand: clipping, dual-clip, and loss aggregation modes.

Run:  python3 -m mini_verl.examples.lab5_policy_loss

Four tokens with advantages A = [+1, +1, -1, -1] and probability ratios r = pi/pi_old =
[1.5, 0.5, 0.5, 5.0]; clip_ratio = 0.2, clip_ratio_c = 3.0 (verl defaults).
"""

import math

import torch

from mini_verl.core_algos import agg_loss, compute_policy_loss


def main():
    A = torch.tensor([[1.0, 1.0, -1.0, -1.0]])
    ratio = torch.tensor([[1.5, 0.5, 0.5, 5.0]])
    old_log_prob, log_prob, mask = torch.zeros(1, 4), ratio.log(), torch.ones(1, 4)

    print("token |  A  | ratio | -A*r   | -A*clip(r,0.8,1.2) | max (PPO) | dual-clip min(.., -A*3)")
    l1 = -A * ratio
    l2 = -A * ratio.clamp(0.8, 1.2)
    ppo = torch.maximum(l1, l2)
    dual = torch.where(A < 0, torch.minimum(ppo, -A * 3.0), ppo)
    for t in range(4):
        print(f"  {t}   | {A[0, t]:+.0f}  | {ratio[0, t]:.1f}   | {l1[0, t]:+.2f}  | {l2[0, t]:+.2f}"
              f"              | {ppo[0, t]:+.2f}     | {dual[0, t]:+.2f}")

    pg_loss, clipfrac, ppo_kl, clipfrac_lower = compute_policy_loss(old_log_prob, log_prob, A, mask, cliprange=0.2,
                                                                    clip_ratio_c=3.0)
    print(f"\ncompute_policy_loss (dual-clip c=3.0): pg_loss={pg_loss.item():.6f}  pg_clipfrac={clipfrac.item():.4f}  "
          f"pg_clipfrac_lower={clipfrac_lower.item():.4f}  ppo_kl={ppo_kl.item():.6f}")
    no_dual, *_ = compute_policy_loss(old_log_prob, log_prob, A, mask, cliprange=0.2, clip_ratio_c=1e9)
    print(f"without dual-clip (c=1e9):             pg_loss={no_dual.item():.6f}   <- token 3 alone contributes 5/4")
    hi, cf_hi, _, _ = compute_policy_loss(old_log_prob, log_prob, A, mask, cliprange_low=0.2, cliprange_high=0.28)
    print(f"clip-higher (low=0.2, high=0.28):      pg_loss={hi.item():.6f}  pg_clipfrac={cf_hi.item():.4f}")
    print(f"ppo_kl = mean(-log r) = -(ln1.5 + 2 ln0.5 + ln5)/4 = "
          f"{-(math.log(1.5) + 2 * math.log(0.5) + math.log(5.0)) / 4:.6f}")

    print("\nloss aggregation on two responses (lengths 3 and 1):")
    loss_mat = torch.tensor([[1.0, 2.0, 3.0, 0.0], [4.0, 0.0, 0.0, 0.0]])
    loss_mask = torch.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    print("  loss_mat =", loss_mat.tolist(), " mask =", loss_mask.tolist())
    for mode, how in (("token-mean", "(1+2+3+4)/4"), ("seq-mean-token-sum", "(6+4)/2"),
                      ("seq-mean-token-mean", "(6/3 + 4/1)/2"), ("seq-mean-token-sum-norm", "(6+4)/2/T, T=4")):
        print(f"  {mode:<24} = {agg_loss(loss_mat, loss_mask, mode).item():.4f}   {how}")

    print("\nsame batch split over 2 DP ranks (one row each), then gradients averaged (all-reduce mean):")
    parts = [agg_loss(loss_mat[i:i + 1], loss_mask[i:i + 1], "token-mean", dp_size=2, batch_num_tokens=4).item()
             for i in (0, 1)]
    naive = [agg_loss(loss_mat[i:i + 1], loss_mask[i:i + 1], "token-mean").item() for i in (0, 1)]
    print(f"  token-mean, batch_num_tokens=4: ranks {parts} -> all-reduce mean {sum(parts) / 2:.4f} (== full batch)")
    print(f"  token-mean, local #tokens     : ranks {naive} -> all-reduce mean {sum(naive) / 2:.4f} (wrong)")


if __name__ == "__main__":
    main()

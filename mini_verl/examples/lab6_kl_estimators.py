"""Lab 6 -- KL estimators k1 / k2 / k3 ("kl", "mse", "low_var_kl") and why GRPO uses k3.

Run:  python3 -m mini_verl.examples.lab6_kl_estimators

Part 1 evaluates each per-token estimator at a few log-ratios x = log pi - log pi_ref.
Part 2 samples tokens from pi over a 5-token vocabulary and compares each estimator's
Monte-Carlo mean and standard deviation with the exact KL(pi || pi_ref).
"""

import math

import torch

from mini_verl.core_algos import kl_penalty


def main():
    print("x = log pi - log pi_ref |   k1 (kl)  |  abs     |  k2 (mse) |  k3 (low_var_kl)")
    for x in (-2.0, -0.5, -0.1, 0.0, 0.1, 0.5, 2.0):
        lp, ref = torch.tensor([x]), torch.tensor([0.0])
        vals = [kl_penalty(lp, ref, k).item() for k in ("kl", "abs", "mse", "low_var_kl")]
        print(f"        {x:+.1f}            | {vals[0]:+.6f} | {vals[1]:.6f} | {vals[2]:.6f}  | {vals[3]:.6f}")
    print("k3 = exp(-x) + x - 1 >= 0 always (k1 can be negative per token); k3 is clamped to [-10, 10]")

    pi = torch.tensor([0.40, 0.30, 0.15, 0.10, 0.05])
    ref = torch.tensor([0.25, 0.25, 0.20, 0.20, 0.10])
    true_kl = float((pi * (pi / ref).log()).sum())
    g = torch.Generator().manual_seed(0)
    tokens = torch.multinomial(pi, 100_000, replacement=True, generator=g)
    lp, lref = pi.log()[tokens], ref.log()[tokens]
    print(f"\nKL(pi || pi_ref) exact = {true_kl:.6f}   (100,000 samples from pi, seed 0)")
    print("estimator | MC mean  | per-sample std | bias")
    for name in ("kl", "mse", "low_var_kl"):
        est = kl_penalty(lp, lref, name)
        print(f"{name:<10}| {est.mean().item():.6f} | {est.std().item():.6f}       | {est.mean().item() - true_kl:+.6f}")
    print("k1 and k3 are unbiased; k3 has the smaller variance; k2 is biased (but its gradient is the right one).")
    print(f"(check: exp is applied to log-ratios of at most ln(0.40/0.25) = {math.log(0.4 / 0.25):.4f})")


if __name__ == "__main__":
    main()

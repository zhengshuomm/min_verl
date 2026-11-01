#!/usr/bin/env python3
"""Standalone book listing — Section 3.2: GRPO: Let Responses to the Same Prompt Be Each Other's Baseline
Pinned reference: verl v0.9.1 (commit 1876b06).
Expected stdout:
    [0.866024, -0.866024, -0.866024, 0.866024]
    [0.666667, -0.666667, -0.666667, 0.666667] [0.0, 0.0, 0.0, 0.0] 3.75
"""

import torch

def grpo_adv(scores, eps=1e-6, norm_by_std=True):
    s = torch.tensor(scores, dtype=torch.float32)
    if len(s) == 1:              # singleton group -> mean 0, std 1
        mean, std = torch.tensor(0.0), torch.tensor(1.0)
    else:                        # torch.std is unbiased (div by G-1)
        mean, std = s.mean(), s.std()
    if not norm_by_std:          # Dr. GRPO: norm_adv_by_std_in_grpo
        return s - mean
    return (s - mean) / (std + eps)

def rloo_adv(scores):
    s = torch.tensor(scores, dtype=torch.float32)
    G = len(s)
    # verl form; equals r_i - mean(other G-1 rewards)
    return s * G / (G - 1) - s.mean() * G / (G - 1)

A = grpo_adv([1, 0, 0, 1])
D = grpo_adv([1, 0, 0, 1], norm_by_std=False)
R = rloo_adv([1, 0, 0, 1])
Z = grpo_adv([1, 1, 1, 1])
S = grpo_adv([1])
H = grpo_adv([1] + [0] * 15)     # 1 correct out of 16
print([round(x, 6) for x in A.tolist()])
print([round(x, 6) for x in R.tolist()], Z.tolist(),
      round(H[0].item(), 4))

# Verification assertions
assert abs(A[0].item() - 0.866024) < 1e-5
assert abs(A[1].item() + 0.866024) < 1e-5
assert abs(A.sum().item()) < 1e-6
assert abs(D[0].item() - 0.5) < 1e-6
assert abs(D[2].item() + 0.5) < 1e-6
assert abs(R[0].item() - 2/3) < 1e-6
assert abs(R[1].item() + 2/3) < 1e-6
assert Z.abs().max().item() == 0.0
assert abs(S[0].item() - 1.0) < 1e-5
assert abs(H[0].item() - 3.75) < 1e-4

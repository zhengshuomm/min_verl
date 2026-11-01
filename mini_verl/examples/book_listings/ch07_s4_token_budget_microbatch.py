#!/usr/bin/env python3
"""Standalone book listing — Section 7.4: Padding Removal and Dynamic Batch Size: Paying Only for Valid Tokens
Pinned reference: verl v0.9.1 (commit 1876b06).
Expected stdout:
    3 [5872, 5850, 5950] 8
"""

import math

def split_by_tokens(lens, budget):
    """Greedy token-budget bin packing; increment k on overflow."""
    assert max(lens) <= budget, "Budget smaller than longest sequence"
    k = min(len(lens), math.ceil(sum(lens) / budget))
    while True:
        bins, load = [[] for _ in range(k)], [0] * k
        for i in sorted(range(len(lens)), key=lambda i: -lens[i]):
            j = load.index(min(load))  # lightest chunk so far
            bins[j].append(i)
            load[j] += lens[i]
        if max(load) <= budget:
            return bins, load
        k += 1

# True lengths (prompt + response) of 16 sequences on one GPU
lens = [3072, 2900, 2400, 1800, 1500, 1200, 900, 800,
        700, 600, 500, 400, 300, 250, 200, 150]
budget = 2 * 3072                    # recommended lower bound in docs
bins, load = split_by_tokens(lens, budget)
fixed_mbs = budget // max(lens)      # safe cap under fixed batch size
n_fixed = math.ceil(len(lens) / fixed_mbs)
print(len(bins), load, n_fixed)

# Verification assertions
assert sum(lens) == 17672
assert len(bins) == 3 and load == [5872, 5850, 5950]
assert max(load) <= budget
assert sorted(i for b in bins for i in b) == list(range(16))
assert fixed_mbs == 2 and n_fixed == 8
assert split_by_tokens(lens, 24576)[1] == [17672]

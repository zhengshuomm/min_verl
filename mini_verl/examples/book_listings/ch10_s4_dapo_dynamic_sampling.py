#!/usr/bin/env python3
"""Standalone book listing — Section 10.4: Dynamic Sampling: Replacing Zero-Gradient Groups
Pinned reference: verl v0.9.1 (commit 1876b06).
Expected stdout:
    ['a', 'd'] {'b': 1, 'c': 0}
    0.5905 0.1853
    0.75
"""

import statistics

def drop_uniform_groups(uids, rewards):
    groups = {}
    for uid, r in zip(uids, rewards):
        groups.setdefault(uid, []).append(r)
    dropped = {u: rs[0] for u, rs in groups.items()
               if len(rs) > 1 and statistics.pstdev(rs) == 0}
    kept = [u for u in groups if u not in dropped]
    return kept, dropped

def all_same_prob(p, G):
    return p ** G + (1 - p) ** G

def effective_scale(kept_tokens, total_tokens):
    # token-mean: zero-advantage tokens still sit in the denominator
    return kept_tokens / total_tokens

uids = ["a"] * 4 + ["b"] * 4 + ["c"] * 4 + ["d"] * 4
rewards = [1, 0, 0, 1,  1, 1, 1, 1,  0, 0, 0, 0,  0, 0, 1, 0]
kept, dropped = drop_uniform_groups(uids, rewards)
print(kept, dropped)
print(round(all_same_prob(0.9, 5), 4), round(all_same_prob(0.9, 16), 4))
print(effective_scale(768 * 5, 1024 * 5))

# Verification assertions
assert kept == ['a', 'd']
assert dropped == {'b': 1, 'c': 0}
assert drop_uniform_groups(['s'], [1]) == (['s'], {})
assert abs(all_same_prob(0.5, 5) - 0.0625) < 1e-12
assert abs(all_same_prob(0.9, 5) - 0.5905) < 1e-4
assert abs(all_same_prob(0.9, 16) - 0.1853) < 1e-4
assert effective_scale(768 * 5, 1024 * 5) == 0.75

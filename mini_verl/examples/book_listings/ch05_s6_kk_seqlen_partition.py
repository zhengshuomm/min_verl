#!/usr/bin/env python3
"""Standalone book listing — Section 5.6: balance_batch: Keeping One Slow GPU from Stalling the Other Seven
Pinned reference: verl v0.9.1 (commit 1876b06).
Expected stdout:
    [11400, 1800] [6600, 6600] 1.75
"""

import heapq

def workload(L):  # verl/utils/seqlen_balancing.py: calculate_workload
    return 24576 * L + L * L

def kk_partition(vals, k):
    assert len(vals) % k == 0
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    heap = []
    for off in range(0, len(order), k):  # k neighbours -> k sets
        sets = sorted(
            ([vals[i], [i]] for i in order[off:off + k]), reverse=True
        )
        heapq.heappush(heap, (-(sets[0][0] - sets[-1][0]), off, sets))
    while len(heap) > 1:  # merge two states with largest spread
        _, off, a = heapq.heappop(heap)
        _, _, b = heapq.heappop(heap)
        merged = sorted(
            ([a[j][0] + b[k - 1 - j][0], a[j][1] + b[k - 1 - j][1]]
             for j in range(k)),
            reverse=True,
        )  # pair heaviest with lightest
        spread = merged[0][0] - merged[-1][0]
        heapq.heappush(heap, (-spread, off, merged))
    return [sorted(s[1]) for s in heap[0][2]]

lens = [3000, 2900, 2800, 2700, 600, 500, 400, 300]  # prompt + resp
w = [workload(L) for L in lens]
naive = [[0, 1, 2, 3], [4, 5, 6, 7]]  # contiguous dispatch chunks
parts = kk_partition(w, 2)

def toks(p):
    return sum(lens[i] for i in p)

def load(p):
    return sum(w[i] for i in p)

slowdown = max(load(p) for p in naive) / max(load(p) for p in parts)
print([toks(p) for p in naive], [toks(p) for p in parts],
      round(slowdown, 2))

# Verification assertions
assert [toks(p) for p in naive] == [11400, 1800]
assert sorted(toks(p) for p in parts) == [6600, 6600]
assert sorted(parts) == [[0, 3, 5, 6], [1, 2, 4, 7]]
assert load(parts[0]) == load(parts[1]) == 178901600
assert round(slowdown, 2) == 1.75
assert workload(3072) == 84934656

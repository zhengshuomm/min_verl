#!/usr/bin/env python3
"""Standalone book listing — Section 6.5: Weight Synchronization: Tensor by Tensor, Bucket by Bucket, over CUDA IPC
Pinned reference: verl v0.9.1 (commit 1876b06).
Expected stdout:
    bf16: 8 buckets, 0 direct transfers; fp32: 14 buckets, 2 direct transfers (embed_tokens and lm_head)
"""

def pack_buckets(sizes, bucket, elem):
    """BucketedWeightSender: (bucket byte sizes, direct count)."""
    buckets, direct, offset, count = [], 0, 0, 0
    for nbytes in sizes:
        offset = (offset + elem - 1) // elem * elem  # align to elem
        if offset + nbytes > bucket and count > 0:   # flush bucket
            buckets.append(offset)
            offset, count = 0, 0
        if offset + nbytes > bucket:                 # > bucket: direct
            direct += 1
            continue
        offset += nbytes
        count += 1
    if count > 0:
        buckets.append(offset)
    return buckets, direct

def qwen3_8b_numels(H=4096, L=36, V=151936, I=12288,
                    NH=32, KV=8, D=128):
    yield V * H                                      # embed_tokens
    for _ in range(L):                               # 11 tensors/layer
        yield from [NH * D * H, KV * D * H, KV * D * H, H * NH * D,
                    D, D, I * H, I * H, H * I, H, H]
    yield H                                          # final norm
    yield V * H                                      # untied lm_head

B = 2048 << 20                                       # 2048 MB bucket
numels = list(qwen3_8b_numels())
bf16, d16 = pack_buckets([n * 2 for n in numels], B, 2)
fp32, d32 = pack_buckets([n * 4 for n in numels], B, 4)
toy = pack_buckets([3, 3, 3, 9], 8, 1)
print(
    f"bf16: {len(bf16)} buckets, {d16} direct transfers; "
    f"fp32: {len(fp32)} buckets, {d32} direct transfers "
    f"(embed_tokens and lm_head)"
)

# Verification assertions
assert sum(numels) == 8190735360
assert len(numels) == 399
assert toy == ([6, 3], 1)
assert (len(bf16), d16) == (8, 0)
assert (len(fp32), d32) == (14, 2)
assert max(fp32) <= B and 151936 * 4096 * 4 > B

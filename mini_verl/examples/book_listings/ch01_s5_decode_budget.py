#!/usr/bin/env python3
"""Standalone book listing — Section 1.5: The Time Ledger: Why Generation Eats Most of the Step
Pinned reference: verl v0.9.1 (commit 1876b06).
Expected stdout:
    539947 175 1280 7.3 2.44 295
    140 9.1
"""

def decode_budget(P=8_190_735_360, tp=2, hbm=80e9, util=0.6,
                  layers=36, kv_heads=8, head_dim=128,
                  prompt_len=1024, resp_len=2048,
                  n_gpus=8, responses=5120, bw=3.35e12, peak=989e12):
    w_per_gpu = 2 * P / tp                      # bf16 weights per GPU
    kv_per_tok = 2 * layers * (kv_heads // tp) * head_dim * 2
    kv_tokens = int((util * hbm - w_per_gpu) // kv_per_tok)
    full_seqs = kv_tokens // (prompt_len + resp_len)  # concurrent max
    per_replica = responses // (n_gpus // tp)   # responses per replica
    rounds = per_replica / full_seqs            # full-length waves
    step_ms = w_per_gpu / bw * 1e3              # weight-read floor (ms)
    ridge = peak / bw                           # roofline ridge batch
    return kv_tokens, full_seqs, per_replica, rounds, step_ms, ridge


(kv_tokens, full_seqs, per_replica,
 rounds, step_ms, ridge) = decode_budget()
print(kv_tokens, full_seqs, per_replica,
      round(rounds, 1), round(step_ms, 2), round(ridge))
low = decode_budget(util=0.5)                   # verl default util=0.5
print(low[1], round(low[3], 1))

# Verification assertions
assert kv_tokens == 539947
assert full_seqs == 175
assert per_replica == 1280
assert round(rounds, 1) == 7.3
assert round(step_ms, 2) == 2.44
assert round(ridge) == 295
assert low[1] == 140 and round(low[3], 1) == 9.1

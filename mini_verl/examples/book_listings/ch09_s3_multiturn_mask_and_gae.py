#!/usr/bin/env python3
"""Standalone book listing — Section 9.3: Hand Calculation on Lab 9: How the Mask Flows Through GRPO, GAE, and token-mean
Pinned reference: verl v0.9.1 (commit 1876b06).
Expected stdout:
    mask: [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0]
    adv : [0.737, 0.724, 0.709, 0.694, 0.678, 0.661, 0.643, 0.413, 0.413, 0.413, 0.413, 0.413, 0.382, 0.35, 0.0, 0.0]
    denominator: 10 | token_mean: 1.0 | with attn: 29.0
"""

def merge_turns(turns):
    """turns: [(role, tokens)], where role is 'model' or 'tool'"""
    ids, mask = [], []
    for role, toks in turns:
        ids += toks
        mask += [1 if role == "model" else 0] * len(toks)
    return ids, mask

def right_pad(ids, mask, length, pad="<pad>"):
    n = length - len(ids)
    return ids + [pad] * n, mask + [0] * n

def token_mean(loss, mask):
    return sum(l * m for l, m in zip(loss, mask)) / sum(mask)

def gae(rewards, values, mask, gamma=1.0, lam=0.95):
    adv, nextv, last = [0.0] * len(rewards), 0.0, 0.0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * nextv - values[t]
        cand = delta + gamma * lam * last
        nextv = values[t] if mask[t] else nextv   # mask 0: skip value
        last = cand if mask[t] else last          # mask 0: skip TD err
        adv[t] = last
    return adv

turns = [("model", ["<c>", "2", "3", "+", "4", "8", "</c>"]),
         ("tool", ["<r>", "7", "1", "</r>"]),
         ("model", ["7", "1", "<eos>"])]
ids, mask = right_pad(*merge_turns(turns), length=16)
loss = [1.0 if m else 99.0 for m in mask]         # huge loss on tools
rewards = [0.0] * 16
rewards[13] = 1.0                                 # last valid token
values = [0.05 * t for t in range(16)]
adv = gae(rewards, values, mask)
keep = [t for t in range(16) if mask[t]]          # excise non-model
adv_cut = gae(
    [rewards[t] for t in keep],
    [values[t] for t in keep],
    [1] * len(keep),
)
attn = [0 if t == "<pad>" else 1 for t in ids]    # tool slots are 1
tm, ta = token_mean(loss, mask), round(token_mean(loss, attn), 1)
print("mask:", mask)
print("adv :", [round(a, 3) for a in adv])
print("denominator:", sum(mask), "| token_mean:", tm,
      "| with attn:", ta)

# Verification assertions
assert mask == [1] * 7 + [0] * 4 + [1] * 3 + [0] * 2
assert mask[7:11] == [0, 0, 0, 0]
assert sum(mask) == 10 and len(ids) == len(mask) == 16
assert token_mean(loss, mask) == 1.0
assert token_mean(loss, attn) == (10 + 4 * 99) / 14
assert adv[7] == adv[10] == adv[11]
assert [adv[t] for t in keep] == adv_cut
assert all(abs(adv[t] - a) < 1e-12 for t, a in zip(keep, adv_cut))

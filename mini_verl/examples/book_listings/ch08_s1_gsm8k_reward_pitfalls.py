#!/usr/bin/env python3
"""Standalone book listing — Section 8.1: GSM8K: From One Parquet Row to a 0 or 1
Pinned reference: verl v0.9.1 (commit 1876b06).
Expected stdout:
    good      strict=1.0  flexible=1.0  naive=1.0
    wrong     strict=0.0  flexible=0.0  naive=0.0
    no_tag    strict=0.0  flexible=1.0  naive=1.0
    as_float  strict=0.0  flexible=0.0  naive=1.0
    period    strict=0.0  flexible=0.0  naive=1.0
    buried    strict=0.0  flexible=0.0  naive=1.0
    spray     strict=0.0  flexible=0.0  naive=1.0
    comma     strict=1.0  flexible=0.0  naive=0.0
    dollar    strict=0.0  flexible=1.0  naive=1.0
    int ground truth 72: strict=0.0
"""

import re

CLIP = 300  # verl: _SOLUTION_CLIP_CHARS

def extract_solution(text, method="strict"):
    text = text[-CLIP:]                    # inspect last 300 chars
    if method == "strict":
        found = re.findall(r"#### (\-?[0-9\.\,]+)", text)
        # verl strips commas then $ ($ not in regex class -> no-op)
        if not found:
            return None
        return found[-1].replace(",", "").replace("$", "")
    final = None                           # flexible: take last number
    for final in reversed(re.findall(r"(\-?[0-9\.\,]+)", text)):
        if final not in ("", "."):         # skip bare dot, keep comma
            break
    return final

def compute_score(solution_str, ground_truth, method="strict",
                  format_score=0.0, score=1.0):
    answer = extract_solution(solution_str, method)
    if answer is None:
        return 0.0
    return score if answer == ground_truth else format_score

def naive_score(solution_str, ground_truth):
    # Award credit on any substring hit
    return 1.0 if ground_truth in solution_str else 0.0

gt = "72"
good = "48 / 2 = 24, 48 + 24 = 72\n#### 72"
wrong = "48 + 20 = 68\n#### 68"
no_tag = "The answer is 72"
as_float = "#### 72.0"
period = "#### 72."
buried = "#### 72\n" + "Let me double check. " * 20
spray = " ".join(str(i) for i in range(100))   # 0 1 2 ... 99
comma = "#### 7,2"
dollar = "#### $72"

cases = [
    ("good", good), ("wrong", wrong), ("no_tag", no_tag),
    ("as_float", as_float), ("period", period), ("buried", buried),
    ("spray", spray), ("comma", comma), ("dollar", dollar),
]
for name, text in cases:
    st = compute_score(text, gt)
    fl = compute_score(text, gt, "flexible")
    nv = naive_score(text, gt)
    print(f"{name:<9} strict={st}  flexible={fl}  naive={nv}")
print(f"int ground truth 72: strict={compute_score(good, 72)}")

# Verification assertions
assert compute_score(good, gt) == 1.0
assert compute_score(wrong, gt) == 0.0
assert compute_score(no_tag, gt) == 0.0
assert compute_score(no_tag, gt, method='flexible') == 1.0
assert compute_score(as_float, gt) == 0.0
assert compute_score(period, gt) == 0.0
assert len(buried) > 300 and compute_score(buried, gt) == 0.0
assert naive_score(spray, gt) == 1.0
assert compute_score(spray, gt) == 0.0
assert compute_score(comma, gt) == 1.0
assert compute_score(wrong, gt, format_score=0.1) == 0.1
assert compute_score(good, 72) == 0.0
assert compute_score(dollar, gt) == 0.0

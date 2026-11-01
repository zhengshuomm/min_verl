# mini_verl — measured results

Every number below comes from the final code in this directory, run with the commands shown.
Runs are deterministic. A second run of Lab 7 printed the same output apart from the timings, and
`tests/test_training.py` (`test_runs_are_reproducible`) checks that the metrics match bit for bit.

## Environment

| item | value |
|---|---|
| machine | Apple M5 Pro (15 cores), macOS 26.7.1, arm64 |
| verl version mirrored | verl v0.9.1 |
| Python / torch | 3.9.6 / torch 2.8.0 (CPU, float32) |
| torch intra-op threads | 1, via `mini_verl.config.set_reproducible()`. Other thread counts split float reductions differently and change the numbers: the same 500-step SFT run reaches 0.155 / 0.240 / 0.315 accuracy with 1 / 2 / 5 threads. |
| simulated DP ranks | 2 (`trainer.n_gpus_per_node=2`; in-process workers, one thread each) |
| seeds | model init `ModelConfig.seed=0`; train/val split `data.seed=0`; SFT `SFTConfig.seed=0`; RL dataloader `trainer.seed=0`; rollout RNG `rollout.seed + rank` = 42 / 43 |

> Other CPUs or BLAS builds may give different numbers, even with one thread. The tests assert the
> qualitative result, with margins: SFT accuracy between 0.2 and 0.5, and after only 50 RL steps at
> least +20 points (GRPO) or +15 points (PPO).

## Task and model

* Prompt `"a+b="` with a, b in 0..99: 10,000 pairs, 200 held out for validation, 9,800 used for SFT and RL.
  Prompts are 4–6 tokens, left-padded to `data.max_prompt_length=6`. Responses `"<sum><eos>"` are
  right-padded to `data.max_response_length=4`.
* Reward: `default_compute_score("arithmetic/addition", ...)` returns 1.0 if the decoded response is
  exactly the sum (strict integer format) and 0.0 otherwise. It is placed on the last valid response token.
* Metric: greedy (`do_sample=False`) accuracy on the 200 validation pairs
  (`val-core/arithmetic/addition/reward/mean@1`).
* Policy `TinyPolicy`: vocab 18, d_model 64, 2 layers, 4 heads, d_ff 256, 32 learned positions,
  untied lm_head, **104,448 parameters**. Critic `TinyCritic`: the same trunk plus a scalar head. The
  trunk is initialised from the SFT policy and the value head starts at zero.

## Headline numbers

| run | steps | greedy val acc | Δ vs SFT | RL wall time | whole lab |
|---|---|---|---|---|---|
| SFT warm start (bs 64, AdamW lr 1e-3, wd 0.01, grad clip 1.0) | 550 | **0.300** | — | — (SFT itself 3.4 s) | — |
| GRPO, `lab7_grpo_training` | 200 | **0.780** | **+48.0 points** | 20.2 s | 24.3 s |
| PPO + GAE critic + KL in reward, `lab8_ppo_with_critic` | 200 | **0.760** | **+46.0 points** | 27.9 s | 32.2 s |

Validation accuracy during training:

| step | 0 | 50 | 100 | 150 | 200 |
|---|---|---|---|---|---|
| GRPO | 0.300 | 0.635 | 0.685 | 0.725 | 0.780 |
| PPO + GAE | 0.300 | 0.675 | 0.720 | 0.735 | 0.760 |

Agent loop (Lab 9): after 150 SFT steps on tool episodes `<c>a+b</c><r>sum</r>sum<eos>`, greedy accuracy
with the calculator is **0.995**. After 5 multi-turn GRPO steps it is **1.000** (sampled score 0.828 → 0.844).

Other lab wall times: lab1 0.60 s, lab2 0.56 s, lab3 4.04 s, lab4 0.56 s, lab5 0.56 s, lab6 0.55 s, lab9 4.03 s.

Test suite: `python3 -m unittest discover -s mini_verl/tests -t .` → **49 tests OK** (including `tests/test_book_listings.py` executing all 10 standalone chapter listings in `examples/book_listings/`).

## Hyperparameters (defaults in `config.py`; per-lab overrides noted)

| key | value |
|---|---|
| data.train_batch_size | 64 prompts |
| actor_rollout_ref.rollout.n | 8 (512 samples per step) |
| actor_rollout_ref.rollout.temperature / top_k | 1.0 / -1 (validation is greedy) |
| actor_rollout_ref.actor.ppo_mini_batch_size | 32 prompts (× n = 256 samples, so 2 optimizer steps per RL step) |
| actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu | 32 |
| actor_rollout_ref.actor.ppo_epochs | 1 |
| actor_rollout_ref.actor.optim | AdamW lr 1e-4, betas (0.9, 0.999), weight_decay 0.01, clip_grad 1.0 |
| actor_rollout_ref.actor.clip_ratio / _low / _high / clip_ratio_c | 0.2 / 0.2 / 0.2 / 3.0 |
| actor_rollout_ref.actor.loss_agg_mode | token-mean |
| actor_rollout_ref.actor.entropy_coeff | 0.0 |
| actor_rollout_ref.actor.use_kl_loss / kl_loss_type / kl_loss_coef | GRPO lab: True / low_var_kl / 0.001; PPO lab: False |
| algorithm.adv_estimator | GRPO lab: grpo (norm_adv_by_std_in_grpo=True); PPO lab: gae |
| algorithm.gamma / lam | 1.0 / 1.0 |
| algorithm.use_kl_in_reward / kl_penalty / kl_ctrl | GRPO lab: False; PPO lab: True / kl / fixed, kl_coef 0.001 |
| critic.optim | AdamW lr 1e-3, wd 0.01, clip_grad 1.0 |
| critic.ppo_mini_batch_size / ppo_micro_batch_size_per_gpu / ppo_epochs / cliprange_value | 32 / 32 / 1 / 0.5 |
| trainer.total_training_steps / test_freq / critic_warmup | 200 / 50 / 0 |
| trainer.balance_batch | True |
| trainer.n_gpus_per_node × nnodes | 2 × 1 |
| SFTConfig (steps / batch_size / lr / seed) | 550 / 64 / 1e-3 / 0 |

## Lab 7 output (GRPO)

```
torch 2.8.0, threads=1, DP ranks=2
task: a+b= with a,b in 0..99; 200 held-out validation pairs
GRPO: train_batch_size=64 prompts x rollout.n=8, ppo_mini_batch_size=32, lr=0.0001, clip=0.2, use_kl_loss=True (low_var_kl, 0.001), steps=200

SFT warm start: 104,448 params, 550 steps, greedy val acc 0.300 (3.4s)

step   0 | val acc 0.300
step  10 | score 0.229 | pg_loss -0.0013 | entropy 0.690 | resp_len 3.48 | 0.10s
step  20 | score 0.238 | pg_loss -0.0058 | entropy 0.669 | resp_len 3.54 | 0.10s
step  30 | score 0.273 | pg_loss -0.0025 | entropy 0.661 | resp_len 3.46 | 0.11s
step  40 | score 0.256 | pg_loss -0.0094 | entropy 0.649 | resp_len 3.49 | 0.10s
step  50 | score 0.359 | pg_loss -0.0003 | entropy 0.594 | resp_len 3.58 | 0.13s | val acc 0.635
step  60 | score 0.354 | pg_loss +0.0008 | entropy 0.595 | resp_len 3.50 | 0.10s
step  70 | score 0.312 | pg_loss +0.0002 | entropy 0.603 | resp_len 3.42 | 0.10s
step  80 | score 0.307 | pg_loss -0.0010 | entropy 0.594 | resp_len 3.51 | 0.10s
step  90 | score 0.357 | pg_loss +0.0011 | entropy 0.591 | resp_len 3.38 | 0.10s
step 100 | score 0.365 | pg_loss +0.0021 | entropy 0.571 | resp_len 3.52 | 0.12s | val acc 0.685
step 110 | score 0.336 | pg_loss -0.0044 | entropy 0.560 | resp_len 3.41 | 0.10s
step 120 | score 0.355 | pg_loss -0.0048 | entropy 0.561 | resp_len 3.49 | 0.10s
step 130 | score 0.367 | pg_loss -0.0008 | entropy 0.529 | resp_len 3.49 | 0.12s
step 140 | score 0.400 | pg_loss +0.0009 | entropy 0.541 | resp_len 3.37 | 0.10s
step 150 | score 0.432 | pg_loss -0.0005 | entropy 0.481 | resp_len 3.52 | 0.12s | val acc 0.725
step 160 | score 0.455 | pg_loss +0.0033 | entropy 0.478 | resp_len 3.46 | 0.10s
step 170 | score 0.457 | pg_loss -0.0070 | entropy 0.473 | resp_len 3.46 | 0.10s
step 180 | score 0.434 | pg_loss -0.0011 | entropy 0.487 | resp_len 3.49 | 0.10s
step 190 | score 0.467 | pg_loss +0.0001 | entropy 0.465 | resp_len 3.53 | 0.10s
step 200 | score 0.566 | pg_loss -0.0046 | entropy 0.445 | resp_len 3.46 | 0.12s | val acc 0.780

greedy val acc: SFT 0.300 -> GRPO step 200 0.780 (+48.0 points) | RL wall time 20.2s
last-step metrics: {'actor/pg_loss': -0.0046, 'actor/pg_clipfrac': 0.0126, 'actor/ppo_kl': 0.0008, 'actor/kl_loss': 0.2187, 'actor/entropy': 0.4454, 'actor/grad_norm': 5.6207, 'critic/score/mean': 0.5664, 'response_length/mean': 3.4629}
```

## Lab 8 output (PPO + GAE)

```
torch 2.8.0, threads=1, DP ranks=2
PPO: gamma=1.0 lam=1.0, critic lr=0.001 cliprange_value=0.5, kl_ctrl=fixed(0.001) kl_penalty=kl, rollout.n=8, steps=200
SFT warm start: greedy val acc 0.300 (3.6s)

worker groups: WorkerDict holds ['actor_rollout', 'critic'] on each of 2 ranks
step   0 | val acc 0.300
step  10 | score 0.238 | pg_loss -0.0013 | entropy 0.690 | resp_len 3.50 | 0.14s
step  20 | score 0.236 | pg_loss -0.0010 | entropy 0.664 | resp_len 3.54 | 0.14s
step  30 | score 0.266 | pg_loss -0.0013 | entropy 0.654 | resp_len 3.46 | 0.14s
step  40 | score 0.266 | pg_loss -0.0022 | entropy 0.640 | resp_len 3.51 | 0.14s
step  50 | score 0.355 | pg_loss -0.0011 | entropy 0.586 | resp_len 3.58 | 0.16s | val acc 0.675
step  60 | score 0.350 | pg_loss -0.0007 | entropy 0.588 | resp_len 3.51 | 0.14s
step  70 | score 0.334 | pg_loss -0.0016 | entropy 0.591 | resp_len 3.41 | 0.14s
step  80 | score 0.305 | pg_loss -0.0018 | entropy 0.585 | resp_len 3.52 | 0.13s
step  90 | score 0.338 | pg_loss -0.0012 | entropy 0.577 | resp_len 3.38 | 0.14s
step 100 | score 0.393 | pg_loss +0.0001 | entropy 0.559 | resp_len 3.52 | 0.18s | val acc 0.720
step 110 | score 0.344 | pg_loss +0.0001 | entropy 0.557 | resp_len 3.41 | 0.14s
step 120 | score 0.334 | pg_loss -0.0002 | entropy 0.553 | resp_len 3.48 | 0.14s
step 130 | score 0.383 | pg_loss -0.0003 | entropy 0.510 | resp_len 3.49 | 0.14s
step 140 | score 0.408 | pg_loss -0.0008 | entropy 0.526 | resp_len 3.34 | 0.14s
step 150 | score 0.443 | pg_loss -0.0005 | entropy 0.474 | resp_len 3.52 | 0.16s | val acc 0.735
step 160 | score 0.486 | pg_loss -0.0003 | entropy 0.466 | resp_len 3.46 | 0.14s
step 170 | score 0.465 | pg_loss +0.0003 | entropy 0.462 | resp_len 3.47 | 0.14s
step 180 | score 0.441 | pg_loss +0.0005 | entropy 0.469 | resp_len 3.49 | 0.14s
step 190 | score 0.459 | pg_loss -0.0021 | entropy 0.446 | resp_len 3.54 | 0.14s
step 200 | score 0.559 | pg_loss -0.0014 | entropy 0.429 | resp_len 3.46 | 0.16s | val acc 0.760

greedy val acc: SFT 0.300 -> PPO step 200 0.760 (+46.0 points) | RL wall time 27.9s
  step   1: critic/vf_loss 0.0946  critic/vf_explained_var +0.000  critic/values/mean +0.000  actor/reward_kl_penalty +0.00000
  step  50: critic/vf_loss 0.1097  critic/vf_explained_var +0.056  critic/values/mean +0.309  actor/reward_kl_penalty +0.20402
  step 100: critic/vf_loss 0.1164  critic/vf_explained_var +0.025  critic/values/mean +0.418  actor/reward_kl_penalty +0.19502
  step 200: critic/vf_loss 0.1132  critic/vf_explained_var +0.085  critic/values/mean +0.513  actor/reward_kl_penalty +0.29308
```

## Lab 9 output (agent loop)

```
== ToolAgentLoop, greedy (max_assistant_turns=2, max_user_turns=1, response_length=16)
  23+48=  turns=4 reward=1
    token:  <c>    2    3    +    4    8 </c>  <r>    7    1 </r>    7    1 <eos>
    mask :    1    1    1    1    1    1    1    0    0    0    0    1    1    1
  7+5=    turns=4 reward=1
    token:  <c>    7    +    5 </c>  <r>    1    2 </r>    1    2 <eos>
    mask :    1    1    1    1    1    0    0    0    0    1    1    1
  99+99=  turns=4 reward=1
    token:  <c>    9    9    +    9    9 </c>  <r>    1    9    8 </r>    1    9    8 <eos>
    mask :    1    1    1    1    1    1    1    0    0    0    0    0    1    1    1    1
  -> <r>..</r> came from the environment: in the context (attention_mask=1) but not in the loss

greedy val acc WITH the calculator: 0.995   (Lab 7's SFT model without tools: ~0.30)

== 5 multi-turn GRPO steps through RayPPOTrainer
step   0 | val acc 0.995
step   1 | score 0.828 | pg_loss -0.0149 | entropy 0.147 | resp_len 14.31 | 0.31s
step   2 | score 0.824 | pg_loss -0.0142 | entropy 0.146 | resp_len 14.46 | 0.24s
step   3 | score 0.844 | pg_loss -0.0119 | entropy 0.143 | resp_len 14.84 | 0.26s
step   4 | score 0.814 | pg_loss -0.0257 | entropy 0.145 | resp_len 14.33 | 0.22s
step   5 | score 0.844 | pg_loss -0.0349 | entropy 0.140 | resp_len 14.38 | 0.34s | val acc 1.000
  sampled score (temperature 1.0): 0.828 -> 0.844; response_length/mean counts tool tokens too (14.38), the loss does not
```

## Lab 4 output (advantages, verbatim)

```
rewards per response: [1.0, 0.0, 0.0, 1.0] (on the last of 3 tokens)

GRPO         (r - mean) / (std + 1e-6), std unbiased
             mean = 0.5, std = sqrt(1/3) = 0.577350
             A = [+0.866024, -0.866024, -0.866024, +0.866024]  (same value on all 3 tokens)
Dr.GRPO      r - mean          A = [+0.500000, -0.500000, -0.500000, +0.500000]
RLOO         r - mean(others)  A = [+0.666667, -0.666667, -0.666667, +0.666667]
REINFORCE++  return-to-go, whitened over the batch
             returns = [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
             A[:,0]  = [+0.957427, -0.957427, -0.957427, +0.957427]
GAE          gamma=1, lam=1, V=0.5: raw A_t = r - 0.5 on every token, then whitened
             raw A  = [+0.500000, -0.500000, -0.500000, +0.500000]
             A[:,0] = [+0.957427, -0.957427, -0.957427, +0.957427]  (12 tokens: std = sqrt(12*0.25/11) = 0.522233)

GAE          gamma=1, lam=0.95, V=[0.2, 0.5, 0.8] (critic grows more confident token by token)
  response 0 (r=1): raw A_t = [+0.765500, +0.490000, +0.200000]  returns = [+0.965500, +0.990000, +1.000000]
  response 1 (r=0): raw A_t = [-0.137000, -0.460000, -0.800000]  returns = [+0.063000, +0.040000, +0.000000]
  -> unlike GRPO, GAE gives different credit to different tokens of the same response
```

## Lab 5 output (policy loss, verbatim)

```
token |  A  | ratio | -A*r   | -A*clip(r,0.8,1.2) | max (PPO) | dual-clip min(.., -A*3)
  0   | +1  | 1.5   | -1.50  | -1.20              | -1.20     | -1.20
  1   | +1  | 0.5   | -0.50  | -0.80              | -0.50     | -0.50
  2   | -1  | 0.5   | +0.50  | +0.80              | +0.80     | +0.80
  3   | -1  | 5.0   | +5.00  | +1.20              | +5.00     | +3.00

compute_policy_loss (dual-clip c=3.0): pg_loss=0.525000  pg_clipfrac=0.5000  pg_clipfrac_lower=0.2500  ppo_kl=-0.157152
without dual-clip (c=1e9):             pg_loss=1.025000   <- token 3 alone contributes 5/4
clip-higher (low=0.2, high=0.28):      pg_loss=0.505000  pg_clipfrac=0.5000
ppo_kl = mean(-log r) = -(ln1.5 + 2 ln0.5 + ln5)/4 = -0.157152

loss aggregation on two responses (lengths 3 and 1):
  loss_mat = [[1.0, 2.0, 3.0, 0.0], [4.0, 0.0, 0.0, 0.0]]  mask = [[1.0, 1.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
  token-mean               = 2.5000   (1+2+3+4)/4
  seq-mean-token-sum       = 5.0000   (6+4)/2
  seq-mean-token-mean      = 3.0000   (6/3 + 4/1)/2
  seq-mean-token-sum-norm  = 1.2500   (6+4)/2/T, T=4

same batch split over 2 DP ranks (one row each), then gradients averaged (all-reduce mean):
  token-mean, batch_num_tokens=4: ranks [3.0, 2.0] -> all-reduce mean 2.5000 (== full batch)
  token-mean, local #tokens     : ranks [2.0, 4.0] -> all-reduce mean 3.0000 (wrong)
```

## Lab 6 output (KL estimators, verbatim)

```
x = log pi - log pi_ref |   k1 (kl)  |  abs     |  k2 (mse) |  k3 (low_var_kl)
        -2.0            | -2.000000 | 2.000000 | 2.000000  | 4.389056
        -0.5            | -0.500000 | 0.500000 | 0.125000  | 0.148721
        -0.1            | -0.100000 | 0.100000 | 0.005000  | 0.005171
        +0.0            | +0.000000 | 0.000000 | 0.000000  | 0.000000
        +0.1            | +0.100000 | 0.100000 | 0.005000  | 0.004837
        +0.5            | +0.500000 | 0.500000 | 0.125000  | 0.106531
        +2.0            | +2.000000 | 2.000000 | 2.000000  | 1.135335
k3 = exp(-x) + x - 1 >= 0 always (k1 can be negative per token); k3 is clamped to [-10, 10]

KL(pi || pi_ref) exact = 0.095574   (100,000 samples from pi, seed 0)
estimator | MC mean  | per-sample std | bias
kl        | 0.094898 | 0.417758       | -0.000676
mse       | 0.091763 | 0.074437       | -0.003811
low_var_kl| 0.096006 | 0.095054       | +0.000432
k1 and k3 are unbiased; k3 has the smaller variance; k2 is biased (but its gradient is the right one).
(check: exp is applied to log-ratios of at most ln(0.40/0.25) = 0.4700)
```

# mini_verl

Companion code for **verl From Scratch** / **《从零吃透 verl》** (Alan Haverford).

`mini_verl` is a CPU-only, pure-PyTorch re-implementation of the core of **verl**, the RL post-training
framework. Size, counted as non-blank, non-comment Python lines: core library 1,826 (13 modules); with the
9 labs (441), 10 standalone chapter listings (479) and 8 test files (524), 3,271 in total. Classes, functions, config keys and metric names follow
**verl v0.9.1** (`commit 1876b06`) so that you can open the real source side by side; every docstring
cites the verl file it mirrors. The only dependency is `torch` (see [requirements.txt](requirements.txt); tested with `torch==2.8.0`, Python 3.9+).

```bash
git clone https://github.com/zhengshuomm/min_verl.git && cd min_verl
pip install -r requirements.txt                                            # CPU PyTorch is enough
python3 -m unittest discover -s mini_verl/tests -t .                       # 49 tests (~20 s; includes all book listings)
python3 -m mini_verl.examples.lab7_grpo_training                           # any lab: lab1_dataproto ... lab9_agent_loop
python3 mini_verl/examples/book_listings/ch04_s2_miniproto_dispatch.py     # any standalone chapter listing
```

```python
from mini_verl import PPOConfig, RayPPOTrainer
from mini_verl.config import SFTConfig, set_reproducible
from mini_verl.sft import warm_start_policy

set_reproducible()                                        # 1 intra-op thread -> the numbers in RESULTS.md
cfg = PPOConfig().apply_overrides({"algorithm.adv_estimator": "grpo", "actor_rollout_ref.rollout.n": 8})
policy = warm_start_policy(cfg.actor_rollout_ref.model, cfg.data, SFTConfig())   # ~30% accurate
trainer = RayPPOTrainer(cfg, policy_state_dict=policy.state_dict())
trainer.init_workers()
trainer.fit(log_every=10)                                 # ~78% greedy validation accuracy after 200 steps
```

## GPU quickstart launcher for real verl (book Section 1.7)

`mini_verl/scripts/verl_quickstart.sh` launches verl v0.9.1's single-GPU PPO quickstart
(Qwen2.5-0.5B-Instruct on GSM8K) so that the driver **and every Ray worker** use the same Python environment
(`uv run --frozen --all-packages --extra vllm --extra fsdp` wired into both the driver and
`ray_kwargs.ray_init.runtime_env.py_executable`). Run it from the root of a verl checkout at `v0.9.1` (`1876b06`):

```bash
# Path A (inside docker/Dockerfile.uv.cu130, which bakes the uv wheel cache and materializes
#         /workspace/verl/.venv offline on first uv run) AND Path B (Linux host + uv, Python 3.12):
bash verl_quickstart.sh

# Append extra Hydra overrides (e.g. switch PPO to GRPO):
bash verl_quickstart.sh algorithm.adv_estimator=grpo actor_rollout_ref.rollout.n=5

# Pre-populated ambient python3 ONLY (e.g. Dockerfile.stable.vllm after `pip install --no-deps -e .`,
# or after an explicit `uv sync --frozen --all-packages --extra vllm --extra fsdp`):
VERL_USE_UV=0 bash verl_quickstart.sh

# Print the exact expanded commands without launching anything (no GPU needed):
DRY_RUN=1 bash verl_quickstart.sh
```

All 26 Hydra override keys were checked against verl's `_generated_ppo_trainer.yaml` at `1876b06`. The GPU
training run itself is not part of this repository's CPU verification.

## The toy task

Two-number addition: the prompt is `"23+48="` (operands `0..99`, so prompts are 4–6 tokens and are
LEFT-padded to 6), the answer is `"71<eos>"` (RIGHT-padded to 4 response tokens). 200 of the 10,000
pairs are held out for validation. The reward is rule-based and verifiable: 1.0 if the response is
exactly the right integer, else 0, placed on the last valid response token. The policy is a 2-layer,
d_model 64, 4-head decoder (104,448 parameters) warm-started with 550 SFT steps to ~30% accuracy.
Measured numbers are in [RESULTS.md](RESULTS.md).

## What it implements

* **DataProto** — `batch` (a tiny `TensorDict`), `non_tensor_batch` (Python lists), `meta_info`;
  `from_dict`, `select`, `pop`, `union`, `chunk`, `concat`, `repeat(interleave=True)`, slicing,
  `pad_dataproto_to_divisor` / `unpad_dataproto`, auto-padding.
* **Single controller** — `Worker`, `WorkerGroup`, `ResourcePool`, `@register(dispatch_mode, execute_mode)`,
  `Dispatch.ONE_TO_ALL / ALL_TO_ALL / DP_COMPUTE / DP_COMPUTE_PROTO / RANK_ZERO`, `Execute.ALL / RANK_ZERO`,
  `create_colocated_worker_cls` + `spawn`. Ranks are in-process objects run in threads, with a
  barrier-based `all_reduce` standing in for NCCL.
* **Workers** — `ActorRolloutRefWorker` (`generate_sequences`, `compute_log_prob`,
  `compute_ref_log_prob`, `update_actor`, `update_weights`) and `CriticWorker` (`compute_values`,
  `update_critic`): PPO epochs × mini-batches × micro-batches, gradient accumulation, global-token
  loss normalisation, DP gradient all-reduce, grad clipping, KL-in-loss.
* **Rollout** — a separate inference engine with its own weight copy and RNG (temperature, top-k,
  greedy, EOS), verl's output layout, and an explicit `sync_weights_from(actor)` / `weight_version`.
* **Agent loop** — multi-turn `<c>expr</c>` calculator tool; tool tokens get `response_mask = 0`.
* **Algorithms** — GAE, GRPO (and Dr.GRPO), RLOO, REINFORCE++; PPO clip + dual-clip policy loss;
  clipped value loss; `agg_loss` (4 modes); KL estimators k1/abs/k2/k3 (+ straight-through `k3+`);
  fixed / adaptive KL controllers; `apply_kl_penalty`; `balance_batch`.
* **Trainer** — `RayPPOTrainer.fit` with verl's step order and metric names.

## File map (mini_verl → verl)

| mini_verl | verl | Lab |
|---|---|---|
| `protocol.py` (`DataProto`, `TensorDict`, `pad_dataproto_to_divisor`) | `verl/protocol.py` | 1 |
| `single_controller.py` (`Worker`, `WorkerGroup`, `ResourcePool`, `register`, `Dispatch`, `Execute`, `create_colocated_worker_cls`) | `verl/single_controller/base/{decorator,worker,worker_group}.py`, `verl/single_controller/ray/base.py` | 2 |
| `data.py` (`Tokenizer`, `RLHFDataset`, `collate_fn`) | `verl/utils/dataset/rl_dataset.py` | 1 |
| `models.py` (`TinyPolicy`, `TinyCritic`, `logprobs_from_logits`, `entropy_from_logits`, `compute_position_id_with_mask`) | HF `AutoModelForCausalLM` / `AutoModelForTokenClassification`; `verl/utils/torch_functional.py`, `verl/utils/model.py` | 3 |
| `rollout.py` (`Rollout`, `build_rollout_batch`) | `verl/workers/rollout/base.py`, `verl/workers/rollout/hf_rollout.py`, `verl/experimental/agent_loop/agent_loop.py` (`_postprocess`); weight sync: `verl/workers/engine_workers.py` (`update_weights`), `verl/checkpoint_engine/base.py` | 3 |
| `agent_loop.py` (`ToolAgentLoop`, `AgentState`, `AgentLoopOutput`) | `verl/experimental/agent_loop/tool_agent_loop.py`, `verl/experimental/agent_loop/agent_loop.py` | 9 |
| `reward.py` (`default_compute_score`, `compute_reward`) | `verl/utils/reward_score/__init__.py`, `verl/utils/reward_score/gsm8k.py`, `verl/workers/reward_manager/naive.py` | 3, 7 |
| `core_algos.py` | `verl/trainer/ppo/core_algos.py`, `verl/utils/torch_functional.py` (`masked_*`) | 4, 5, 6 |
| `workers.py` (`ActorRolloutRefWorker`, `CriticWorker`, `ppo_train_loop`) | `verl/workers/engine_workers.py`, `verl/workers/utils/losses.py`, `verl/workers/engine/fsdp/transformer_impl.py` (older releases: `DataParallelPPOActor` / `DataParallelPPOCritic`) | 3, 7, 8 |
| `trainer.py` (`RayPPOTrainer`, `compute_advantage`, `apply_kl_penalty`, `compute_data_metrics`) | `verl/trainer/ppo/ray_trainer.py`, `verl/trainer/ppo/metric_utils.py`, `verl/utils/seqlen_balancing.py` | 7, 8 |
| `sft.py` (`run_sft`, `sft_loss`, `greedy_accuracy`, `warm_start_policy`) | `verl/trainer/sft_trainer.py`, `verl/workers/utils/losses.py` (`sft_loss`) | 7, 9 |
| `config.py` (`PPOConfig` and friends) | `verl/trainer/config/ppo_trainer.yaml`, `verl/workers/config/*.py`, `verl/trainer/config/algorithm.py` | all |

## Labs

| Lab | Command | Shows |
|---|---|---|
| 1 | `python3 -m mini_verl.examples.lab1_dataproto` | left-padded batch, pop / repeat / union / chunk / concat / padding |
| 2 | `python3 -m mini_verl.examples.lab2_single_controller` | which rank gets which rows; auto-padding; RANK_ZERO; all_reduce; colocation + spawn |
| 3 | `python3 -m mini_verl.examples.lab3_rollout_and_weight_sync` | rollout tensors and masks; stale rollout weights until `update_weights` |
| 4 | `python3 -m mini_verl.examples.lab4_advantages` | GRPO vs Dr.GRPO vs RLOO vs REINFORCE++ vs GAE on rewards [1, 0, 0, 1] |
| 5 | `python3 -m mini_verl.examples.lab5_policy_loss` | clip, dual-clip, clip-higher, `agg_loss` modes, DP-invariant normalisation |
| 6 | `python3 -m mini_verl.examples.lab6_kl_estimators` | k1 / k2 / k3 values; Monte-Carlo bias and variance |
| 7 | `python3 -m mini_verl.examples.lab7_grpo_training` | SFT 30% → GRPO 78% greedy accuracy in 200 steps (~25 s) |
| 8 | `python3 -m mini_verl.examples.lab8_ppo_with_critic` | PPO + GAE + colocated critic + KL in reward: 30% → 76% (~30 s) |
| 9 | `python3 -m mini_verl.examples.lab9_agent_loop` | multi-turn tool calls, response_mask = 0 on tool tokens, multi-turn GRPO |

## Standalone Book Listings (`examples/book_listings/`)

Every runnable listing printed in *verl From Scratch* is also exported as a standalone `.py` script with built-in assertions (no PDF copy-paste required):

- `ch01_s5_decode_budget.py` — KV-cache capacity, waves, and roofline ridge on 8×H100
- `ch02_s6_ppo_dual_clip_loss.py` — Vanilla PPO + dual-clip token-mean policy loss
- `ch03_s2_grpo_rloo_advantages.py` — GRPO, Dr. GRPO, and RLOO advantage estimators
- `ch04_s2_miniproto_dispatch.py` — `MiniProto` and `DP_COMPUTE_PROTO` auto-padding dispatch
- `ch05_s6_kk_seqlen_partition.py` — Karmarkar–Karp sequence-length workload balancing
- `ch06_s5_bucketed_weight_sender.py` — `BucketedWeightSender` packing for Qwen3-8B (bf16 vs fp32)
- `ch07_s4_token_budget_microbatch.py` — Greedy token-budget micro-batch packing (`use_dynamic_bsz`)
- `ch08_s1_gsm8k_reward_pitfalls.py` — `extract_solution` (`strict` vs `flexible` vs substring) edge cases
- `ch09_s3_multiturn_mask_and_gae.py` — Multi-turn `response_mask` and masked backward GAE
- `ch10_s4_dapo_dynamic_sampling.py` — DAPO `drop_uniform_groups`, uniform-group probability, gradient dilution

## Deliberate simplifications (vs. real verl)

* CPU, float32, no Ray: workers are in-process objects run in threads; `all_reduce` is a barrier
  plus an ordered sum. Every DP rank holds a full replica (no FSDP sharding, no TP/PP/SP).
* The rollout engine recomputes the full prefix every decoding step (no KV cache, no vLLM /
  SGLang server, no async rollout); weight sync is an in-process parameter copy.
* The trainer creates `trainer.n_gpus_per_node` ranks in one resource pool and always colocates
  actor / rollout / ref / critic (verl also supports separate pools). In v0.9.1 verl marks the
  legacy `ray_trainer.RayPPOTrainer` as deprecated in favour of `verl/trainer/ppo/v1`; mini_verl
  follows the legacy file because its `fit` shows the whole dataflow in one place.
* Padded tensors everywhere (verl's engine workers convert to no-padding / nested tensors).
* `balance_batch` uses a greedy partitioner instead of Karmarkar–Karp and the raw token count as
  the workload; `uid` is a deterministic string instead of `uuid4`.
* Only the vanilla policy loss and the four advantage estimators above; no reward model, no
  rollout correction (importance sampling), no dynamic batch size, no checkpointing.
* Non-tensor columns are Python lists (verl: numpy object arrays); auto-padding in
  `dispatch_dp_compute_data_proto` pads a copy (verl pads the caller's DataProto in place).

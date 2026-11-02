#!/usr/bin/env bash
# verl_quickstart.sh — verl v0.9.1 single-GPU PPO quickstart (Qwen2.5-0.5B-Instruct on GSM8K).
# Companion launcher for "verl From Scratch", Section 1.7. Run it from the ROOT of a verl checkout at
# tag v0.9.1 (commit 1876b06). It follows the LAUNCH/RAY convention of verl's own examples/*.sh
# (docs/start/install.rst & docs/start/quickstart.rst), so the driver AND every Ray worker run in the
# same Python environment.
#
#   Path A — official uv Docker image (docker/Dockerfile.uv.cu130; bakes uv cache, materializes .venv
#            offline on first uv run):                          bash verl_quickstart.sh
#   Path B — Linux host + uv (Python 3.12, committed uv.lock):  bash verl_quickstart.sh
#   Pre-populated ambient python3 ONLY (e.g. Dockerfile.stable.vllm after `pip install --no-deps -e .`,
#            or after an explicit `uv sync --frozen --all-packages --extra vllm --extra fsdp`):
#                                                               VERL_USE_UV=0 bash verl_quickstart.sh
#
# Extra Hydra overrides are appended verbatim, e.g.
#   bash verl_quickstart.sh algorithm.adv_estimator=grpo actor_rollout_ref.rollout.n=5 \
#        actor_rollout_ref.actor.use_kl_loss=True
# DRY_RUN=1 prints the exact commands instead of running them (no GPU needed).
set -euo pipefail

if [ "$(git rev-parse --short HEAD 2>/dev/null || true)" != "1876b06" ]; then
  echo "warning: not a verl checkout at v0.9.1 (expected commit 1876b06); flags were verified only there" >&2
fi

DATA_DIR="${DATA_DIR:-$HOME/data/gsm8k}"
LAUNCH=(python3)
RAY=(ray_kwargs.ray_init.runtime_env.py_executable=null)
if [ "${VERL_USE_UV:-1}" != 0 ]; then
  UV_RUN="uv run --frozen --all-packages --extra vllm --extra fsdp"
  # shellcheck disable=SC2206  # intentional word splitting of the uv prefix
  LAUNCH=(${UV_RUN} python3)
  # Ray starts TaskRunner and every worker actor with this prefix, inside the same locked .venv
  RAY=("ray_kwargs.ray_init.runtime_env.py_executable=${UV_RUN}")
elif [ "${DRY_RUN:-0}" != 1 ]; then
  if ! python3 -c "import torch, vllm, verl" >/dev/null 2>&1; then
    echo "error: VERL_USE_UV=0 requires torch, vllm, and verl in ambient python3." >&2
    echo "hint: docker/Dockerfile.uv.cu130 bakes the uv wheel cache, NOT a fixed .venv;" >&2
    echo "      either drop VERL_USE_UV=0 (default uv run) or materialize .venv first via:" >&2
    echo "      uv sync --frozen --all-packages --extra vllm --extra fsdp" >&2
    exit 1
  fi
fi

run() {
  if [ "${DRY_RUN:-0}" = 1 ]; then printf '%q ' "$@"; printf '\n'; else "$@"; fi
}

if [ ! -f "${DATA_DIR}/train.parquet" ] || [ "${DRY_RUN:-0}" = 1 ]; then
  run "${LAUNCH[@]}" examples/data_preprocess/gsm8k.py --local_save_dir "${DATA_DIR}"
fi

ARGS=(
  data.train_files="${DATA_DIR}/train.parquet"
  data.val_files="${DATA_DIR}/test.parquet"
  data.train_batch_size=256
  data.max_prompt_length=512
  data.max_response_length=512
  actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct
  actor_rollout_ref.actor.optim.lr=1e-6
  actor_rollout_ref.actor.ppo_mini_batch_size=64
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8
  actor_rollout_ref.rollout.tensor_model_parallel_size=1
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4
  critic.optim.lr=1e-5
  critic.model.path=Qwen/Qwen2.5-0.5B-Instruct
  critic.ppo_micro_batch_size_per_gpu=4
  algorithm.kl_ctrl.kl_coef=0.001
  trainer.logger=console
  trainer.val_before_train=False
  trainer.n_gpus_per_node=1
  trainer.nnodes=1
  trainer.save_freq=10
  trainer.test_freq=10
  trainer.total_epochs=15
)

export PYTHONUNBUFFERED=1
if [ "${DRY_RUN:-0}" = 1 ]; then
  run "${LAUNCH[@]}" -m verl.trainer.main_ppo "${ARGS[@]}" "${RAY[@]}" "$@"
else
  "${LAUNCH[@]}" -m verl.trainer.main_ppo "${ARGS[@]}" "${RAY[@]}" "$@" 2>&1 | tee verl_demo.log
fi

"""Lab 7 -- End-to-end GRPO: SFT warm start, then RayPPOTrainer.fit with adv_estimator=grpo.

Run:  python3 -m mini_verl.examples.lab7_grpo_training        (about 40 s on a laptop CPU)

The verl equivalent is examples/grpo_trainer/*.sh:
    algorithm.adv_estimator=grpo actor_rollout_ref.rollout.n=8 actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_type=low_var_kl actor_rollout_ref.actor.kl_loss_coef=0.001
"""

import time

import torch

from mini_verl.config import PPOConfig, SFTConfig, set_reproducible
from mini_verl.data import Tokenizer, make_addition_pairs
from mini_verl.sft import greedy_accuracy, warm_start_policy
from mini_verl.trainer import RayPPOTrainer

OVERRIDES = {"algorithm.adv_estimator": "grpo", "actor_rollout_ref.actor.use_kl_loss": True}


def main():
    set_reproducible()
    cfg, sft_cfg = PPOConfig().apply_overrides(OVERRIDES), SFTConfig()
    d, a, r = cfg.data, cfg.actor_rollout_ref.actor, cfg.actor_rollout_ref.rollout
    print(f"torch {torch.__version__}, threads={torch.get_num_threads()}, DP ranks={cfg.trainer.n_gpus_per_node}")
    print(f"task: a+b= with a,b in 0..{d.operand_max}; {d.val_size} held-out validation pairs")
    print(f"GRPO: train_batch_size={d.train_batch_size} prompts x rollout.n={r.n}, ppo_mini_batch_size={a.ppo_mini_batch_size}, "
          f"lr={a.optim.lr}, clip={a.clip_ratio}, use_kl_loss={a.use_kl_loss} ({a.kl_loss_type}, {a.kl_loss_coef}), "
          f"steps={cfg.trainer.total_training_steps}")

    t0 = time.time()
    policy = warm_start_policy(cfg.actor_rollout_ref.model, d, sft_cfg)
    _, val_pairs = make_addition_pairs(d.operand_max, d.val_size, d.seed)
    sft_acc = greedy_accuracy(policy, val_pairs, Tokenizer(), d.max_prompt_length, d.max_response_length)
    print(f"\nSFT warm start: {sum(p.numel() for p in policy.parameters()):,} params, {sft_cfg.steps} steps, "
          f"greedy val acc {sft_acc:.3f} ({time.time() - t0:.1f}s)\n")

    t1 = time.time()
    trainer = RayPPOTrainer(cfg, policy_state_dict=policy.state_dict())
    trainer.init_workers()
    history = trainer.fit(log_every=10)
    rl_time = time.time() - t1
    key = f"val-core/{d.data_source}/reward/mean@1"
    final = history[-1]
    print(f"\ngreedy val acc: SFT {sft_acc:.3f} -> GRPO step {final['training/global_step']} {final[key]:.3f} "
          f"({(final[key] - sft_acc) * 100:+.1f} points) | RL wall time {rl_time:.1f}s")
    print("last-step metrics:", {k: round(final[k], 4) for k in ("actor/pg_loss", "actor/pg_clipfrac", "actor/ppo_kl",
                                                                  "actor/kl_loss", "actor/entropy", "actor/grad_norm",
                                                                  "critic/score/mean", "response_length/mean")})


if __name__ == "__main__":
    main()

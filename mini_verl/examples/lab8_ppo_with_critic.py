"""Lab 8 -- PPO with a critic: adv_estimator=gae, CriticWorker colocated with the actor.

Run:  python3 -m mini_verl.examples.lab8_ppo_with_critic        (about 50 s on a laptop CPU)

Same task, warm start and actor settings as Lab 7, but advantages come from GAE with a learned
value model (initialised from the SFT policy trunk + a fresh value head), and the KL to the
reference policy is subtracted from the reward (algorithm.use_kl_in_reward=True, apply_kl_penalty).
"""

import time

import torch

from mini_verl.config import PPOConfig, SFTConfig, set_reproducible
from mini_verl.data import Tokenizer, make_addition_pairs
from mini_verl.sft import greedy_accuracy, warm_start_policy
from mini_verl.trainer import RayPPOTrainer

OVERRIDES = {"algorithm.adv_estimator": "gae", "algorithm.use_kl_in_reward": True}


def main():
    set_reproducible()
    cfg, sft_cfg = PPOConfig().apply_overrides(OVERRIDES), SFTConfig()
    d, c, alg = cfg.data, cfg.critic, cfg.algorithm
    print(f"torch {torch.__version__}, threads={torch.get_num_threads()}, DP ranks={cfg.trainer.n_gpus_per_node}")
    print(f"PPO: gamma={alg.gamma} lam={alg.lam}, critic lr={c.optim.lr} cliprange_value={c.cliprange_value}, "
          f"kl_ctrl={alg.kl_ctrl.type}({alg.kl_ctrl.kl_coef}) kl_penalty={alg.kl_penalty}, rollout.n="
          f"{cfg.actor_rollout_ref.rollout.n}, steps={cfg.trainer.total_training_steps}")

    t0 = time.time()
    policy = warm_start_policy(cfg.actor_rollout_ref.model, d, sft_cfg)
    _, val_pairs = make_addition_pairs(d.operand_max, d.val_size, d.seed)
    sft_acc = greedy_accuracy(policy, val_pairs, Tokenizer(), d.max_prompt_length, d.max_response_length)
    print(f"SFT warm start: greedy val acc {sft_acc:.3f} ({time.time() - t0:.1f}s)\n")

    t1 = time.time()
    trainer = RayPPOTrainer(cfg, policy_state_dict=policy.state_dict())
    trainer.init_workers()
    print("worker groups:", type(trainer.actor_rollout_wg.workers[0]).__name__, "holds",
          list(trainer.actor_rollout_wg.workers[0].worker_dict), "on each of", trainer.actor_rollout_wg.world_size, "ranks")
    history = trainer.fit(log_every=10)
    rl_time = time.time() - t1
    key = f"val-core/{d.data_source}/reward/mean@1"
    final = history[-1]
    print(f"\ngreedy val acc: SFT {sft_acc:.3f} -> PPO step {final['training/global_step']} {final[key]:.3f} "
          f"({(final[key] - sft_acc) * 100:+.1f} points) | RL wall time {rl_time:.1f}s")
    for step in (1, 50, 100, 200):
        h = history[step]
        print(f"  step {step:>3}: critic/vf_loss {h['critic/vf_loss']:.4f}  critic/vf_explained_var "
              f"{h['critic/vf_explained_var']:+.3f}  critic/values/mean {h['critic/values/mean']:+.3f}  "
              f"actor/reward_kl_penalty {h['actor/reward_kl_penalty']:+.5f}")


if __name__ == "__main__":
    main()

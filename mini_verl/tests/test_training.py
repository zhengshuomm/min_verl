"""Headline tests: from the same SFT warm start, GRPO and PPO+GAE both raise greedy validation
accuracy by a clear margin; runs are bit-for-bit reproducible."""

import unittest

from mini_verl.config import PPOConfig, SFTConfig, set_reproducible
from mini_verl.data import Tokenizer, make_addition_pairs
from mini_verl.sft import greedy_accuracy, warm_start_policy
from mini_verl.trainer import RayPPOTrainer

KEY = "val-core/arithmetic/addition/reward/mean@1"


def train(state_dict, steps, **overrides):
    cfg = PPOConfig().apply_overrides({"trainer.total_training_steps": steps, "trainer.test_freq": steps, **overrides})
    trainer = RayPPOTrainer(cfg, policy_state_dict=state_dict)
    trainer.init_workers()
    return trainer.fit(log_every=0)


class TestTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_reproducible()
        cfg = PPOConfig()
        d = cfg.data
        policy = warm_start_policy(cfg.actor_rollout_ref.model, d, SFTConfig())
        cls.state_dict = policy.state_dict()
        _, val_pairs = make_addition_pairs(d.operand_max, d.val_size, d.seed)
        cls.sft_acc = greedy_accuracy(policy, val_pairs, Tokenizer(), d.max_prompt_length, d.max_response_length)

    def test_sft_is_clearly_imperfect(self):
        self.assertGreaterEqual(self.sft_acc, 0.2)
        self.assertLessEqual(self.sft_acc, 0.5)

    def test_grpo_improves_validation_accuracy(self):
        history = train(self.state_dict, 50, **{"algorithm.adv_estimator": "grpo",
                                                "actor_rollout_ref.actor.use_kl_loss": True})
        self.assertEqual(history[0][KEY], self.sft_acc)  # trainer._validate (2 DP ranks) == standalone eval
        self.assertGreaterEqual(history[-1][KEY] - self.sft_acc, 0.20, f"{self.sft_acc} -> {history[-1][KEY]}")
        for k in ("actor/pg_loss", "actor/pg_clipfrac", "actor/ppo_kl", "actor/kl_loss", "actor/entropy",
                  "critic/score/mean", "response_length/mean", "timing_s/gen", "timing_s/update_actor"):
            self.assertIn(k, history[-1])

    def test_ppo_with_critic_improves_validation_accuracy(self):
        history = train(self.state_dict, 50, **{"algorithm.adv_estimator": "gae", "algorithm.use_kl_in_reward": True})
        self.assertGreaterEqual(history[-1][KEY] - self.sft_acc, 0.15, f"{self.sft_acc} -> {history[-1][KEY]}")
        self.assertIn("critic/vf_loss", history[-1])
        self.assertIn("actor/reward_kl_penalty", history[-1])

    def test_runs_are_reproducible(self):
        a = train(self.state_dict, 3, **{"algorithm.adv_estimator": "grpo"})
        b = train(self.state_dict, 3, **{"algorithm.adv_estimator": "grpo"})
        for ha, hb in zip(a, b):
            self.assertEqual({k: v for k, v in ha.items() if not k.startswith("timing")},
                             {k: v for k, v in hb.items() if not k.startswith("timing")})


if __name__ == "__main__":
    unittest.main()

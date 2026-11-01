"""Workers: simulated data parallelism equals single-process training; rollout weights are stale
until ``update_weights`` syncs them from the actor."""

import unittest

import torch

from mini_verl.config import PPOConfig
from mini_verl.data import RLHFDataset, Tokenizer, collate_fn, make_addition_pairs
from mini_verl.protocol import DataProto
from mini_verl.single_controller import ClassWithInitArgs, ResourcePool, WorkerGroup
from mini_verl.workers import ActorRolloutRefWorker, CriticWorker, forward_micro_batch


def actor_group(n, cfg, state_dict, role="actor_rollout"):
    wg = WorkerGroup(ResourcePool([n]), ClassWithInitArgs(ActorRolloutRefWorker, config=cfg.actor_rollout_ref, role=role))
    wg.init_model(state_dict)
    return wg


def prompts(n=8):
    train, _ = make_addition_pairs(99, 200, 0)
    ds = RLHFDataset(train[:n], Tokenizer(), 6, "arithmetic/addition")
    batch = DataProto.from_single_dict(collate_fn([ds[i] for i in range(n)]))
    return batch.pop(batch_keys=["input_ids", "attention_mask", "position_ids"])


def rollout_batch(wg, n_prompts=8, n=2):
    gen = prompts(n_prompts).repeat(n)
    out = wg.generate_sequences(gen)
    out.meta_info["temperature"] = 1.0
    out = out.union(wg.compute_log_prob(out))
    out.batch.pop("entropys")
    g = torch.Generator().manual_seed(1)
    out.batch["advantages"] = torch.randn(out.batch["responses"].shape, generator=g) * out.batch["response_mask"]
    out.batch["returns"] = torch.randn(out.batch["responses"].shape, generator=g)
    return out


class TestDataParallel(unittest.TestCase):
    def setUp(self):
        self.cfg = PPOConfig()
        self.cfg.actor_rollout_ref.actor.optim.lr = 1e-3
        self.sd = actor_group(1, self.cfg, None).get_actor_state_dict()

    def run_update(self, world_size, micro, mode="token-mean", epochs=1):
        cfg = self.cfg
        cfg.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu = micro
        cfg.actor_rollout_ref.actor.ppo_epochs = epochs
        cfg.actor_rollout_ref.actor.loss_agg_mode = mode
        data = rollout_batch(actor_group(1, cfg, self.sd))
        data.meta_info["mini_batch_size"] = len(data)  # one global mini-batch per epoch
        wg = actor_group(world_size, cfg, self.sd)
        metrics = wg.update_actor(data).meta_info["metrics"]
        return wg, metrics

    def test_dp2_equals_single_process(self):
        for mode in ("token-mean", "seq-mean-token-mean", "seq-mean-token-sum-norm"):
            single, m1 = self.run_update(1, micro=16, mode=mode)  # 1 rank x 1 micro-batch of 16
            dp, m2 = self.run_update(2, micro=4, mode=mode)       # 2 ranks x 2 micro-batches of 4
            s, r0, r1 = (wg.workers[i].actor_module for wg, i in ((single, 0), (dp, 0), (dp, 1)))
            for a, b in zip(r0.parameters(), r1.parameters()):
                self.assertTrue(torch.equal(a, b) and torch.equal(a.grad, b.grad), "DP replicas diverged")
            for a, b in zip(s.parameters(), r0.parameters()):  # the all-reduced gradient == full-batch gradient
                self.assertTrue(torch.allclose(a.grad, b.grad, atol=1e-7), f"{mode}: DP grad != single-process grad")
                self.assertTrue(torch.allclose(a, b, atol=1e-4))  # Adam amplifies 1e-8 rounding a little
            self.assertAlmostEqual(m1["actor/grad_norm"][0], m2["actor/grad_norm"][0], places=5)
            self.assertAlmostEqual(m1["actor/pg_loss"][0], sum(m2["actor/pg_loss"]) / 2, places=6)

    def test_params_actually_change(self):
        dp, _ = self.run_update(2, micro=4, epochs=2)
        before = actor_group(1, self.cfg, self.sd).workers[0].actor_module
        self.assertFalse(all(torch.equal(a, b) for a, b in zip(before.parameters(), dp.workers[0].actor_module.parameters())))

    def test_critic_dp_equals_single_process(self):
        data = rollout_batch(actor_group(1, self.cfg, self.sd))
        data.meta_info["mini_batch_size"] = len(data)
        out = []
        for n, micro in ((1, 16), (2, 4)):
            self.cfg.critic.ppo_micro_batch_size_per_gpu = micro
            wg = WorkerGroup(ResourcePool([n]), ClassWithInitArgs(CriticWorker, config=self.cfg.critic,
                                                                  model_config=self.cfg.actor_rollout_ref.model))
            wg.init_model(self.sd)
            d = data.union(wg.compute_values(data)) if "values" not in data.batch else data
            wg.update_critic(d)
            out.append(wg.workers[-1].critic_module)
        for a, b in zip(out[0].parameters(), out[1].parameters()):
            self.assertTrue(torch.allclose(a.grad, b.grad, atol=1e-7))
            self.assertTrue(torch.allclose(a, b, atol=1e-4))


class TestWeightSync(unittest.TestCase):
    def test_rollout_is_stale_until_update_weights(self):
        cfg = PPOConfig()
        cfg.actor_rollout_ref.actor.optim.lr = 1e-2
        wg = actor_group(1, cfg, None)
        worker = wg.workers[0]
        data = rollout_batch(wg)
        mask = data.batch["response_mask"].bool()
        # right after init the rollout holds the same weights: rollout log-probs == actor log-probs
        self.assertTrue(torch.allclose(data.batch["rollout_log_probs"][mask], data.batch["old_log_probs"][mask], atol=1e-5))
        data.meta_info["mini_batch_size"] = len(data)
        wg.update_actor(data)

        def gap(batch):
            actor_lp, _ = forward_micro_batch(worker.actor_module, batch.batch, 1.0)
            m = batch.batch["response_mask"].bool()
            return (actor_lp[m] - batch.batch["rollout_log_probs"][m]).abs().max().item()

        stale = wg.generate_sequences(prompts().repeat(2))
        self.assertEqual(stale.meta_info["weight_version"], 0)
        self.assertGreater(gap(stale), 1e-2)                     # actor moved, rollout did not
        wg.update_weights(1)
        fresh = wg.generate_sequences(prompts().repeat(2))
        self.assertEqual(fresh.meta_info["weight_version"], 1)
        self.assertLess(gap(fresh), 1e-5)
        for a, b in zip(worker.actor_module.parameters(), worker.rollout.module.parameters()):
            self.assertTrue(torch.equal(a, b))
            self.assertNotEqual(a.data_ptr(), b.data_ptr())      # a COPY, not shared storage


if __name__ == "__main__":
    unittest.main()

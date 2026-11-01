"""core_algos against hand-computed values (every number here is derived in a comment)."""

import math
import unittest

import torch

from mini_verl import core_algos as ca
from mini_verl.config import AlgoConfig


def close(tc, a, b, tol=1e-5):
    tc.assertTrue(torch.allclose(torch.as_tensor(a, dtype=torch.float32), torch.as_tensor(b, dtype=torch.float32),
                                 atol=tol), f"{a} != {b}")


class TestMasked(unittest.TestCase):
    def test_masked_mean_var_whiten(self):
        v, m = torch.tensor([1.0, 2.0, 3.0, 100.0]), torch.tensor([1.0, 1.0, 1.0, 0.0])
        close(self, ca.masked_mean(v, m), 2.0)
        close(self, ca.masked_var(v, m), 1.0)                       # unbiased: (1+0+1)/2
        close(self, ca.masked_whiten(v, m)[:3], [-1.0, 0.0, 1.0])
        close(self, ca.masked_sum(torch.tensor([1.0, float("nan")]), torch.tensor([1.0, 0.0])), 1.0)


class TestAdvantages(unittest.TestCase):
    rewards = torch.tensor([1.0, 0.0, 0.0, 1.0])  # one group of 4 responses to the same prompt

    def outcome(self, length=3):
        tlr = torch.zeros(4, length)
        tlr[:, -1] = self.rewards  # reward sits on the last token
        return tlr, torch.ones(4, length)

    def test_grpo(self):
        tlr, mask = self.outcome()
        adv, ret = ca.compute_grpo_outcome_advantage(tlr, mask, index=["p"] * 4)
        # mean 0.5, std (unbiased) = sqrt(4 * 0.25 / 3) = 0.577350 -> +-0.5 / (0.577350 + 1e-6) = +-0.866024
        close(self, adv[:, 0], [0.866024, -0.866024, -0.866024, 0.866024])
        close(self, adv, adv[:, :1].expand(4, 3))  # the same advantage on every response token
        self.assertTrue(torch.equal(adv, ret))

    def test_dr_grpo_no_std(self):
        tlr, mask = self.outcome()
        adv, _ = ca.compute_grpo_outcome_advantage(tlr, mask, index=["p"] * 4, norm_adv_by_std_in_grpo=False)
        close(self, adv[:, 0], [0.5, -0.5, -0.5, 0.5])

    def test_grpo_groups_and_singletons(self):
        tlr = torch.tensor([[1.0], [0.0], [3.0]])
        adv, _ = ca.compute_grpo_outcome_advantage(tlr, torch.ones(3, 1), index=["a", "a", "b"])
        # group a: mean .5, std sqrt(.5) = .707107 -> +-.707106 ; singleton b: mean 0, std 1 -> 3/(1+1e-6)
        close(self, adv[:, 0], [0.707106, -0.707106, 2.999997])

    def test_all_equal_group_gives_zero(self):
        adv, _ = ca.compute_grpo_outcome_advantage(torch.ones(4, 2), torch.ones(4, 2), index=["p"] * 4)
        close(self, adv, torch.zeros(4, 2))

    def test_rloo(self):
        tlr, mask = self.outcome()
        adv, _ = ca.compute_rloo_outcome_advantage(tlr, mask, index=["p"] * 4)
        # r_i - mean(others): 1 - 1/3 = 0.666667 ; 0 - 2/3 = -0.666667
        close(self, adv[:, 0], [2 / 3, -2 / 3, -2 / 3, 2 / 3])

    def test_reinforce_plus_plus(self):
        tlr = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
        adv, ret = ca.compute_reinforce_plus_plus_outcome_advantage(tlr, torch.ones(2, 3), config=AlgoConfig(gamma=1.0))
        close(self, ret, [[1, 1, 1], [0, 0, 0]])
        # whiten 6 values: mean .5, var 6*.25/5 = .3 -> +-.5/sqrt(.3) = +-0.912871
        close(self, adv, [[0.912871] * 3, [-0.912871] * 3])

    def test_gae_hand_example(self):
        r, v, m = torch.tensor([[0.0, 0.0, 1.0]]), torch.tensor([[0.5, 0.6, 0.7]]), torch.ones(1, 3)
        # deltas: 0+.6-.5=.1, 0+.7-.6=.1, 1-.7=.3 ; lam=1: A=[.5,.4,.3], returns=A+V=[1,1,1]
        adv, ret = ca.compute_gae_advantage_return(r, v, m, gamma=1.0, lam=1.0)
        close(self, ret, [[1.0, 1.0, 1.0]])
        close(self, adv, [[1.0, 0.0, -1.0]], tol=1e-4)  # whitened: mean .4, std .1
        # lam=.95: A2=.3, A1=.1+.95*.3=.385, A0=.1+.95*.385=.46575
        _, ret = ca.compute_gae_advantage_return(r, v, m, gamma=1.0, lam=0.95)
        close(self, ret, [[0.96575, 0.985, 1.0]])

    def test_gae_skips_masked_tool_tokens(self):
        # same episode with an observation token (mask 0, garbage value 9.9) inserted at t=1
        r, v = torch.tensor([[0.0, 0.0, 0.0, 1.0]]), torch.tensor([[0.5, 9.9, 0.6, 0.7]])
        _, ret = ca.compute_gae_advantage_return(r, v, torch.tensor([[1.0, 0.0, 1.0, 1.0]]), 1.0, 0.95)
        close(self, ret[0, [0, 2, 3]], [0.96575, 0.985, 1.0])


class TestKL(unittest.TestCase):
    def test_estimators(self):
        lp, ref = torch.tensor([math.log(0.5)]), torch.tensor([math.log(0.25)])  # x = log 2 = 0.693147
        expected = {"kl": 0.693147, "k1": 0.693147, "abs": 0.693147, "mse": 0.240227, "k2": 0.240227,
                    "low_var_kl": 0.193147, "k3": 0.193147}  # k3 = e^-x + x - 1 = .5 + .693147 - 1
        for name, val in expected.items():
            close(self, ca.kl_penalty(lp, ref, name), [val])
        close(self, ca.kl_penalty(ref, lp, "k3"), [0.306853])  # 2 - .693147 - 1
        close(self, ca.kl_penalty(ref, lp, "k1"), [-0.693147])

    def test_low_var_kl_clamp(self):
        close(self, ca.kl_penalty(torch.tensor([0.0]), torch.tensor([-50.0]), "low_var_kl"), [10.0])

    def test_k3_plus_straight_through(self):
        lp = torch.tensor([math.log(0.5)], requires_grad=True)
        ref = torch.tensor([math.log(0.25)])
        out = ca.kl_penalty(lp, ref, "k3+")
        close(self, out.detach(), [0.193147])  # forward value of k3
        out.sum().backward()
        close(self, lp.grad, [0.693147])        # gradient of k2 = x


class TestLosses(unittest.TestCase):
    loss_mat = torch.tensor([[1.0, 2.0, 3.0, 0.0], [4.0, 0.0, 0.0, 0.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0]])

    def test_agg_modes(self):
        expected = {"token-mean": 2.5,              # (1+2+3+4)/4
                    "token-sum": 10.0,
                    "seq-mean-token-sum": 5.0,      # (6+4)/2
                    "seq-mean-token-mean": 3.0,     # (2+4)/2
                    "seq-mean-token-sum-norm": 1.25}  # 5 / T(=4)
        for mode, val in expected.items():
            close(self, ca.agg_loss(self.loss_mat, self.mask, mode), val)
        close(self, ca.agg_loss(self.loss_mat, self.mask, "seq-mean-token-sum-norm", loss_scale_factor=2), 2.5)

    def test_agg_global_normalisation_is_split_invariant(self):
        for mode, kw in (("token-mean", {"batch_num_tokens": 4}), ("seq-mean-token-mean", {"global_batch_size": 2})):
            parts = [ca.agg_loss(self.loss_mat[i:i + 1], self.mask[i:i + 1], mode, dp_size=2, **kw) for i in (0, 1)]
            close(self, sum(parts) / 2, ca.agg_loss(self.loss_mat, self.mask, mode))  # DP average == full batch

    def test_policy_loss_clip_and_dual_clip(self):
        A = torch.tensor([[1.0, 1.0, -1.0, -1.0]])
        ratio = torch.tensor([[1.5, 0.5, 0.5, 5.0]])
        old = torch.zeros(1, 4)
        # tok0 A>0 r=1.5 -> clipped at 1.2: -1.2 | tok1 A>0 r=.5 -> -0.5 (no clip for A>0 when r<1)
        # tok2 A<0 r=.5 -> clipped at .8: 0.8    | tok3 A<0 r=5 -> 5, dual clip caps it at c=3
        pg_loss, clipfrac, ppo_kl, clipfrac_lower = ca.compute_policy_loss(
            old, ratio.log(), A, torch.ones(1, 4), cliprange=0.2, clip_ratio_c=3.0)
        close(self, pg_loss, (-1.2 - 0.5 + 0.8 + 3.0) / 4)          # 0.525
        close(self, clipfrac, 0.5)                                     # tokens 0 and 2
        close(self, clipfrac_lower, 0.25)                              # token 3
        close(self, ppo_kl, -(math.log(1.5) + 2 * math.log(0.5) + math.log(5.0)) / 4)  # -0.157152

    def test_policy_loss_asymmetric_clip(self):
        # clip_ratio_high=0.28 (DAPO clip-higher): r=1.25 is no longer clipped for A>0
        pg, cf, _, _ = ca.compute_policy_loss(torch.zeros(1, 1), torch.tensor([[math.log(1.25)]]), torch.ones(1, 1),
                                              torch.ones(1, 1), cliprange_low=0.2, cliprange_high=0.28)
        close(self, pg, -1.25)
        close(self, cf, 0.0)

    def test_value_loss(self):
        vpreds, values, returns = torch.tensor([[1.0, 0.6]]), torch.tensor([[0.5, 0.5]]), torch.tensor([[2.0, 0.0]])
        # clip(v, .5 +- .2) = [.7, .6]; (v-R)^2=[1, .36]; (vclip-R)^2=[1.69, .36]; max=[1.69,.36]; .5*mean=.5125
        vf_loss, vf_clipfrac = ca.compute_value_loss(vpreds, returns, values, torch.ones(1, 2), cliprange_value=0.2)
        close(self, vf_loss, 0.5125)
        close(self, vf_clipfrac, 0.5)


class TestKLControllers(unittest.TestCase):
    def test_adaptive_and_fixed(self):
        ctrl = ca.AdaptiveKLController(init_kl_coef=0.1, target_kl=0.05, horizon=100)
        ctrl.update(current_kl=0.1, n_steps=10)    # error clip(2-1)=.2 -> x(1 + .2*10/100) = x1.02
        self.assertAlmostEqual(ctrl.value, 0.102)
        ctrl.update(current_kl=0.045, n_steps=10)  # error -.1 -> x0.99
        self.assertAlmostEqual(ctrl.value, 0.102 * 0.99)
        fixed = ca.FixedKLController(kl_coef=0.001)
        fixed.update(current_kl=5.0, n_steps=10)
        self.assertEqual(fixed.value, 0.001)

    def test_registry(self):
        self.assertIs(ca.get_adv_estimator_fn("grpo"), ca.compute_grpo_outcome_advantage)
        self.assertIs(ca.get_adv_estimator_fn(ca.AdvantageEstimator.GAE), ca.compute_gae_advantage_return)
        with self.assertRaises(ValueError):
            ca.get_adv_estimator_fn("nope")


if __name__ == "__main__":
    unittest.main()

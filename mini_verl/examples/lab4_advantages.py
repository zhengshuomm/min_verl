"""Lab 4 -- Advantage estimators on one hand-sized group: GRPO vs Dr.GRPO vs RLOO vs REINFORCE++ vs GAE.

Run:  python3 -m mini_verl.examples.lab4_advantages

One prompt, rollout.n = 4 responses of 3 tokens each, rule rewards [1, 0, 0, 1] placed on the
last token (as the reward manager does). GAE additionally needs a critic; we use V = 0.5 on every
token (a critic that has learned the average success rate).
"""

import torch

from mini_verl import core_algos as ca
from mini_verl.config import AlgoConfig


def fmt(t):
    return "[" + ", ".join(f"{x:+.6f}" for x in t.tolist()) + "]"


def main():
    rewards = torch.tensor([1.0, 0.0, 0.0, 1.0])
    token_level_rewards = torch.zeros(4, 3)
    token_level_rewards[:, -1] = rewards
    response_mask = torch.ones(4, 3)
    index = ["prompt-0"] * 4
    print("rewards per response:", rewards.tolist(), "(on the last of 3 tokens)\n")

    adv, _ = ca.compute_grpo_outcome_advantage(token_level_rewards, response_mask, index)
    print("GRPO         (r - mean) / (std + 1e-6), std unbiased")
    print("             mean = 0.5, std = sqrt(1/3) = 0.577350")
    print("             A =", fmt(adv[:, 0]), " (same value on all 3 tokens)")

    adv, _ = ca.compute_grpo_outcome_advantage(token_level_rewards, response_mask, index, norm_adv_by_std_in_grpo=False)
    print("Dr.GRPO      r - mean          A =", fmt(adv[:, 0]))

    adv, _ = ca.compute_rloo_outcome_advantage(token_level_rewards, response_mask, index)
    print("RLOO         r - mean(others)  A =", fmt(adv[:, 0]))

    adv, ret = ca.compute_reinforce_plus_plus_outcome_advantage(token_level_rewards, response_mask,
                                                                 config=AlgoConfig(gamma=1.0))
    print("REINFORCE++  return-to-go, whitened over the batch")
    print("             returns =", ret.tolist())
    print("             A[:,0]  =", fmt(adv[:, 0]))

    values = torch.full((4, 3), 0.5)
    adv, ret = ca.compute_gae_advantage_return(token_level_rewards, values, response_mask, gamma=1.0, lam=1.0)
    print("GAE          gamma=1, lam=1, V=0.5: raw A_t = r - 0.5 on every token, then whitened")
    print("             raw A  =", fmt((ret - values)[:, 0]))
    print("             A[:,0] =", fmt(adv[:, 0]), " (12 tokens: std = sqrt(12*0.25/11) = 0.522233)")

    values = torch.tensor([[0.2, 0.5, 0.8]] * 4)
    adv, ret = ca.compute_gae_advantage_return(token_level_rewards, values, response_mask, gamma=1.0, lam=0.95)
    print("\nGAE          gamma=1, lam=0.95, V=[0.2, 0.5, 0.8] (critic grows more confident token by token)")
    for i in (0, 1):
        print(f"  response {i} (r={rewards[i]:.0f}): raw A_t = {fmt((ret - values)[i])}  returns = {fmt(ret[i])}")
    print("  -> unlike GRPO, GAE gives different credit to different tokens of the same response")


if __name__ == "__main__":
    main()

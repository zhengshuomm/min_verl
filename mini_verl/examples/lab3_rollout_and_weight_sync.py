"""Lab 3 -- The rollout engine and the actor -> rollout weight sync.

Run:  python3 -m mini_verl.examples.lab3_rollout_and_weight_sync

1. Warm-start a policy (SFT), start an ActorRolloutRefWorker group with 2 DP ranks.
2. generate_sequences: left-padded prompts + right-padded responses, response_mask,
   position_ids, rollout_log_probs.
3. One GRPO actor update. The rollout engine still holds the OLD weights (it is a separate copy,
   like vLLM in verl), so its log-probs disagree with the actor until update_weights() runs.
"""

from mini_verl.config import PPOConfig, SFTConfig, set_reproducible
from mini_verl.core_algos import compute_grpo_outcome_advantage
from mini_verl.data import RLHFDataset, Tokenizer, collate_fn, make_addition_pairs
from mini_verl.protocol import DataProto
from mini_verl.reward import compute_reward
from mini_verl.sft import warm_start_policy
from mini_verl.single_controller import ClassWithInitArgs, ResourcePool, WorkerGroup
from mini_verl.workers import ActorRolloutRefWorker, forward_micro_batch


def main():
    set_reproducible()
    cfg = PPOConfig()
    tok = Tokenizer()
    policy = warm_start_policy(cfg.actor_rollout_ref.model, cfg.data, SFTConfig())
    wg = WorkerGroup(ResourcePool([2]), ClassWithInitArgs(ActorRolloutRefWorker, config=cfg.actor_rollout_ref,
                                                          role="actor_rollout"))
    wg.init_model(policy.state_dict())

    train, _ = make_addition_pairs(99, 200, 0)
    ds = RLHFDataset([(7, 5), (23, 48)] + train[:14], tok, 6, "arithmetic/addition")
    batch = DataProto.from_single_dict(collate_fn([ds[i] for i in range(len(ds))]), meta_info={"temperature": 1.0})
    batch.non_tensor_batch["uid"] = [str(i) for i in range(len(batch))]
    gen = batch.pop(batch_keys=["input_ids", "attention_mask", "position_ids"]).repeat(4)  # rollout.n = 4
    out = wg.generate_sequences(gen)
    batch = batch.repeat(4).union(out)

    print("== generate_sequences: 16 prompts x n=4 = 64 samples (2 ranks x 32). First 5 rows:")
    print("   prompt + response tokens                 | attention_mask        | response_mask | position_ids")
    for i in [0, 1, 2, 4, 5]:
        text = tok.decode(batch.batch["input_ids"][i].tolist(), skip_special_tokens=False)
        print(f"   {text:<42}| {''.join(map(str, batch.batch['attention_mask'][i].tolist())):<22}| "
              f"{''.join(map(str, batch.batch['response_mask'][i].tolist())):<14}| {batch.batch['position_ids'][i].tolist()}")
    print("   rollout_log_probs[0] =", [round(x, 4) for x in batch.batch["rollout_log_probs"][0].tolist()])

    scores, extra = compute_reward(batch, tok)
    batch.batch["token_level_rewards"] = scores
    j = extra["acc"].index(1.0)
    print(f"\n== reward: {sum(extra['acc']):.0f}/64 samples correct; e.g. row {j} "
          f"{tok.decode(batch.batch['input_ids'][j].tolist())!r}: token_level_scores = {scores[j].tolist()}")
    adv, _ = compute_grpo_outcome_advantage(scores, batch.batch["response_mask"], batch.non_tensor_batch["uid"])
    batch.batch["advantages"] = adv
    batch.union(wg.compute_log_prob(batch))
    batch.batch.pop("entropys")
    batch.meta_info["mini_batch_size"] = len(batch)

    actor, rollout = wg.workers[0].actor_module, wg.workers[0].rollout

    def gap(module):
        lp, _ = forward_micro_batch(module, batch.batch, 1.0)
        m = batch.batch["response_mask"].bool()
        return (lp[m] - batch.batch["rollout_log_probs"][m]).abs().max().item()

    print("\n== weight sync")
    print(f"   before update : max |actor logp - rollout logp| = {gap(actor):.2e}   (weight_version={rollout.weight_version})")
    wg.update_actor(batch)
    print(f"   after update  : actor moved,  rollout still has the old weights:")
    print(f"                   max |actor logp - rollout logp| = {gap(actor):.2e}   (weight_version={rollout.weight_version})")
    print(f"                   max |rollout-module logp - rollout logp| = {gap(rollout.module):.2e}")
    wg.update_weights(1)
    print(f"   update_weights: max |rollout-module logp - actor logp| = "
          f"{(forward_micro_batch(rollout.module, batch.batch, 1.0)[0] - forward_micro_batch(actor, batch.batch, 1.0)[0]).abs().max().item():.2e}"
          f"   (weight_version={rollout.weight_version})")
    same = all((a == b).all() for w in wg.workers for a, b in zip(w.actor_module.parameters(), wg.workers[0].actor_module.parameters()))
    print(f"   both DP ranks hold identical actor weights after the all-reduced update: {bool(same)}")


if __name__ == "__main__":
    main()

"""Lab 9 -- Multi-turn agent loop with a calculator tool, and why tool tokens get response_mask = 0.

Run:  python3 -m mini_verl.examples.lab9_agent_loop        (about 15 s)

1. SFT the tiny model on tool-use episodes  "23+48=" -> "<c>23+48</c><r>71</r>71<eos>"
   (loss_mask = 0 on the <r>...</r> observation, exactly like the RL response_mask).
2. Run ToolAgentLoop: generation stops at </c>, the environment appends <r>result</r>, the model
   continues. Print every response token with its response_mask bit.
3. Plug the agent loop into RayPPOTrainer (actor_rollout_ref.rollout.multi_turn.enable=True) for
   a few GRPO steps.
"""

import copy

from mini_verl.agent_loop import ToolAgentLoop
from mini_verl.config import PPOConfig, SFTConfig, set_reproducible
from mini_verl.data import RLHFDataset, Tokenizer, collate_fn, make_addition_pairs
from mini_verl.protocol import DataProto
from mini_verl.reward import compute_reward
from mini_verl.rollout import Rollout
from mini_verl.sft import greedy_accuracy, warm_start_policy
from mini_verl.trainer import RayPPOTrainer

OVERRIDES = {"data.max_response_length": 16, "data.data_source": "arithmetic/addition_tool",
             "actor_rollout_ref.rollout.multi_turn.enable": True, "algorithm.adv_estimator": "grpo",
             "trainer.total_training_steps": 5, "trainer.test_freq": 5}


def main():
    set_reproducible()
    cfg = PPOConfig().apply_overrides(OVERRIDES)
    d, tok = cfg.data, Tokenizer()
    policy = warm_start_policy(cfg.actor_rollout_ref.model, d, SFTConfig(steps=150), use_tool=True)
    _, val_pairs = make_addition_pairs(d.operand_max, d.val_size, d.seed)

    ds = RLHFDataset([(23, 48), (7, 5), (99, 99)], tok, d.max_prompt_length, d.data_source)
    batch = DataProto.from_single_dict(collate_fn([ds[i] for i in range(len(ds))]))
    gen = batch.pop(batch_keys=["input_ids", "attention_mask", "position_ids"])
    gen.meta_info = {"do_sample": False}
    rollout = Rollout(copy.deepcopy(policy), cfg.actor_rollout_ref.rollout, tok, seed=0)
    out = ToolAgentLoop(rollout, tok, cfg.actor_rollout_ref.rollout).generate_sequences(gen)
    batch.union(out)
    scores, extra = compute_reward(batch, tok)
    print("== ToolAgentLoop, greedy (max_assistant_turns=2, max_user_turns=1, response_length=16)")
    for i in range(len(batch)):
        n = int(batch.batch["attention_mask"][i, d.max_prompt_length:].sum())
        toks = [tok.vocab[t] for t in batch.batch["responses"][i, :n].tolist()]
        mask = batch.batch["response_mask"][i, :n].tolist()
        print(f"  {batch.non_tensor_batch['raw_prompt'][i]:<7} turns={batch.non_tensor_batch['__num_turns__'][i]} "
              f"reward={extra['acc'][i]:.0f}")
        print("    token: " + " ".join(f"{t:>4}" for t in toks))
        print("    mask : " + " ".join(f"{m:>4}" for m in mask))
    print("  -> <r>..</r> came from the environment: in the context (attention_mask=1) but not in the loss")

    acc_tool = greedy_accuracy(policy, val_pairs, tok, d.max_prompt_length, d.max_response_length, d.data_source,
                               multi_turn=True)
    print(f"\ngreedy val acc WITH the calculator: {acc_tool:.3f}   (Lab 7's SFT model without tools: ~0.30)")

    print("\n== 5 multi-turn GRPO steps through RayPPOTrainer")
    trainer = RayPPOTrainer(cfg, policy_state_dict=policy.state_dict())
    trainer.init_workers()
    history = trainer.fit(log_every=1)
    print(f"  sampled score (temperature 1.0): {history[1]['critic/score/mean']:.3f} -> "
          f"{history[-1]['critic/score/mean']:.3f}; response_length/mean counts tool tokens too "
          f"({history[-1]['response_length/mean']:.2f}), the loss does not")


if __name__ == "__main__":
    main()

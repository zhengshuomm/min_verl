"""Multi-turn tool loop: tool-response tokens are in ``responses`` but masked out of the loss."""

import unittest

import torch

from mini_verl.agent_loop import ToolAgentLoop, calculator
from mini_verl.config import RolloutConfig
from mini_verl.core_algos import compute_gae_advantage_return, compute_policy_loss
from mini_verl.data import Tokenizer
from mini_verl.protocol import DataProto
from mini_verl.reward import compute_reward
from mini_verl.rollout import Rollout


class ScriptedRollout(Rollout):
    """Replays fixed model outputs: script[call][row] (rows = samples still GENERATING)."""

    def __init__(self, script):
        self.tokenizer, self.script, self.calls, self.seen = Tokenizer(), script, 0, []

    def generate_tokens(self, prompts, max_new_tokens, stop_token_ids, do_sample, temperature, top_k=-1):
        self.seen.append([self.tokenizer.decode(p, False) for p in prompts])
        out = [self.tokenizer.encode(s)[:m] for s, m in zip(self.script[self.calls], max_new_tokens)]
        self.calls += 1
        return out, [[-0.1] * len(t) for t in out]


def run(script, response_length=16, prompts=("23+48=", "5+7=")):
    tok = Tokenizer()
    cfg = RolloutConfig(response_length=response_length)
    loop = ToolAgentLoop(ScriptedRollout(script), tok, cfg)
    ids = torch.full((len(prompts), 6), tok.pad_token_id)
    mask = torch.zeros(len(prompts), 6, dtype=torch.long)
    for i, p in enumerate(prompts):
        e = tok.encode(p)
        ids[i, 6 - len(e):], mask[i, 6 - len(e):] = torch.tensor(e), 1
    out = loop.generate_sequences(DataProto.from_dict({"input_ids": ids, "attention_mask": mask}))
    return loop, out, tok


class TestAgentLoop(unittest.TestCase):
    def test_tool_tokens_are_masked(self):
        # row 0 calls the calculator then answers; row 1 answers directly
        loop, out, tok = run([["<c>23+48</c>", "12<eos>"], ["71<eos>"]])
        self.assertEqual(tok.decode(out.batch["responses"][0].tolist(), False),
                         "<c>23+48</c><r>71</r>71<eos><pad><pad>")
        self.assertEqual(out.batch["response_mask"][0].tolist(), [1] * 7 + [0] * 4 + [1] * 3 + [0] * 2)
        self.assertEqual(out.batch["response_mask"][1].tolist(), [1] * 3 + [0] * 13)
        # tool tokens ARE part of the context: attention_mask covers them, positions are contiguous
        self.assertEqual(out.batch["attention_mask"][0, 6:].tolist(), [1] * 14 + [0] * 2)
        self.assertEqual(out.batch["position_ids"][0, :20].tolist(), list(range(20)))
        self.assertEqual(out.batch["rollout_log_probs"][0, 7:11].tolist(), [0.0] * 4)
        self.assertEqual(out.non_tensor_batch["__num_turns__"], [4, 2])
        # the second turn saw the tool result in its context
        self.assertEqual(loop.rollout.seen[1], ["23+48=<c>23+48</c><r>71</r>"])

    def test_reward_on_last_valid_token(self):
        _, out, _ = run([["<c>23+48</c>", "71<eos>"], ["5+7=<eos>"]], prompts=("23+48=", "70+1="))
        out.non_tensor_batch["data_source"] = ["arithmetic/addition_tool"] * 2
        out.non_tensor_batch["reward_model"] = [{"ground_truth": "71"}] * 2
        scores, extra = compute_reward(out, Tokenizer())
        self.assertEqual(extra["acc"], [0.0, 1.0])  # row 0 answers "5+7=" after the tool: malformed
        self.assertEqual(scores[1].tolist(), [0, 0, 1.0] + [0] * 13)

    def test_masked_tokens_get_no_gradient_or_td_error(self):
        _, out, _ = run([["<c>23+48</c>", "71<eos>"], ["71<eos>"]])
        mask = out.batch["response_mask"].float()
        log_prob = torch.zeros(mask.shape, requires_grad=True)
        loss, *_ = compute_policy_loss(torch.zeros(mask.shape), log_prob, torch.ones(mask.shape), mask, cliprange=0.2)
        loss.backward()
        self.assertTrue(torch.all(log_prob.grad[mask == 0] == 0))
        self.assertTrue(torch.all(log_prob.grad[mask == 1] != 0))
        rewards = torch.zeros(mask.shape)
        rewards[0, 13] = 1.0
        values = torch.rand(mask.shape) * 10 * (1 - mask)   # garbage values on tool / pad tokens
        _, ret = compute_gae_advantage_return(rewards, values, mask, gamma=1.0, lam=1.0)
        self.assertTrue(torch.allclose(ret[0][mask[0] == 1], torch.ones(10)))

    def test_truncation_and_bad_tool_call(self):
        _, out, tok = run([["<c>23+48</c>"], []], response_length=9, prompts=("23+48=",))
        self.assertEqual(tok.decode(out.batch["responses"][0].tolist(), False), "<c>23+48</c><r>7")
        self.assertEqual(out.batch["response_mask"][0].tolist(), [1] * 7 + [0] * 2)
        _, out, tok = run([["<c>2+</c>"], ["0<eos>"]], prompts=("2+2=",))
        self.assertIn("<c>2+</c><r></r>0<eos>", tok.decode(out.batch["responses"][0].tolist(), False))
        self.assertEqual(calculator("12+30+1"), "43")


if __name__ == "__main__":
    unittest.main()

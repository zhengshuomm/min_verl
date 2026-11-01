"""A toy multi-turn tool-calling agent loop with verl's response_mask semantics.

Mirrors ``verl/experimental/agent_loop/tool_agent_loop.py`` (``ToolAgentLoop``, ``AgentState``,
``AgentData``) and ``verl/experimental/agent_loop/agent_loop.py`` (``AgentLoopOutput``,
``_postprocess``). The model may emit ``<c>expr</c>``; generation then stops, the environment runs
a calculator and appends ``<r>result</r>`` as *observation* tokens, and the model continues::

    responses:     <c> 2 3 + 4 8 </c> | <r> 7 1 </r> | 7 1 <eos> | <pad> ...
    response_mask:  1  1 1 1 1 1  1   |  0  0 0  0   | 1 1  1    |  0 ...

response_mask is 1 only on model-generated tokens, so the policy loss, the KL and the GAE
recursion all skip the tool tokens (the model did not choose them).
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List

from .config import RolloutConfig
from .data import Tokenizer
from .protocol import DataProto
from .rollout import Rollout, build_rollout_batch, strip_left_padding


class AgentState(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    PROCESSING_TOOLS = "processing_tools"
    TERMINATED = "terminated"


@dataclass
class AgentData:
    prompt_ids: List[int]
    response_ids: List[int] = field(default_factory=list)
    response_mask: List[int] = field(default_factory=list)
    response_logprobs: List[float] = field(default_factory=list)
    assistant_turns: int = 0
    user_turns: int = 0
    state: AgentState = AgentState.GENERATING


@dataclass
class AgentLoopOutput:
    prompt_ids: List[int]
    response_ids: List[int]     # LLM tokens AND tool-response tokens
    response_mask: List[int]    # 1 for LLM generated token, 0 for tool response token
    response_logprobs: List[float]
    num_turns: int = 0


def calculator(expr: str) -> str:
    """The only tool: sums '+'-separated integers; anything else returns '' (an error)."""
    return str(sum(int(x) for x in expr.split("+"))) if re.fullmatch(r"\d+(\+\d+)*", expr) else ""


class ToolAgentLoop:
    def __init__(self, rollout: Rollout, tokenizer: Tokenizer, config: RolloutConfig):
        self.rollout, self.tokenizer, self.config = rollout, tokenizer, config
        self.response_length = config.response_length
        self.max_assistant_turns = config.multi_turn.max_assistant_turns
        self.max_user_turns = config.multi_turn.max_user_turns
        tid = tokenizer.token_to_id
        self.call_start, self.call_end = tid["<c>"], tid["</c>"]
        self.resp_start, self.resp_end = tid["<r>"], tid["</r>"]

    def run(self, prompt_ids_list: List[List[int]], do_sample: bool, temperature: float) -> List[AgentLoopOutput]:
        """Run every sample's state machine; all samples in GENERATING state decode as one batch."""
        agents = [AgentData(prompt_ids=list(p)) for p in prompt_ids_list]
        while True:
            gen = [a for a in agents if a.state == AgentState.GENERATING]
            if not gen:
                break
            toks, lps = self.rollout.generate_tokens(
                [a.prompt_ids + a.response_ids for a in gen], [self.response_length - len(a.response_ids) for a in gen],
                {self.tokenizer.eos_token_id, self.call_end}, do_sample, temperature, self.config.top_k)
            for a, t, lp in zip(gen, toks, lps):
                a.state = self._handle_generating_state(a, t, lp)
                if a.state == AgentState.PROCESSING_TOOLS:
                    a.state = self._handle_processing_tools_state(a)
        return [AgentLoopOutput(a.prompt_ids, a.response_ids[:self.response_length],
                                a.response_mask[:self.response_length], a.response_logprobs[:self.response_length],
                                num_turns=1 + a.assistant_turns + a.user_turns) for a in agents]

    def _handle_generating_state(self, a: AgentData, tokens, logprobs) -> AgentState:
        a.response_ids += tokens
        a.response_mask += [1] * len(tokens)          # the model chose these tokens
        a.response_logprobs += logprobs
        a.assistant_turns += 1
        if len(a.response_mask) >= self.response_length:
            return AgentState.TERMINATED
        if self.max_assistant_turns and a.assistant_turns >= self.max_assistant_turns:
            return AgentState.TERMINATED
        if self.max_user_turns and a.user_turns >= self.max_user_turns:
            return AgentState.TERMINATED
        if tokens and tokens[-1] == self.call_end:
            return AgentState.PROCESSING_TOOLS
        return AgentState.TERMINATED

    def _handle_processing_tools_state(self, a: AgentData) -> AgentState:
        ids = a.response_ids
        start = max(i for i, t in enumerate(ids) if t == self.call_start) if self.call_start in ids else len(ids)
        result = calculator(self.tokenizer.decode(ids[start + 1:-1]))
        obs = [self.resp_start] + self.tokenizer.encode(result) + [self.resp_end]
        a.response_ids += obs
        a.response_mask += [0] * len(obs)             # the environment wrote these tokens
        a.response_logprobs += [0.0] * len(obs)
        a.user_turns += 1
        if len(a.response_mask) >= self.response_length:
            return AgentState.TERMINATED
        return AgentState.GENERATING

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Same contract as ``Rollout.generate_sequences`` (AgentLoopManager.generate_sequences)."""
        do_sample = prompts.meta_info.get("do_sample", self.config.do_sample)
        temperature = prompts.meta_info.get("temperature", self.config.temperature)
        outs = self.run(strip_left_padding(prompts.batch["input_ids"], prompts.batch["attention_mask"]),
                        do_sample, temperature)
        batch = build_rollout_batch([o.prompt_ids for o in outs], [o.response_ids for o in outs],
                                    [o.response_mask for o in outs], [o.response_logprobs for o in outs],
                                    prompts.batch["input_ids"].size(1), self.response_length,
                                    self.tokenizer.pad_token_id)
        batch.non_tensor_batch["__num_turns__"] = [o.num_turns for o in outs]
        return batch

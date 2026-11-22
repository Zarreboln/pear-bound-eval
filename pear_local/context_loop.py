"""PEAR's agent loop with managed context: before every generation the prompt is rebuilt from the message list, with
the reasoning of old rounds removed, an optional one-line progress note per round, and an optional context cap.

Settings come from the workers' environment (eval_long.sbatch passes them as Ray runtime env vars):
    CTX_K        rounds whose reasoning is kept, the most recent ones; -1 keeps all (the rebuild path itself, which
                 reproduces the official prompt token for token)
    CTX_SUMMARY  1: the system prompt asks for a "Progress:" line after the thinking, kept when the reasoning goes
    CTX_CAP      tokens; when the rebuilt context is longer, whole rounds are evicted, oldest first (0: no cap)
    CTX_GEN_CAP  generated tokens per episode; the episode ends (budget) once it has written this many
The 256k budget applies to the context the model sees (for the full context, the same as the official rule), and
CTX_GEN_CAP bounds what an episode may generate in total.

The system prompt, the task and the first observation are never evicted. The rebuild renders assistant turns with
reasoning_content and preserve_thinking, which matches the official incremental tokens exactly (ctx_probe.py); a
round without its reasoning renders as an empty <think></think> block, as Qwen's template renders past turns.

Each episode's record gets "ctx": the settings, peak context, generated and reasoning tokens per turn, and the
context length at every rebuild.
"""
from __future__ import annotations

import os
from typing import Any

from pear.agent_loop.agent_loop import PearAgentLoop
from pear.agent_loop.episode import EndReason
from pear.utils.logging_utils import get_logger
from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState
from verl.utils.tokenizer.continuous_token_wiring import create_continuous_token_builder

logger = get_logger(__name__)

THINK_END = "</think>"
SUMMARY_RULE = """

## PROGRESS NOTE

Your reasoning from earlier turns is removed from your context; your actions and these notes remain. After your \
thinking and before the json block, write one line that starts with "Progress:" and says in one sentence what the \
last action achieved and what remains to be done."""


class ContextAgentLoop(PearAgentLoop):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.k = int(os.environ.get("CTX_K", "-1"))
        self.summary = os.environ.get("CTX_SUMMARY", "0") == "1"
        self.cap = int(os.environ.get("CTX_CAP", "0"))
        self.gen_cap = int(os.environ.get("CTX_GEN_CAP", "1000000"))
        # the rebuild renders past reasoning itself (reasoning_content), so the template must keep it
        self.ctx_builder = create_continuous_token_builder(
            self.tokenizer,
            hf_model_type=kwargs.get("hf_model_type"),
            chat_template_kwargs={**self.apply_chat_template_kwargs, "preserve_thinking": True},
            mm_processor_kwargs=self.mm_processor_kwargs,
            processor=self.processor,
        )
        self.im_start = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.ctx: dict[str, Any] = {}
        self.initial_ids: list[int] = []

    async def run(self, sampling_params: dict[str, Any], **kwargs):
        self.ctx = {"k": self.k, "summary": self.summary, "cap": self.cap, "gen_cap": self.gen_cap, "peak": 0,
                    "generated": [], "reasoning": [], "rebuilt": [], "evicted_rounds": 0, "gen_capped": False}
        output = await super().run(sampling_params, **kwargs)
        output.extra_fields["episode"]["ctx"] = self.ctx
        return output

    async def _handle_pending_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        state = await super()._handle_pending_state(agent_data, sampling_params)
        if self.summary:  # the rule joins the system prompt before anything is generated
            agent_data.messages[0] = {**agent_data.messages[0],
                                      "content": agent_data.messages[0]["content"] + SUMMARY_RULE}
            agent_data.prompt_ids = await self.ct_build_initial_tokens(
                agent_data.messages, tools=[], images=agent_data.image_data)
        self.initial_ids = list(agent_data.prompt_ids)
        self.ctx["peak"] = len(self.initial_ids)
        return state

    async def _handle_generating_state(
        self, agent_data: AgentData, sampling_params: dict[str, Any], ignore_termination: bool = False
    ) -> AgentState:
        if agent_data.assistant_turns:  # the first prompt is the initial one
            await self.loop.run_in_executor(None, self._rebuild, agent_data)

        state = await super()._handle_generating_state(agent_data, sampling_params, ignore_termination)

        text = await self.loop.run_in_executor(None, self.tokenizer.decode, agent_data.response_ids)
        reasoning = text.partition(THINK_END)[0] if THINK_END in text else text
        self.ctx["generated"].append(len(agent_data.response_ids))
        self.ctx["reasoning"].append(len(self.tokenizer(reasoning, add_special_tokens=False).input_ids))
        self.ctx["peak"] = max(self.ctx["peak"], len(agent_data.prompt_ids))

        if state != AgentState.TERMINATED and sum(self.ctx["generated"]) >= self.gen_cap:
            self.ctx["gen_capped"] = True
            self.end_reason = EndReason.BUDGET
            return AgentState.TERMINATED
        return state

    def _rebuild(self, agent_data: AgentData) -> None:
        """Rebuilds agent_data's prompt (and images) from its messages under the settings."""
        messages = agent_data.messages
        prefix, rounds = messages[:2], [messages[i:i + 2] for i in range(2, len(messages), 2)]
        n = len(rounds)

        def render(kept: list[list[dict]]) -> tuple[list[int], list]:
            out = list(prefix)
            first = n - len(kept)  # index of the first kept round
            for j, (assistant, user) in enumerate(kept):
                keep_reasoning = self.k < 0 or first + j >= n - self.k
                out += [self._assistant(assistant, keep_reasoning), user]
            images = [b["image"] for m in out if isinstance(m["content"], list)
                      for b in m["content"] if b.get("type") == "image"]
            return self.ctx_builder.build_initial_tokens(out, tools=[], images=images), images

        kept = rounds
        ids, images = render(kept)
        while self.cap and len(ids) > self.cap and len(kept) > 1:  # whole rounds, oldest first; the last one stays
            kept = kept[1:]
            ids, images = render(kept)
        self.ctx["evicted_rounds"] = max(self.ctx["evicted_rounds"], n - len(kept))

        # the fixed prefix: the initial prompt up to its generation prompt (<|im_start|>assistant\n<think>\n), which a
        # rebuilt round renders differently (<think>\n\n</think> without its reasoning)
        cut = len(self.initial_ids) - self.initial_ids[::-1].index(self.im_start) - 1
        if ids[:cut] != self.initial_ids[:cut]:
            logger.warning("CTX_PREFIX_MISMATCH env=%s turn=%d", self.episode.env_args, n)
        agent_data.prompt_ids = ids
        agent_data.image_data = images
        # verl's response = every token after the initial prompt; an eval trains on nothing, so the mask only sizes it
        agent_data.response_mask = [1] * (len(ids) - len(self.initial_ids))
        if agent_data.response_logprobs:
            agent_data.response_logprobs = [0.0] * len(agent_data.response_mask)
        self.ctx["rebuilt"].append(len(ids))
        self.ctx["peak"] = max(self.ctx["peak"], len(ids))

    @staticmethod
    def _assistant(message: dict, keep_reasoning: bool) -> dict:
        """An assistant message as the template renders it: reasoning in reasoning_content, the answer in content.
        verl's tool_calls entry is left out: the official prompt carries the generated text only, never a rendered
        <tool_call> block."""
        text = message["content"]
        if THINK_END not in text:  # a turn cut off inside its thinking: all of it is reasoning
            reasoning, answer = text, ""
        else:
            reasoning, _, answer = text.partition(THINK_END)
        return {"role": "assistant", "reasoning_content": reasoning.strip("\n") if keep_reasoning else "",
                "content": answer.lstrip("\n")}

"""CPU test of ContextAgentLoop._rebuild on a real episode of a val dump (dummy frames), without Ray or a server.

    python ctx_loop_test.py <model dir> <val dump jsonl> [rounds]
Builds the official incremental tokens of the first rounds (as ctx_probe.py), then calls _rebuild on the message list
(with verl's tool_calls entries, as the loop holds them) under several settings and checks:
k=-1 reproduces the official tokens; k=0 and k=1 shrink the context; a cap evicts whole rounds and keeps the prefix.
"""
import json
import re
import sys
import types

from PIL import Image
from transformers import AutoProcessor, AutoTokenizer

from pear.agent_loop.context_loop import ContextAgentLoop
from verl.utils.tokenizer.continuous_token_wiring import create_continuous_token_builder

model, dump = sys.argv[1], sys.argv[2]
n_rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 6
tok = AutoTokenizer.from_pretrained(model)
proc = AutoProcessor.from_pretrained(model)
hf_type = json.load(open(f"{model}/config.json"))["model_type"]
official = create_continuous_token_builder(tok, hf_model_type=hf_type, chat_template_kwargs={"enable_thinking": True},
                                           processor=proc)

r = json.loads(open(dump).readline())
system = r["input"].split("system\n", 1)[1].split("\nuser\n", 1)[0]
REMINDER = "The current camera frames follow. End your response with one json block containing the next action."


def obs(state_text):
    frame = lambda c: Image.new("RGB", (512, 512), c)
    return [{"type": "text", "text": f"{state_text}\n\n{REMINDER}"},
            {"type": "text", "text": "[agentview]"}, {"type": "image", "image": frame("gray")},
            {"type": "text", "text": "[wrist]"}, {"type": "image", "image": frame("white")}]


first_state = r["input"].split("\nuser\n", 1)[1].split("\n\n", 1)[0]
parts = re.split(r"\nuser\n(.*?)\n\n" + re.escape(REMINDER) + r"\[agentview\]\[wrist\]\nassistant\n<think>\n", r["output"],
                 flags=re.S)
answers, states = parts[0::2], parts[1::2]
messages = [{"role": "system", "content": system}, {"role": "user", "content": obs(first_state)}]
images = [b["image"] for b in messages[1]["content"] if b.get("type") == "image"]
ids = official.build_initial_tokens(messages, images=images)
initial = list(ids)
cut = len(initial) - initial[::-1].index(tok.convert_tokens_to_ids("<|im_start|>")) - 1
end = tok.convert_tokens_to_ids("<|im_end|>")
for i in range(n_rounds):
    gen = tok.encode(answers[i], add_special_tokens=False) + [end]
    ids = official.merge_assistant_tokens(ids, gen).token_ids
    messages.append({"role": "assistant", "content": tok.decode(gen, skip_special_tokens=True),
                     "tool_calls": [{"type": "function", "function": {"name": "act", "arguments": "{}"}}]})
    user = {"role": "user", "content": obs(states[i])}
    ids = official.merge_context_tokens(messages, messages + [user], ids).token_ids
    messages.append(user)


def rebuild(k, cap=0):
    loop = ContextAgentLoop.__new__(ContextAgentLoop)
    loop.k, loop.cap, loop.tokenizer = k, cap, tok
    loop.im_start = tok.convert_tokens_to_ids("<|im_start|>")
    loop.ctx_builder = create_continuous_token_builder(
        tok, hf_model_type=hf_type, chat_template_kwargs={"enable_thinking": True, "preserve_thinking": True},
        processor=proc)
    loop.ctx = {"evicted_rounds": 0, "rebuilt": [], "peak": 0}
    loop.initial_ids = initial
    loop.episode = types.SimpleNamespace(env_args={})
    data = types.SimpleNamespace(messages=messages, prompt_ids=None, image_data=None, response_mask=None,
                                 response_logprobs=None)
    loop._rebuild(data)
    return data, loop.ctx


data, _ = rebuild(-1)
print(f"k=-1: {len(data.prompt_ids)} tokens, identical to official {len(ids)}: {data.prompt_ids == ids}, "
      f"images {len(data.image_data)}, mask {len(data.response_mask)} = {len(ids) - len(initial)}")
for k in (0, 1):
    data, _ = rebuild(k)
    print(f"k={k}: {len(data.prompt_ids)} tokens, prefix kept: {data.prompt_ids[:cut] == initial[:cut]}")
    if k == 1:
        text = tok.decode(data.prompt_ids)
        print("  think blocks with content:", len(re.findall(r"<think>\n(?!\n</think>)", text)) - 0,
              "| empty think blocks:", text.count("<think>\n\n</think>"),
              "| <tool_call> rendered:", "<tool_call>" in text)
cap = len(rebuild(0)[0].prompt_ids) - 1500
data, ctx = rebuild(0, cap)
print(f"k=0 cap {cap}: {len(data.prompt_ids)} tokens, evicted rounds {ctx['evicted_rounds']}, images {len(data.image_data)}, "
      f"prefix kept: {data.prompt_ids[:cut] == initial[:cut]}")

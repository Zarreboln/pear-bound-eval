"""CPU probe: does rebuilding the prompt from the message list reproduce the tokens the official loop accumulates?

    python ctx_probe.py <model dir> <val dump jsonl> [rounds]
Takes the first episode of a val dump, splits its text into rounds, and builds the prompt two ways with verl's
Continuous Token builder (dummy 512 x 512 frames): (a) incrementally, as the official loop does (initial tokens, then
merge_assistant_tokens / merge_context_tokens per round); (b) in one go from the message list with preserve_thinking.
Prints whether they match and where they first differ, then shows how an assistant turn renders with its reasoning
removed.
"""
import json
import re
import sys

from PIL import Image
from transformers import AutoProcessor, AutoTokenizer

from verl.utils.tokenizer.continuous_token_wiring import create_continuous_token_builder

model, dump = sys.argv[1], sys.argv[2]
n_rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 3
tok = AutoTokenizer.from_pretrained(model)
proc = AutoProcessor.from_pretrained(model)
hf_type = json.load(open(f"{model}/config.json"))["model_type"]
kw = {"enable_thinking": True}
official = create_continuous_token_builder(tok, hf_model_type=hf_type, chat_template_kwargs=kw, processor=proc)
rebuild = create_continuous_token_builder(tok, hf_model_type=hf_type, chat_template_kwargs={**kw, "preserve_thinking": True},
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
# output: <answer 1>\nuser\n<state 2>\n\n<reminder>[agentview][wrist]\nassistant\n<think>\n<answer 2>...
parts = re.split(r"\nuser\n(.*?)\n\n" + re.escape(REMINDER) + r"\[agentview\]\[wrist\]\nassistant\n<think>\n", r["output"],
                 flags=re.S)
answers, states = parts[0::2], parts[1::2]
messages = [{"role": "system", "content": system}, {"role": "user", "content": obs(first_state)}]
images = [b["image"] for b in messages[1]["content"] if b.get("type") == "image"]
ids = official.build_initial_tokens(messages, images=images)
end = tok.convert_tokens_to_ids("<|im_end|>")
for i in range(n_rounds):
    gen = tok.encode(answers[i], add_special_tokens=False) + [end]
    ids = official.merge_assistant_tokens(ids, gen).token_ids
    messages.append({"role": "assistant", "content": tok.decode(gen, skip_special_tokens=True)})
    user = {"role": "user", "content": obs(states[i])}
    ids = official.merge_context_tokens(messages, messages + [user], ids).token_ids
    messages.append(user)
    images += [b["image"] for b in user["content"] if b.get("type") == "image"]

def split(m):
    """An assistant message as the template wants it: reasoning in reasoning_content, the answer in content."""
    if m["role"] != "assistant" or "</think>" not in m["content"]:
        return m
    reasoning, _, answer = m["content"].partition("</think>")
    return {**m, "reasoning_content": reasoning.strip("\n"), "content": answer.lstrip("\n")}


full = rebuild.build_initial_tokens([split(m) for m in messages], images=images)
same = full == ids
print(f"rounds {n_rounds}: incremental {len(ids)} tokens, rebuilt {len(full)} tokens, identical: {same}")
if not same:
    k = next((j for j, (a, b) in enumerate(zip(ids, full)) if a != b), min(len(ids), len(full)))
    print("first difference at", k)
    print("incremental:", repr(tok.decode(ids[max(0, k - 40):k + 40])))
    print("rebuilt:    ", repr(tok.decode(full[max(0, k - 40):k + 40])))

# an assistant turn with its reasoning removed: keep what follows </think>
stripped = [dict(split(m), reasoning_content="") if m["role"] == "assistant" else m for m in messages]
short = rebuild.build_initial_tokens(stripped, images=images)
text = tok.decode(short)
i = text.find("<|im_start|>assistant")
print(f"reasoning removed from all {n_rounds} turns: {len(short)} tokens")
print(repr(text[i:i + 300]))
print("generation prompt at the end:", repr(tok.decode(short[-8:])))

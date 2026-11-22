"""Per-episode breakdown of a PEAR LIBERO eval run: outcome, end reason, rounds, and where the context tokens go.

    python analyze_run.py <run dir> <model dir>  -> prints a table + totals, writes <run dir>/analysis/episodes.json
Token categories of the response (everything after the initial prompt), counted with the model's tokenizer:
reasoning (<think> ... </think>), action (the json block), observation text (state json + reminder + camera labels),
images (2 frames per observation x 256 tokens: model_image_tokens(512, 512) of the model file), template (the rest).
"""
import ast
import json
import os
import re
import sys
from collections import Counter

from transformers import AutoTokenizer

IMAGE_TOKENS = 256  # one 512 x 512 frame: (512+16)//32 = 16 columns x 16 rows
FRAMES_PER_OBS = 2  # agentview, wrist

run, model = sys.argv[1], sys.argv[2]
tok = AutoTokenizer.from_pretrained(model)
n = lambda s: len(tok(s, add_special_tokens=False).input_ids)

ends = {}
for line in open(os.path.join(run, "train.log"), errors="ignore"):
    m = re.search(r"EPISODE_END (\{.*\})", line)
    if m:
        rec = ast.literal_eval(m.group(1))
        ends[rec["uid"]] = rec

rows = []
for line in open(os.path.join(run, "val_dumps", "eval_0.jsonl")):
    r = json.loads(line)
    ep, out = r["extra_fields"]["episode"], r["output"]
    end = ends.get(r["uid"], {})
    reasoning = re.findall(r"<think>\n?(.*?)</think>", out, flags=re.S)
    if "<think>" in out.rsplit("</think>", 1)[-1]:  # a last turn cut off inside its thinking (budget)
        reasoning.append(out.rsplit("<think>", 1)[-1])
    actions = re.findall(r"```json.*?```", out, flags=re.S)
    obs = re.findall(r"\nuser\n(.*?)\nassistant\n", out, flags=re.S)
    c = Counter(
        reasoning=sum(map(n, reasoning)),
        action=sum(map(n, actions)),
        obs_text=sum(map(n, obs)),
        images=len(obs) * FRAMES_PER_OBS * IMAGE_TOKENS,
    )
    c["template"] = max(0, n(out) - c["reasoning"] - c["action"] - c["obs_text"])
    ctx = ep.get("ctx")
    if ctx:  # managed context (context_loop.py): the dump holds the final managed prompt, the record the full counts
        c = Counter(reasoning=sum(ctx["reasoning"]))
    rows.append({
        "task": ep["env"]["task_id"], "trial": ep["env"]["trial_id"], "success": ep["success"],
        "end_reason": end.get("end_reason", "?"), "rounds": ep["num_chunks"], "steps": ep["executed_steps"],
        "prompt_tokens": n(r["input"]) + FRAMES_PER_OBS * IMAGE_TOKENS, "response_tokens": sum(c.values()),
        **c, "uid": r["uid"], "episode_uid": ep["uid"],
        **({"peak_context_tokens": ctx["peak"], "generated": sum(ctx["generated"]), "gen_capped": ctx["gen_capped"],
            "evicted_rounds": ctx["evicted_rounds"]} if ctx else {}),
    })

rows.sort(key=lambda x: (x["task"], x["trial"]))
os.makedirs(os.path.join(run, "analysis"), exist_ok=True)
json.dump(rows, open(os.path.join(run, "analysis", "episodes.json"), "w"), indent=1)

cols = ["task", "trial", "success", "end_reason", "rounds", "steps", "response_tokens", "reasoning", "action",
        "obs_text", "images", "template"]
print("\t".join(cols))
for x in rows:
    print("\t".join(str(x.get(k, "")) for k in cols))
tot = Counter()
for x in rows:
    tot.update({k: x.get(k, 0) for k in ["reasoning", "action", "obs_text", "images", "template", "response_tokens"]})
print("\nshare of response tokens:", {k: f"{tot[k] / max(1, tot['response_tokens']):.1%}" for k in
                                     ["reasoning", "action", "obs_text", "images", "template"]})
print("end reasons:", Counter(x["end_reason"] for x in rows), "| success:", sum(x["success"] for x in rows), "/", len(rows))
for reason in ["env", "budget"]:
    sub = [x for x in rows if x["end_reason"] == reason]
    if sub:
        print(f"{reason}: n={len(sub)} rounds mean {sum(x['rounds'] for x in sub) / len(sub):.1f}, "
              f"response tokens mean {sum(x['response_tokens'] for x in sub) / len(sub):,.0f}, "
              f"tokens/round {sum(x['response_tokens'] for x in sub) / max(1, sum(x['rounds'] for x in sub)):,.0f}")

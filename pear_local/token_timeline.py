"""Round-by-round token timeline of one episode of a PEAR LIBERO run: reasoning tokens, other tokens (action, template,
observation text, 2 x 256 image tokens) and the running total against the 256k budget.

    python token_timeline.py <run dir> <model dir> <task> <trial>
"""
import glob
import json
import re
import sys

from transformers import AutoTokenizer

run, model, task, trial = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
tok = AutoTokenizer.from_pretrained(model)
n = lambda s: len(tok(s, add_special_tokens=False).input_ids)
for line in open(f"{run}/val_dumps/eval_0.jsonl"):
    r = json.loads(line)
    e = r["extra_fields"]["episode"]["env"]
    if (e["task_id"], e["trial_id"]) == (task, trial):
        out = r["output"]
calls = sorted([c for p in glob.glob(f"{run}/sim/actions_*.jsonl") for c in map(json.loads, open(p))
                if c["action"] is not None and (c["env"]["task_id"], c["env"]["trial_id"]) == (task, trial)],
               key=lambda c: c["steps_before"])
segs = [s for s in re.split(r"(?=<think>)", out) if s.strip()]  # round i: reasoning, action, next observation
cum = 0
for i, s in enumerate(segs):
    th = re.search(r"<think>(.*?)(?:</think>|$)", s, flags=re.S)
    t = n(th.group(1)) if th else 0
    other = n(s) - t + (512 if i < len(calls) else 0)
    cum += t + other
    if i < len(calls):
        c = calls[i]
        arm = ",".join(f"{k}{v:+g}" for k, v in c["action"]["arm"].items() if v)
        what = f"{arm or '-'} grip {c['action']['gripper']['width']} -> {c['outcome']}, step {c['steps_before'] + c['executed']}"
    else:
        what = "(cut off by the budget)"
    print(f"r{i + 1:2d} think {t:6d}  other {other:5d}  total {cum:7d} | {what}")

"""Per-round reasoning tokens of the budget-ended episodes of a PEAR LIBERO run, to tell accumulation (many normal
rounds) from explosion (a few huge rounds).

    python budget_rounds.py <run dir> <model dir>  -> prints one line per episode, writes analysis/budget_rounds.json
"""
import json
import re
import sys

from transformers import AutoTokenizer

run, model = sys.argv[1], sys.argv[2]
tok = AutoTokenizer.from_pretrained(model)
n = lambda s: len(tok(s, add_special_tokens=False).input_ids)
eps = {(e["task"], e["trial"]): e for e in json.load(open(f"{run}/analysis/episodes.json"))}
out = []
for line in open(f"{run}/val_dumps/eval_0.jsonl"):
    r = json.loads(line)
    env = r["extra_fields"]["episode"]["env"]
    e = eps[(env["task_id"], env["trial_id"])]
    if e["end_reason"] != "budget":
        continue
    # the last round is cut off by the budget: its <think> may never close
    rounds = [n(t) for t in re.findall(r"<think>(.*?)(?:</think>|$)", r["output"], flags=re.S)]
    top3 = sum(sorted(rounds)[-3:]) / sum(rounds)
    big = sum(x > 20000 for x in rounds)
    out.append({"task": env["task_id"], "trial": env["trial_id"], "rounds": len(rounds), "max": max(rounds),
                "median": sorted(rounds)[len(rounds) // 2], "top3_share": top3, "over20k": big,
                "per_round": rounds})
out.sort(key=lambda o: (o["task"], o["trial"]))
for o in out:
    print(f"t{o['task']} r{o['trial']}  rounds {o['rounds']:2d}  median {o['median']:6d}  max {o['max']:6d}  "
          f"top3 {o['top3_share']:.0%}  >20k {o['over20k']}  last5 {o['per_round'][-5:]}")
json.dump(out, open(f"{run}/analysis/budget_rounds.json", "w"), indent=1)

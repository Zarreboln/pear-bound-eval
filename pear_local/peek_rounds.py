"""Print chosen rounds of chosen episodes of a PEAR LIBERO run: the action, its outcome, the state after it and the
tail of the model's reasoning for that round.

    python3 peek_rounds.py <run dir> <task>:<trial>:<rounds> ...   rounds: comma list, negative = from the end
"""
import glob
import json
import re
import sys

run = sys.argv[1]
dumps = {}
for line in open(f"{run}/val_dumps/eval_0.jsonl"):
    r = json.loads(line)
    e = r["extra_fields"]["episode"]["env"]
    dumps[(e["task_id"], e["trial_id"])] = r["output"]
calls = {}
for path in glob.glob(f"{run}/sim/actions_*.jsonl"):
    for line in open(path):
        c = json.loads(line)
        if c["action"] is not None:
            calls.setdefault((c["env"]["task_id"], c["env"]["trial_id"]), []).append(c)

for spec in sys.argv[2:]:
    task, trial, picks = spec.split(":")
    key = (int(task), int(trial))
    cs = sorted(calls[key], key=lambda c: c["steps_before"])
    thoughts = re.findall(r"<think>\n?(.*?)</think>", dumps[key], flags=re.S)
    n = len(cs)
    print(f"===== task {task} trial {trial}: {n} rounds")
    for p in map(int, picks.split(",")):
        i = p if p >= 0 else n + p
        c = cs[i]
        a = c["action"]
        arm = {k: v for k, v in a["arm"].items() if v}
        st = c["state"]
        print(f"-- round {i + 1}: action arm {arm} gripper {a['gripper']['width']} -> {c['outcome']}, "
              f"{c['executed']} steps; after: xyz ({st['arm']['x']}, {st['arm']['y']}, {st['arm']['z']}) "
              f"grip {st['gripper']['width']}")
        if i < len(thoughts):
            t = " ".join(thoughts[i].split())
            print(f"   reasoning ({len(t)} chars) ...{t[-500:]}")

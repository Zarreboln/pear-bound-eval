"""Write one replay spec per episode of a PEAR LIBERO eval run, for replay.sh (CPU rendering).

    python render_run.py <run dir>   -> <run dir>/videos/specs/<suite>_t<task>_r<trial>.{actions,rounds,args}
The native actions come from the val dump (extra_fields.episode.actions); the rounds (steps_before, executed, action)
from the tool's call log sim/actions_*.jsonl, matched by the episode's uid.
"""
import glob
import json
import os
import sys

run = sys.argv[1]
spec_dir = os.path.join(run, "videos", "specs")
os.makedirs(spec_dir, exist_ok=True)

calls = {}
for path in glob.glob(os.path.join(run, "sim", "actions_*.jsonl")):
    for line in open(path):
        c = json.loads(line)
        if c.get("action") is not None:  # the open call has no action
            calls.setdefault(c["uid"], []).append(c)

for line in open(os.path.join(run, "val_dumps", "eval_0.jsonl")):
    ep = json.loads(line)["extra_fields"]["episode"]
    env = ep["env"]
    name = f"{env['suite']}_t{env['task_id']}_r{env['trial_id']}"
    rounds = sorted(calls.get(ep["uid"], []), key=lambda c: c["steps_before"])
    json.dump(ep["actions"], open(os.path.join(spec_dir, name + ".actions"), "w"))
    json.dump([{k: c[k] for k in ("steps_before", "executed", "action")} for c in rounds],
              open(os.path.join(spec_dir, name + ".rounds"), "w"))
    tag = "ok" if ep["success"] else "fail"
    out = os.path.join(run, "videos", f"{name}_{tag}.mp4")
    args = (f"--suite {env['suite']} --task {env['task_id']} --trial {env['trial_id']} "
            f"--actions {spec_dir}/{name}.actions --rounds {spec_dir}/{name}.rounds "
            f"--success {int(ep['success'])} --out {out}")
    open(os.path.join(spec_dir, name + ".args"), "w").write(args + "\n")
    print(name, tag, len(ep["actions"]), "steps", len(rounds), "rounds")

"""Failure summary of a PEAR LIBERO run: subgoal progress (goal-predicate traces), grasps and stopped moves.

    python3 failure_summary.py <run dir>
Needs analysis/episodes.json (analyze_run.py) and analysis/goals/*.json (trace_goals.py). A grasp is a round whose
action closes the gripper (width 0) after an open one; it is empty when the fingers end under 0.5 cm apart.
"""
import collections
import glob
import json
import sys

run = sys.argv[1]
eps = {f"libero_10_t{e['task']}_r{e['trial']}": e for e in json.load(open(f"{run}/analysis/episodes.json"))}
calls = collections.defaultdict(list)
for path in glob.glob(f"{run}/sim/actions_*.jsonl"):
    for line in open(path):
        c = json.loads(line)
        if c["action"] is not None:
            calls[(c["env"]["task_id"], c["env"]["trial_id"])].append(c)

mismatch, rows = [], []
for name, e in sorted(eps.items(), key=lambda kv: (kv[1]["task"], kv[1]["trial"])):
    g = json.load(open(f"{run}/analysis/goals/{name}.json"))
    final = g["trace"][-1]["goals"]
    if e["success"] != all(final):
        mismatch.append(name)
    cs = sorted(calls[(e["task"], e["trial"])], key=lambda c: c["steps_before"])
    closes = [i for i, c in enumerate(cs)
              if c["action"]["gripper"]["width"] == 0 and (i == 0 or cs[i - 1]["action"]["gripper"]["width"] != 0)]
    row = {
        "name": name, "task": e["task"], "trial": e["trial"], "success": e["success"], "end": e["end_reason"],
        "rounds": e["rounds"], "steps": e["steps"], "goals": g["goals"], "final": final,
        "first_true": [next((t["round"] for t in g["trace"] if t["goals"][i]), None) for i in range(len(final))],
        "max_goals": max(sum(t["goals"]) for t in g["trace"]),
        "grasps": len(closes), "empty_grasps": sum(cs[i]["state"]["gripper"]["width"] < 0.5 for i in closes),
        "stopped": sum(c["outcome"] == "stopped" for c in cs), "moves": len(cs),
    }
    rows.append(row)
    if not e["success"]:
        lost = [row["first_true"][i] is not None and not final[i] for i in range(len(final))]
        print(f"t{e['task']} r{e['trial']} {e['end_reason']:6s} rounds {e['rounds']:2d} steps {e['steps']:3d} | "
              f"goals done {sum(final)}/{len(final)} (max {row['max_goals']}) first-true round {row['first_true']} "
              f"lost-later {lost} | grasps {row['grasps']} empty {row['empty_grasps']} | stopped {row['stopped']}/{len(cs)}")
        print(f"      {g['goals']}")

json.dump(rows, open(f"{run}/analysis/failures.json", "w"), indent=1)
ok = [r for r in rows if r["success"]]
fail = [r for r in rows if not r["success"]]
for label, sub in [("success", ok), ("failure", fail)]:
    print(f"{label}: n={len(sub)} grasps/ep {sum(r['grasps'] for r in sub) / len(sub):.1f}, "
          f"empty-grasp rate {sum(r['empty_grasps'] for r in sub) / max(1, sum(r['grasps'] for r in sub)):.0%}, "
          f"stopped-move rate {sum(r['stopped'] for r in sub) / sum(r['moves'] for r in sub):.0%}")
print("replay/eval success mismatches:", mismatch or "none")

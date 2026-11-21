"""Write the episode files of the context-management evaluation (LIBERO-Long, libero_10).

    python make_sets.py <baseline run dir> <out dir>
dev  (50): the baseline's 35 budget-ended episodes + its 15 successes with the most response tokens (all above 32k, so
           a 32k context cap cuts every one of them; short successes would tell nothing); used only to choose design settings (k, C).
test (100): trials 10-19 of every task, never looked at before the final runs (trials 20-29 can be added later).
"""
import json
import sys

run, out = sys.argv[1], sys.argv[2]
eps = json.load(open(f"{run}/analysis/episodes.json"))
budget = [e for e in eps if e["end_reason"] == "budget"]
long_ok = sorted([e for e in eps if e["success"]], key=lambda e: -e["response_tokens"])[:15]
dev = sorted({(e["task"], e["trial"]) for e in budget + long_ok})
spec = lambda envs: {"env_class": "pear.envs.libero.env.LiberoEnv", "data_source": "suite", "env": envs}
json.dump(spec([{"suite": ["libero_10"], "task_id": [t], "trial_id": [r]} for t, r in dev]),
          open(f"{out}/ctx_dev.json", "w"), indent=1)
json.dump(spec({"suite": ["libero_10"], "task_id": "0-9", "trial_id": "10-19"}), open(f"{out}/ctx_test.json", "w"), indent=1)
print(f"dev {len(dev)} episodes ({len(budget)} budget-ended + {len(long_ok)} long successes), test 100")

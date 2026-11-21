"""Per-round trace of a replayed LIBERO episode: which goal predicates of the task's bddl hold after each round.

    bash run_libero_py.sh trace_goals.py <spec dir> <name> <out json>
<spec dir>/<name>.{actions,rounds} come from render_run.py. Replays the native actions unrendered with PEAR's own
LiberoEnv (same scene, init state, seed and stabilization as the eval) and evaluates every goal predicate
(LIBERO's _eval_predicate, the terms of its _check_success) at the end of each round.
"""
import json
import sys

import numpy as np

from pear.envs.libero.env import LiberoEnv

spec_dir, name, out = sys.argv[1], sys.argv[2], sys.argv[3]
suite, task, trial = name.rsplit("_t", 1)[0], *map(int, name.rsplit("_t", 1)[1].split("_r"))
actions = np.asarray(json.load(open(f"{spec_dir}/{name}.actions")), dtype=np.float64).reshape(-1, LiberoEnv.ACTION_DIM)
rounds = json.load(open(f"{spec_dir}/{name}.rounds"))

env = LiberoEnv(suite, task, trial)
try:
    env._open_scene()
    domain = env.scene.env
    goals = domain.parsed_problem["goal_state"]
    holds = lambda: [bool(domain._eval_predicate(g)) for g in goals]
    trace = [{"round": 0, "step": 0, "goals": holds()}]
    step = 0
    for r, call in enumerate(rounds, start=1):
        for _ in range(call["executed"]):
            env._step(actions[step])
            step += 1
        trace.append({"round": r, "step": step, "goals": holds()})
    json.dump({"name": name, "instruction": str(env._benchmark().get_task(task).language),
               "goals": [" ".join(g) for g in goals], "trace": trace}, open(out, "w"), indent=1)
    print(name, "goals", [" ".join(g) for g in goals], "final", trace[-1]["goals"])
finally:
    env._close_scene()

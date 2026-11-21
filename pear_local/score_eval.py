"""Scores a PEAR LIBERO run for the context-management evaluation: one table for every condition.

    python score_eval.py <run dir> <episode file> [--ref <run dir>]
Needs <run>/analysis/episodes.json (analyze_run.py). Only the episodes of <episode file> count.
Metrics (per episode, then mean over the set):
  success          LIBERO success; 95% CI by bootstrap over episodes
  end reason       env success / env step limit (520) / budget (context full)
  peak context     the most tokens the model saw at once: the episode's peak_context_tokens when the run logs it (a
                   managed context), else prompt + response (the official loop never drops anything)
  KV               peak context x 64 KiB (16 full-attention layers x 4 KV heads x 256 dims x K,V x bf16); the 48
                   linear-attention layers hold a fixed state per session, the same for every condition
  generated        tokens the model wrote (reasoning + action + template): the compute a condition spends
  reasoning/round  mean over rounds; also over the container tasks alone (3, 5, 9: the budget failures)
With --ref, the paired comparison on the episodes both runs have: discordant pairs, exact McNemar p, bootstrap CI of
the success difference.
"""
import argparse
import itertools
import json
import math
import random

KV_BYTES_PER_TOKEN = 2 * 16 * 4 * 256 * 2
CONTAINER_TASKS = {3, 5, 9}
BOOT = 10000


def parse_range(text):
    out = []
    for part in text.split(","):
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b or a) + 1))
    return out


def episode_keys(path):
    spec = json.load(open(path))
    keys = set()
    for product in spec["env"] if isinstance(spec["env"], list) else [spec["env"]]:
        values = [parse_range(v) if isinstance(v, str) else list(v) for v in (product["task_id"], product["trial_id"])]
        keys.update(itertools.product(*values))
    return keys


def load(run, keys):
    rows = {(e["task"], e["trial"]): e for e in json.load(open(f"{run}/analysis/episodes.json"))}
    return {k: rows[k] for k in keys if k in rows}


def end_class(e):
    if e["success"]:
        return "success"
    return "budget" if e["end_reason"] == "budget" else "step_limit"


def peak_context(e):
    return e.get("peak_context_tokens", e["prompt_tokens"] + e["response_tokens"])


def ci(values, stat, rng):
    boots = sorted(stat([rng.choice(values) for _ in values]) for _ in range(BOOT))
    return boots[int(0.025 * BOOT)], boots[int(0.975 * BOOT)]


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def summary(rows, rng):
    es = list(rows.values())
    succ = [float(e["success"]) for e in es]
    lo, hi = ci(succ, mean, rng)
    gen = [e["reasoning"] + e["action"] + e["template"] for e in es]
    peak = [peak_context(e) for e in es]
    per_round = lambda sub: sum(e["reasoning"] for e in sub) / max(1, sum(e["rounds"] for e in sub))
    ends = {c: sum(end_class(e) == c for e in es) for c in ["success", "step_limit", "budget"]}
    print(f"episodes {len(es)} | success {mean(succ):.1%} [{lo:.1%}, {hi:.1%}] | ends {ends}")
    print(f"peak context mean {mean(peak):,.0f} max {max(peak):,} tokens | KV mean {mean(peak) * KV_BYTES_PER_TOKEN / 2**30:.2f}"
          f" max {max(peak) * KV_BYTES_PER_TOKEN / 2**30:.2f} GiB per session")
    print(f"generated mean {mean(gen):,.0f} tokens | reasoning/round {per_round(es):,.0f} "
          f"(container tasks {per_round([e for e in es if e['task'] in CONTAINER_TASKS]):,.0f}) | "
          f"rounds mean {mean([e['rounds'] for e in es]):.1f}, steps mean {mean([e['steps'] for e in es]):.0f}")


def paired(rows, ref, rng):
    keys = sorted(rows.keys() & ref.keys())
    a = [ref[k]["success"] for k in keys]
    b = [rows[k]["success"] for k in keys]
    only_ref, only_run = sum(x and not y for x, y in zip(a, b)), sum(y and not x for x, y in zip(a, b))
    n = only_ref + only_run
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(min(only_ref, only_run) + 1)) / 2**n) if n else 1.0
    diffs = [float(y) - float(x) for x, y in zip(a, b)]
    lo, hi = ci(diffs, mean, rng)
    print(f"paired on {len(keys)} episodes vs ref: {mean([float(x) for x in a]):.1%} -> {mean([float(y) for y in b]):.1%}, "
          f"diff {mean(diffs):+.1%} [{lo:+.1%}, {hi:+.1%}] | gained {only_run}, lost {only_ref}, McNemar exact p {p:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("episodes")
    ap.add_argument("--ref")
    args = ap.parse_args()
    rng = random.Random(0)
    keys = episode_keys(args.episodes)
    rows = load(args.run, keys)
    if len(rows) < len(keys):
        print(f"WARNING: {len(keys) - len(rows)} of {len(keys)} episodes missing from the run")
    summary(rows, rng)
    if args.ref:
        paired(rows, load(args.ref, keys), rng)

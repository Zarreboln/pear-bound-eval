"""Episode-level resume for PEAR evals: the official eval rewrites val_dumps/eval_<step>.jsonl from scratch on every
start, and an episode cut off midway cannot be continued, so a requeued or restarted run keeps its finished episodes
this way.

    python3 resume_episodes.py prepare <run dir> <episode file> <out episode file>
        Moves an existing val_dumps/eval_0.jsonl to val_dumps/part_<n>.jsonl, then writes <out episode file> with only
        the episodes that no part has recorded; prints how many remain (0: nothing to run).
    python3 resume_episodes.py merge <run dir>
        Writes val_dumps/eval_0.jsonl as every part followed by the latest run's records (parts kept), one record per
        episode (the first one recorded).
"""
import glob
import itertools
import json
import os
import sys


def parse_range(text):  # as pear.agent_loop.dataset.EpisodeDataset._parse_range
    out = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part.strip():
            out.append(int(part))
    return out


def episodes(spec):
    products = spec["env"] if isinstance(spec["env"], list) else [spec["env"]]
    for product in products:
        keys = list(product)
        values = [parse_range(v) if isinstance(v, str) else list(v) for v in product.values()]
        for combo in itertools.product(*values):
            yield dict(zip(keys, combo))


def key(env):
    return json.dumps(env, sort_keys=True)


def parts(run):
    return sorted(glob.glob(os.path.join(run, "val_dumps", "part_*.jsonl")),
                  key=lambda p: int(p.rsplit("_", 1)[1].split(".")[0]))


def recorded(paths):
    for path in paths:
        for line in open(path):
            if line.strip():
                yield json.loads(line)


def prepare(run, episode_file, out_file):
    dumps = os.path.join(run, "val_dumps")
    current = os.path.join(dumps, "eval_0.jsonl")
    if os.path.exists(current) and os.path.getsize(current) > 0:
        os.rename(current, os.path.join(dumps, f"part_{len(parts(run))}.jsonl"))
    done = {key(r["extra_fields"]["episode"]["env"]) for r in recorded(parts(run))}
    spec = json.load(open(episode_file))
    left = [env for env in episodes(spec) if key(env) not in done]
    # one single-valued product per remaining episode (EpisodeDataset takes a list of products)
    json.dump({**spec, "env": [{k: [v] for k, v in env.items()} for env in left]}, open(out_file, "w"), indent=1)
    print(len(left))


def merge(run):
    dumps = os.path.join(run, "val_dumps")
    latest = os.path.join(dumps, "eval_0.jsonl")
    sources = parts(run) + ([latest] if os.path.exists(latest) else [])
    seen, rows = set(), []
    for r in recorded(sources):
        k = key(r["extra_fields"]["episode"]["env"])
        if k not in seen:
            seen.add(k)
            rows.append(r)
    tmp = latest + ".tmp"
    with open(tmp, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    os.replace(tmp, latest)
    print(f"merged {len(rows)} episodes from {len(sources)} files")


if __name__ == "__main__":
    {"prepare": lambda: prepare(*sys.argv[2:5]), "merge": lambda: merge(sys.argv[2])}[sys.argv[1]]()

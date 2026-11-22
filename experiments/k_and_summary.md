# Experiment: kept reasoning rounds (*k*) and outcome summaries

**Question.** PEAR keeps the reasoning of every past round in its context. If we keep the observations and actions of
every round but remove the reasoning of all rounds except the last *k*, does PEAR still succeed, and how much memory
does it save? Does a one-line outcome summary per round help?

## Setup

| | |
|---|---|
| Method | [`context_loop.py`](../pear_local/context_loop.py) (see [Method implementation](../README.md#method-implementation)) |
| Conditions | *k* ∈ {0, 1} × outcome summary off / on |
| *k* = 0 | no past reasoning is kept |
| *k* = 1 | the previous round's reasoning is kept; older reasoning is removed |
| Outcome summary | the system prompt asks for one line starting with `Progress:` after the thinking and before the action; the line stays when the reasoning is removed |
| Not used | context cap *C*; per-round reasoning cap (not implemented yet) |
| Budget | 256k tokens of visible context (the official rule for the full context), plus a cap of 1M generated tokens per episode |
| Episodes | the development set, [`ctx_dev.json`](../pear_local/ctx_dev.json): 50 episodes of LIBERO-Long |
| Reference | full context, run 9100, on the same 50 episodes (no re-run, so the comparison slightly favors the full context) |
| Model and decoding | official PEAR LIBERO-only checkpoint and settings ([Setup](../README.md#setup)) |
| Hardware | 8×H100 per job (two TP4 replicas), one job per machine |
| Dates | 2026-11-21 to 2026-11-22 |

Paired comparison: bootstrap 95% CI of the success difference and the exact McNemar test. "Gained" and "lost" count
the episodes that the condition solves and the full context does not, and the reverse.

## Success

| condition | success rate | Δ vs. full context | gained | lost | McNemar *p* | ended: 520-step limit | ended: token budget | ended: parse failure |
|---|---|---|---|---|---|---|---|---|
| Full context | 30.0% [18, 44] | – | – | – | – | 0 | 35 | 0 |
| *k* = 0, no summary | 4.0% [0, 10] | −26.0 [−38, −14] | 0 | 13 | < 0.001 | 30 | 0 | 18 |
| *k* = 0, summary | 2.0% [0, 6] | −28.0 [−40, −16] | 0 | 14 | < 0.001 | 15 | 17 | 17 |
| *k* = 1, no summary | 20.0% [10, 32] | −10.0 [−24, +4] | 5 | 10 | 0.30 | 36 | 3 | 1 |
| ***k* = 1, summary** | **28.0% [16, 40]** | **−2.0 [−18, +14]** | 7 | 8 | 1.00 | 17 | 16 | 3 |

"Token budget" includes episodes that reached the 1M generated-token cap (*k* = 0 with summary: 17; *k* = 1 without
summary: 1; *k* = 1 with summary: 10).

## Efficiency

| condition | peak KV cache, mean / max (GiB) | generated tokens per episode | generated tokens per success | reasoning tokens per round | prefix-cache hit rate |
|---|---|---|---|---|---|
| Full context | 12.30 / 15.68 | 183k | 0.61M | 6,159 | 91% |
| *k* = 0, no summary | 3.71 / 9.41 | 313k | 7.83M | 8,674 | 94% |
| *k* = 0, summary | 5.83 / 13.53 | 604k | 30.2M | 17,863 | 90% |
| *k* = 1, no summary | 6.30 / 15.69 | 387k | 1.93M | 8,291 | 54% |
| *k* = 1, summary | 7.93 / 15.69 | 473k | 1.69M | 13,012 | 46% |

- **Peak KV cache** is computed from the peak context length (64 KiB per token).
- **Prefix-cache hit rate** is vLLM's own metric, averaged over its 10-second samples (not token-weighted).

## Findings

1. **Removing all past reasoning (*k* = 0) breaks the agent.** Success drops from 30% to 2–4%. About a third of the
   episodes end with a parse failure: the model stops without writing an action block. With only empty thinking
   blocks in its history, the model drifts from its output format.
2. **Keeping the previous round's reasoning with outcome summaries (*k* = 1, summary) preserves success**
   (28% vs 30%, not significant) and reduces the mean peak KV cache by 36% (12.30 → 7.93 GiB).
3. **The model generates more when its past reasoning is removed.** Reasoning per round rises by 35% to 2×, and with
   summaries the generated tokens per success are 2.8× the full context. Episodes no longer stop at the 256k budget,
   so they run longer, and some reach the 1M cap. The method currently trades memory for compute; the per-round reasoning cap is meant to
   address this.
4. **Evicting reasoning lowers prefix-cache reuse.** With *k* = 1, each eviction edits the middle of the context, so
   everything after it is recomputed, and the hit rate falls from 91% to about 50%.

## Data and logs

| condition | job | results | per-round log |
|---|---|---|---|
| *k* = 0, no summary | 9452 | [`results/dev_k0_nosummary_9452/`](../results/dev_k0_nosummary_9452/) | [`turns.jsonl.gz`](../results/dev_k0_nosummary_9452/turns.jsonl.gz) |
| *k* = 0, summary | 9453 | [`results/dev_k0_summary_9453/`](../results/dev_k0_summary_9453/) | [`turns.jsonl.gz`](../results/dev_k0_summary_9453/turns.jsonl.gz) |
| *k* = 1, no summary | 9454 | [`results/dev_k1_nosummary_9454/`](../results/dev_k1_nosummary_9454/) | [`turns.jsonl.gz`](../results/dev_k1_nosummary_9454/turns.jsonl.gz) |
| *k* = 1, summary | 9455 | [`results/dev_k1_summary_9455/`](../results/dev_k1_summary_9455/) | [`turns.jsonl.gz`](../results/dev_k1_summary_9455/turns.jsonl.gz) |
| Full context (reference) | 9100 | [`results/baseline_9100/`](../results/baseline_9100/) | in [`eval_0.jsonl.gz`](../results/baseline_9100/eval_0.jsonl.gz) |

Each results folder has:

- `analysis/episodes.json`: per-episode outcome, end reason, peak context and token counts;
- `eval_0.jsonl.gz`: the eval dump. For the method runs it holds the final managed context, not the full history;
- `turns.jsonl.gz`: every round's full generated text (reasoning, summary and action), one JSON line per round.

To reproduce the tables:

```bash
python3 pear_local/score_eval.py results/dev_k1_summary_9455 pear_local/ctx_dev.json --ref results/baseline_9100
```

To run a condition (see [Running an evaluation](../README.md#running-an-evaluation)):

```bash
sbatch --gres=gpu:8 \
    --export=ALL,MODEL_FILE=$HOME/code/pear_local/qwen3_8_27b_pear_libonly.sh,EPISODES=$HOME/code/pear_local/ctx_dev.json,TAG=ctxdev_k1_sum1,CTX_K=1,CTX_SUMMARY=1 \
    pear_local/eval_long.sbatch
```

## Compute

| condition | GPU-hours |
|---|---|
| *k* = 0, no summary | 24.0 |
| *k* = 0, summary | 31.3 |
| *k* = 1, no summary | 46.7 |
| *k* = 1, summary | 47.4 |
| **Total** | **149.4** |

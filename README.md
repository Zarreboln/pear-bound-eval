# PEAR-bound: evaluation

Evaluation code and baseline results for **PEAR-bound**, our 6.5940 (TinyML) final project on bounding the context of
PEAR (Qwen3.8-27B) on LIBERO-Long without retraining.

This repository covers the evaluation side:

- the full-context baseline;
- the failure analysis;
- the development and test episode sets;
- the scoring script used to compare conditions.

The context-management method itself (reasoning eviction, outcome summaries, context cap) is implemented separately.

## Status (2026-11-21)

| item | status |
|---|---|
| Full-context baseline | Run 9100 finished: 100 episodes, 50% success, 35% ended by the token budget. The paper reports 53.2% / 33%. |
| Failure analysis | Budget failures concentrate on the tasks that place an object into a closed container (tasks 3, 5, 9). The cause is very long reasoning in single rounds. |
| Development set | 50 episodes: the 35 budget-ended episodes and the 15 longest successful ones (`pear_local/ctx_dev.json`). |
| Test set | Initial states 10–19 of every task, 100 episodes (`pear_local/ctx_test.json`). |
| Scoring | `pear_local/score_eval.py`: success with a bootstrap CI, end reasons, peak KV cache, and a paired comparison (exact McNemar test and bootstrap CI of the difference). |

## Setup

We follow the official PEAR evaluation setting. The released LIBERO-only checkpoint is
`mit-han-lab/Qwen3.8-27B-PEAR-LIBERO_only` at revision `63e96d9`. Settings:

- temperature 1.0, top-*p* 0.95, `xhigh` reasoning effort;
- at most 520 simulator steps and 256k tokens per episode;
- two camera views (agent view and wrist) per observation, 256 image tokens each.

Serving uses vLLM 0.28.0 with tensor parallelism 4 on 4×H100 (one replica), plus an LMCache host tier.

The scripts in `pear_local/` assume:

- the PEAR repository at `~/code/pear`;
- this folder at `~/code/pear_local`;
- the environments and runs under `$WORK_DIR` (set in `pear_local/env.sh`).

The model files `qwen3_8_27b_pear_libonly.sh` and `qwen3_8_27b_pear_ckpt.sh` are copies of PEAR's
`examples/models/qwen3_8_27b_pear/qwen3_8_27b.sh` with only the checkpoint changed.

## Baseline: full context (run 9100)

LIBERO-Long (`libero_10`), tasks 0–9 × initial states 0–9, 100 episodes. The job ran for 3 h 50 min on 4×H100
(≈15.3 GPU-hours).

| | ours (100 episodes) | PEAR paper (500 episodes) |
|---|---|---|
| Success rate | 50.0% [40.0, 60.0] | 53.2% |
| Ended because the token budget ran out | 35% | 33% |
| Ended at the 520-step limit | 15% | – |
| Peak context per episode (mean / max) | 136,689 / 256,894 tokens | – |
| Peak KV cache per episode (mean / max) | 8.34 / 15.68 GiB | – |

KV cache size per token is 64 KiB: 16 full-attention layers × 4 KV heads × 256 dims × (K, V) × bf16. The other 48
layers are linear attention (Gated DeltaNet), which keeps a fixed-size state per session.

### Where the context goes

| reasoning | images | template | observation text | actions |
|---|---|---|---|---|
| 83.0% | 10.5% | 3.5% | 1.9% | 1.2% |

Budget-ended episodes do not take more rounds than episodes that end in the simulator (28.3 vs 28.6 on average).
Instead, they write more per round: 9.1k vs 2.5k tokens per round. In 34 of the 35 budget-ended episodes, at least
one round exceeds 20k reasoning tokens; for example, round 12 of task 9 / initial state 4 alone has 66,754.

Budget-ended episodes by task: task 9: 10/10, task 5: 9/10, task 3: 8/10, tasks 2, 4, 8: 2 each, tasks 1, 6: 1 each.

### Failure analysis

We replay every episode in the simulator (CPU) and evaluate LIBERO's goal predicates after each round. The replay
agrees with the evaluation on all 100 episodes. The 50 failures fall into three groups:

| group | episodes | tasks | what happens |
|---|---|---|---|
| A. cannot place a held object into a closed container | 29 | 9 (microwave), 5 (caddy back compartment), 3 (drawer) | The grasp is stable, but the move into the container is blocked repeatedly. The model then re-derives the scene geometry from pixels at length in every round until the token budget runs out. |
| B. unstable grasp or placement | 19 | 8, 2, 4, 1 | Repeated grasp attempts. These episodes reach the 520-step limit. |
| C. knocks off a completed subgoal | 2 | 6 | A later placement undoes an earlier one. |

Compared with successful episodes, failed ones have a much higher empty-grasp rate (23% vs 4%) and more blocked moves
(49% vs 26% of moves are stopped).

## Episode sets

| set | file | episodes | used for |
|---|---|---|---|
| Baseline | `libero_long_first10.json` | tasks 0–9 × initial states 0–9 (100) | the full-context run above |
| Development | `ctx_dev.json` | 50 | choosing design settings ($k$, $C$) |
| Test | `ctx_test.json` | tasks 0–9 × initial states 10–19 (100) | final results only |

The development set contains:

- the 35 episodes of run 9100 that ran out of budget;
- the 15 successful episodes with the most response tokens.

`make_sets.py` writes both files from run 9100.

The full-context baseline scores 30.0% [18.0, 44.0] on the development set, with a mean peak KV cache of 12.30 GiB.

The set is selected from one stochastic run, so a re-run of the same setting will score higher on it (regression to
the mean). Conditions should therefore be compared against a **fresh full-context run on the development set**, not
against run 9100.

## Scoring a run

```bash
python3 pear_local/analyze_run.py <run dir> <model dir>     # writes <run>/analysis/episodes.json
python3 pear_local/score_eval.py <run dir> pear_local/ctx_dev.json [--ref <reference run dir>]
```

`score_eval.py` reports:

- success rate with a 95% bootstrap CI;
- end reasons (success / step limit / budget);
- peak context and peak KV cache per episode;
- generated tokens and reasoning tokens per round, with the container tasks reported separately.

With `--ref`, it adds the paired comparison on the shared episodes: episodes gained and lost, the exact McNemar *p*,
and a bootstrap CI of the success difference.

A run with managed context should log `peak_context_tokens` per episode. Without it, the script uses
prompt + response, which is only correct for the official append-only loop.

Example on the baseline:

```bash
python3 pear_local/score_eval.py results/baseline_9100 pear_local/libero_long_first10.json
```

## Running an evaluation

```bash
sbatch --export=ALL,MODEL_FILE=$HOME/code/pear_local/qwen3_8_27b_pear_libonly.sh,EPISODES=$HOME/code/pear_local/ctx_dev.json,TAG=<name> \
    pear_local/eval_long.sbatch
```

`eval_long.sbatch` runs the official `examples/eval/eval_libero.sh` on one TP4 replica. It is submitted with `--requeue`, so a preempted job restarts;
add the partition and QoS of your cluster.

The official eval rewrites its output file on every start, so `resume_episodes.py` adds episode-level resume:

- before each start, it keeps the finished episodes and passes only the remaining ones to the eval;
- at the end, it merges all parts.

## Files

| file | purpose |
|---|---|
| `pear_local/eval_long.sbatch` | evaluation job (official eval + episode-level resume) |
| `pear_local/resume_episodes.py` | resume: `prepare` (skip finished episodes) and `merge` |
| `pear_local/analyze_run.py` | per-episode outcome, end reason, rounds and token breakdown → `analysis/episodes.json` |
| `pear_local/budget_rounds.py` | reasoning tokens per round of the budget-ended episodes → `analysis/budget_rounds.json` |
| `pear_local/token_timeline.py` | round-by-round token timeline of one episode against the 256k budget |
| `pear_local/make_sets.py` | writes the development and test sets |
| `pear_local/score_eval.py` | scores a run, optionally paired against a reference run |
| `pear_local/render_run.py`, `replay.sh`, `replay_video.py`, `render_videos.sh` | replay episodes into videos (CPU rendering) |
| `pear_local/trace_goals.py`, `trace_all.sh`, `run_libero_py.sh` | goal-predicate trace after every round (CPU) |
| `pear_local/failure_summary.py` | subgoal progress, grasps and stopped moves → `analysis/failures.json` |
| `pear_local/peek_rounds.py` | prints the action, outcome and reasoning tail of chosen rounds |
| `pear_local/build_cpu.sh` | builds the LIBERO and verl environments on a CPU node |
| `results/baseline_9100/` | run 9100: summary, per-episode analysis, goal traces, and the full eval dump (`eval_0.jsonl.gz`, with the model's reasoning) |

Earlier, smaller runs are also included: `eval_libero_long30.sbatch` (30 episodes with the general PEAR checkpoint)
and `eval_libero40.sbatch`.

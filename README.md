# PEAR-bound: evaluation

Evaluation code and results for **PEAR-bound**, our 6.5940 (TinyML) final project on bounding the context of PEAR
(Qwen3.8-27B) on LIBERO-Long without retraining.

This repository contains:

- the full-context baseline and its failure analysis;
- the development and test episode sets;
- the implementation of the context-management method (`pear_local/context_loop.py`);
- the results of the first method runs;
- the scoring script used to compare conditions.

## Status (2026-11-22)

| item | status |
|---|---|
| Full-context baseline | Run 9100 (initial states 0–9): 50% success, 35% ended by the token budget. Test set (initial states 10–19): 54% success, 36% ended by the budget. The paper reports 53.2% / 33%. |
| Failure analysis | Budget failures concentrate on the tasks that place an object into a closed container (tasks 3, 5, 9). The cause is very long reasoning in single rounds. |
| Method | Implemented in `pear_local/context_loop.py`: reasoning eviction with the last *k* rounds kept, per-round outcome summaries, and a context cap *C*. The per-round reasoning cap is not implemented yet. |
| Development-set runs | *k* ∈ {0, 1} × with / without summaries: done. Results [below](#results-on-the-development-set). |
| Test-set runs | Full context: done. Method: not run yet. |

## Setup

We follow the official PEAR evaluation setting. The released LIBERO-only checkpoint is
`mit-han-lab/Qwen3.8-27B-PEAR-LIBERO_only` at revision `63e96d9`. Settings:

- temperature 1.0, top-*p* 0.95, `xhigh` reasoning effort;
- at most 520 simulator steps and 256k tokens per episode;
- two camera views (agent view and wrist) per observation, 256 image tokens each.

Serving uses vLLM 0.28.0 with tensor parallelism 4 (TP4) on 4×H100 per replica. Run 9100 used one replica; all later
runs use 8×H100 per job, i.e., two TP4 replicas. The replica configuration is the same, so results are comparable; only
throughput differs.

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

Data: [`results/baseline_9100/`](results/baseline_9100/) (per-episode analysis, goal traces, full eval dump with the
model's reasoning).

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
| Baseline | [`libero_long_first10.json`](pear_local/libero_long_first10.json) | tasks 0–9 × initial states 0–9 (100) | the full-context run above |
| Development | [`ctx_dev.json`](pear_local/ctx_dev.json) | 50 | choosing design settings (*k*, *C*) |
| Test | [`ctx_test.json`](pear_local/ctx_test.json) | tasks 0–9 × initial states 10–19 (100) | final results only |

The development set contains:

- the 35 episodes of run 9100 that ran out of budget;
- the 15 successful episodes with the most response tokens.

[`make_sets.py`](pear_local/make_sets.py) writes both files from run 9100.

The full-context baseline scores 30.0% [18.0, 44.0] on the development set, with a mean peak KV cache of 12.30 GiB.

The development set is selected from one stochastic run, so a re-run of the same setting would likely score higher on
it (regression to the mean). The comparisons below use run 9100 as the reference, so they slightly favor the full
context. Final conclusions come from the test set.

## Method implementation

[`context_loop.py`](pear_local/context_loop.py) subclasses PEAR's agent loop. Before every generation, it rebuilds the
prompt from the message list:

- **Fixed prefix:** the system prompt, the task and the first observation are never evicted.
- **Observations and actions:** kept for every round, with their images.
- **Reasoning:** kept for the last *k* rounds and removed from older ones. A round without its reasoning renders as an
  empty `<think></think>` block, as Qwen's chat template renders past turns.
- **Outcome summary** (optional): the system prompt asks for one line starting with `Progress:` after the thinking and
  before the action. The line is kept when the round's reasoning is removed.
- **Context cap *C*** (optional): if the rebuilt context exceeds *C*, whole rounds are evicted, oldest first. Not used
  in the runs below.

**Budget rule.** The 256k-token budget applies to the context the model sees. For the full context this is identical
to the official rule (it never removes anything). Because evicting reasoning frees budget, each episode also has a
cap of 1M generated tokens.

**Correctness check (CPU).** With nothing evicted, the rebuilt prompt matches the official loop's tokens exactly
(8 rounds of a real episode, 10,181 tokens): [`ctx_probe.py`](pear_local/ctx_probe.py),
[`ctx_loop_test.py`](pear_local/ctx_loop_test.py). An end-to-end run with nothing evicted is still pending.

Each episode record gets a `ctx` field: the settings, peak context, generated and reasoning tokens per round, and the
context length at every rebuild. [`analyze_run.py`](pear_local/analyze_run.py) reads it.

## Results on the development set

50 episodes; reference: full context (run 9100) on the same episodes. Paired comparison: bootstrap 95% CI of the
success difference and the exact McNemar test.

### Success

| condition | success rate | Δ vs. full context | gained / lost | McNemar *p* | end reasons (success / step limit / budget / parse failure) |
|---|---|---|---|---|---|
| Full context | 30.0% [18, 44] | – | – | – | 15 / 0 / 35 / 0 |
| *k* = 0, no summary | 4.0% [0, 10] | −26.0 [−38, −14] | 0 / 13 | < 0.001 | 2 / 30 / 0 / 18 |
| *k* = 0, summary | 2.0% [0, 6] | −28.0 [−40, −16] | 0 / 14 | < 0.001 | 1 / 15 / 17 / 17 |
| *k* = 1, no summary | 20.0% [10, 32] | −10.0 [−24, +4] | 5 / 10 | 0.30 | 10 / 36 / 3 / 1 |
| ***k* = 1, summary** | **28.0% [16, 40]** | **−2.0 [−18, +14]** | 7 / 8 | 1.00 | 14 / 17 / 16 / 3 |

"Budget" includes episodes that reached the 1M generated-token cap (*k* = 0 with summary: 17; *k* = 1 without
summary: 1; *k* = 1 with summary: 10).

### Efficiency

| condition | peak KV cache, mean / max (GiB) | generated tokens per episode | generated tokens per success | reasoning tokens per round | prefix-cache hit rate | preemptions |
|---|---|---|---|---|---|---|
| Full context | 12.30 / 15.68 | 183k | 0.61M | 6,159 | 91% | yes* |
| *k* = 0, no summary | 3.71 / 9.41 | 313k | 7.83M | 8,674 | 94% | 0 |
| *k* = 0, summary | 5.83 / 13.53 | 604k | 30.2M | 17,863 | 90% | 0 |
| *k* = 1, no summary | 6.30 / 15.69 | 387k | 1.93M | 8,291 | 54% | 0 |
| *k* = 1, summary | 7.93 / 15.69 | 473k | 1.69M | 13,012 | 46% | 0 |

- **Peak KV cache** is computed from the peak context length (64 KiB per token).
- **Prefix-cache hit rate** is vLLM's own metric, averaged over its 10-second samples (not token-weighted).
- **\*Preemptions:** the full-context test-set run (same 8×H100, two-replica setup) had 4.6 preemptions per minute on
  average and up to 65; none of the method runs had any.

### Findings

1. **Removing all past reasoning (*k* = 0) breaks the agent.** Success drops from 30% to 2–4%. About a third of the
   episodes end with a parse failure: the model stops without writing an action block. With only empty thinking
   blocks in its history, the model drifts from its output format.
2. **Keeping the previous round's reasoning with outcome summaries (*k* = 1, summary) preserves success**
   (28% vs 30%, not significant) and reduces the mean peak KV cache by 36% (12.30 → 7.93 GiB). The method runs had
   no preemptions.
3. **The model generates more when its past reasoning is removed.** Reasoning per round rises by 35% to 2×, and with
   summaries the generated tokens per success are 2.8× the full context. Episodes no longer stop at the 256k budget, so they run longer, and
   some reach the 1M cap. The method currently trades memory for compute; the per-round reasoning cap is meant to
   address this.
4. **Evicting reasoning lowers prefix-cache reuse.** With *k* = 1, each eviction edits the middle of the context, so
   everything after it is recomputed, and the hit rate falls from 91% to about 50%.

### Data and logs

| condition | results | per-round log |
|---|---|---|
| *k* = 0, no summary (job 9452) | [`results/dev_k0_nosummary_9452/`](results/dev_k0_nosummary_9452/) | [`turns.jsonl.gz`](results/dev_k0_nosummary_9452/turns.jsonl.gz) |
| *k* = 0, summary (job 9453) | [`results/dev_k0_summary_9453/`](results/dev_k0_summary_9453/) | [`turns.jsonl.gz`](results/dev_k0_summary_9453/turns.jsonl.gz) |
| *k* = 1, no summary (job 9454) | [`results/dev_k1_nosummary_9454/`](results/dev_k1_nosummary_9454/) | [`turns.jsonl.gz`](results/dev_k1_nosummary_9454/turns.jsonl.gz) |
| *k* = 1, summary (job 9455) | [`results/dev_k1_summary_9455/`](results/dev_k1_summary_9455/) | [`turns.jsonl.gz`](results/dev_k1_summary_9455/turns.jsonl.gz) |

Each results folder has:

- `analysis/episodes.json`: per-episode outcome, end reason, peak context and token counts;
- `eval_0.jsonl.gz`: the eval dump. For the method runs it holds the final managed context, not the full history;
- `turns.jsonl.gz`: every round's full generated text (reasoning, summary and action), one JSON line per round.

Compute: 168.2 GPU-hours for the five runs on this page after run 9100 (*k* = 0: 24.0 and 31.3; *k* = 1: 46.7
and 47.4, each including one preemption; test-set full context: 18.8).

## Results on the test set

| condition | success rate | ended by budget | peak KV cache, mean / max (GiB) | generated tokens per episode | data |
|---|---|---|---|---|---|
| Full context (job 9456) | 54.0% [44, 64] | 36% | 8.19 / 15.68 | 116k | [`results/test_full_9456/`](results/test_full_9456/) |

## Next steps

1. **End-to-end correctness check:** run `context_loop.py` with nothing evicted (`CTX_K=-1`) on the development set;
   it should reproduce the full context's ~30%.
2. **Per-round reasoning cap** with a forced action when a round is cut off (part of the method, not implemented yet).
3. **Context cap *C*** (32k, 64k) and the **recent-window baseline** at the same *C* (RQ1, RQ2).
4. **Test set** for the best configuration and the recent window.
5. **Efficiency:** offline measurement of how many recomputed tokens suffix cache reuse could save, and a throughput
   test at maximum concurrency.

## Scoring a run

```bash
python3 pear_local/analyze_run.py <run dir> <model dir>     # writes <run>/analysis/episodes.json
python3 pear_local/score_eval.py <run dir> pear_local/ctx_dev.json [--ref <reference run dir>]
```

[`score_eval.py`](pear_local/score_eval.py) reports:

- success rate with a 95% bootstrap CI;
- end reasons (success / step limit / budget / parse failure);
- peak context and peak KV cache per episode;
- generated tokens and reasoning tokens per round, with the container tasks reported separately.

With `--ref`, it adds the paired comparison on the shared episodes: episodes gained and lost, the exact McNemar *p*,
and a bootstrap CI of the success difference.

Examples with the data in this repository:

```bash
python3 pear_local/score_eval.py results/test_full_9456 pear_local/ctx_test.json
python3 pear_local/score_eval.py results/dev_k1_summary_9455 pear_local/ctx_dev.json --ref results/baseline_9100
```

## Running an evaluation

Full context (official loop):

```bash
sbatch --gres=gpu:8 \
    --export=ALL,MODEL_FILE=$HOME/code/pear_local/qwen3_8_27b_pear_libonly.sh,EPISODES=$HOME/code/pear_local/ctx_dev.json,TAG=<name> \
    pear_local/eval_long.sbatch
```

Managed context: link the loop into PEAR once, then add the settings:

```bash
ln -s ~/code/pear_local/context_loop.py ~/code/pear/pear/agent_loop/context_loop.py
sbatch --gres=gpu:8 \
    --export=ALL,MODEL_FILE=...,EPISODES=...,TAG=<name>,CTX_K=1,CTX_SUMMARY=1[,CTX_CAP=32768] \
    pear_local/eval_long.sbatch
```

| variable | meaning |
|---|---|
| `CTX_K` | rounds whose reasoning is kept; `-1` keeps all |
| `CTX_SUMMARY` | `1`: ask for a `Progress:` line each round |
| `CTX_CAP` | context cap in tokens (`0`: none) |
| `CTX_GEN_CAP` | generated tokens per episode (default 1M) |

Notes:

- [`eval_long.sbatch`](pear_local/eval_long.sbatch) runs the official `examples/eval/eval_libero.sh`, one TP4 replica
  per 4 GPUs. Add the partition and QoS of your cluster.
- **Run one job per machine.** Each job starts its own Ray cluster, and two jobs on one machine interfere (a job's
  `ray stop` kills the other's Ray).
- The job is submitted with `--requeue`, so a preempted job restarts. The official eval rewrites its output file on
  every start, so [`resume_episodes.py`](pear_local/resume_episodes.py) keeps the finished episodes and runs only the
  remaining ones.

## Files

| file | purpose |
|---|---|
| [`pear_local/context_loop.py`](pear_local/context_loop.py) | the method: managed-context agent loop |
| [`pear_local/ctx_agent_loop.yaml`](pear_local/ctx_agent_loop.yaml) | agent-loop config that registers it |
| [`pear_local/ctx_probe.py`](pear_local/ctx_probe.py), [`ctx_loop_test.py`](pear_local/ctx_loop_test.py) | CPU checks of the prompt rebuild against the official tokens |
| [`pear_local/eval_long.sbatch`](pear_local/eval_long.sbatch) | evaluation job (official eval, episode-level resume, optional managed context) |
| [`pear_local/resume_episodes.py`](pear_local/resume_episodes.py) | resume: `prepare` (skip finished episodes) and `merge` |
| [`pear_local/analyze_run.py`](pear_local/analyze_run.py) | per-episode outcome, end reason, rounds and token counts → `analysis/episodes.json` |
| [`pear_local/score_eval.py`](pear_local/score_eval.py) | scores a run, optionally paired against a reference run |
| [`pear_local/budget_rounds.py`](pear_local/budget_rounds.py) | reasoning tokens per round of the budget-ended episodes |
| [`pear_local/token_timeline.py`](pear_local/token_timeline.py) | round-by-round token timeline of one episode against the 256k budget |
| [`pear_local/make_sets.py`](pear_local/make_sets.py) | writes the development and test sets |
| `pear_local/render_run.py`, `replay.sh`, `replay_video.py`, `render_videos.sh` | replay episodes into videos (CPU rendering) |
| `pear_local/trace_goals.py`, `trace_all.sh`, `run_libero_py.sh` | goal-predicate trace after every round (CPU) |
| [`pear_local/failure_summary.py`](pear_local/failure_summary.py) | subgoal progress, grasps and stopped moves |
| [`pear_local/peek_rounds.py`](pear_local/peek_rounds.py) | prints the action, outcome and reasoning tail of chosen rounds |
| `pear_local/build_cpu.sh` | builds the LIBERO and verl environments on a CPU node |
| [`results/`](results/) | all runs on this page |

Earlier, smaller runs are also included: `eval_libero_long30.sbatch` (30 episodes with the general PEAR checkpoint)
and `eval_libero40.sbatch`.

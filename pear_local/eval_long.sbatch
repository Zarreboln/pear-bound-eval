#!/bin/bash
# A PEAR checkpoint on a LIBERO episode file, official full-context setting, on 4 H100s = one TP4 replica: official
# GPU install check (install.sh libero_run), then a single-node Ray cluster (this node = head, rollout role), then the
# official examples/eval/eval_libero.sh with EPISODES pointed at the file.
#   sbatch --export=ALL,MODEL_FILE=<pear_local model file>,EPISODES=<episode file>,TAG=<run name>[,CTX_K=..,CTX_SUMMARY=..,CTX_CAP=..] eval_long.sbatch
# MODEL is the downloaded directory of MODEL_FILE's repository ($WORK_DIR/models/<repo name>).
# --requeue: a preempted job restarts with the same job id; add --partition/--qos for your cluster.
#SBATCH --job-name=pear_eval
#SBATCH --gres=gpu:4
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.out
set -uo pipefail
MODEL_FILE_ARG=${MODEL_FILE:?pear_local model file}; EPISODES=${EPISODES:?episode file}; TAG=${TAG:?run name}
source ~/code/pear_local/env.sh; cd ~/code/pear
export MODEL_FILE=$MODEL_FILE_ARG
REPO=$(sed -n 's/^MODEL_REPO=//p' "$MODEL_FILE"); export MODEL=$WORK_DIR/models/${REPO##*/}
# one TP4 replica per 4 GPUs the job got (slurm submit -g 4 | -g 8 overrides --gres): the model file's H100 setting
# reads these (pear_local copy); 400 GB of LMCache host tier per replica fits a job's ~194 GB per GPU; Ray offers the
# job's CPUs
export GPUS_PER_NODE=$(nvidia-smi -L | wc -l) LMCACHE_L1_GB=400 NUM_CPUS=${SLURM_CPUS_ON_NODE:-80}
T0=$(date +%s); echo "host=$(hostname) start=$(date -Is) model=$MODEL episodes=$EPISODES tag=$TAG"
nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader

s=$(date +%s); bash scripts/install.sh libero_run > $WORK_DIR/logs/libero_run_check_$SLURM_JOB_ID.log 2>&1; rc=$?
echo "[$(date -Is)] libero_run check rc=$rc sec=$(( $(date +%s)-s ))"; tail -3 $WORK_DIR/logs/libero_run_check_$SLURM_JOB_ID.log
[ $rc -eq 0 ] || exit $rc

# Ray with token auth (shared cluster network)
export RAY_AUTH_MODE=token RAY_AUTH_TOKEN_PATH=$HOME/.pear_ray_token
[ -f $RAY_AUTH_TOKEN_PATH ] || (umask 077; head -c 32 /dev/urandom | base64 > $RAY_AUTH_TOKEN_PATH)
bash scripts/start_ray_node.sh rollout > $WORK_DIR/logs/ray_head_$SLURM_JOB_ID.log 2>&1 &
RAY_PID=$!
for i in $(seq 1 60); do $WORK_DIR/envs/verl/bin/ray status > /dev/null 2>&1 && break; sleep 10; done
$WORK_DIR/envs/verl/bin/ray status 2>&1 | head -20

s=$(date +%s)
RUN=$WORK_DIR/runs/${TAG}_$SLURM_JOB_ID
# episode-level resume (a requeue keeps the job id, so the same RUN): finished episodes are kept, only the rest run
mkdir -p $RUN/val_dumps
LEFT=$(python3 ~/code/pear_local/resume_episodes.py prepare $RUN $EPISODES $RUN/episodes_left.json)
echo "[$(date -Is)] restart_count=${SLURM_RESTART_COUNT:-0} episodes left: $LEFT"
if [ "$LEFT" -gt 0 ]; then
  # managed context (context_loop.py): CTX_K set selects the ctx_agent loop; its settings reach the workers as Ray env vars
  # (quoted: Ray takes string values only)
  CTX_ARGS=()
  if [ -n "${CTX_K:-}" ]; then
    E=+ray_kwargs.ray_init.runtime_env.env_vars
    CTX_ARGS=(actor_rollout_ref.rollout.agent.agent_loop_config_path=$HOME/code/pear_local/ctx_agent_loop.yaml
              actor_rollout_ref.rollout.agent.default_agent_loop=ctx_agent
              "$E.CTX_K='$CTX_K'" "$E.CTX_SUMMARY='${CTX_SUMMARY:-0}'" "$E.CTX_CAP='${CTX_CAP:-0}'" "$E.CTX_GEN_CAP='${CTX_GEN_CAP:-1000000}'")
    echo "ctx: k=$CTX_K summary=${CTX_SUMMARY:-0} cap=${CTX_CAP:-0} gen_cap=${CTX_GEN_CAP:-1000000}"
  fi
  ROLLOUT_NNODES=1 EPISODES=$RUN/episodes_left.json EVAL_STEP=0 RUN=$RUN bash examples/eval/eval_libero.sh "${CTX_ARGS[@]}"
  echo "[$(date -Is)] eval rc=$? sec=$(( $(date +%s)-s ))"
fi
python3 ~/code/pear_local/resume_episodes.py merge $RUN
cat $RUN/val_dumps/eval_0_summary.json 2>/dev/null
$WORK_DIR/envs/verl/bin/ray stop --force > /dev/null 2>&1; kill $RAY_PID 2> /dev/null
echo "end=$(date -Is) total_sec=$(( $(date +%s)-T0 ))"

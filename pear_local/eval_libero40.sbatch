#!/bin/bash
# Released Qwen3.8-27B-PEAR on a 40-episode LIBERO subset (4 suites x 10 tasks x trial 0) on ONE 8xH100 node:
# official GPU install check (install.sh libero_run), then a single-node Ray cluster (this node = head, rollout role),
# then the official examples/eval/eval_libero.sh with EPISODES pointed at the subset.
# --requeue: a preempted job restarts with the same job id; add --partition/--qos for your cluster.
#SBATCH --job-name=pear_eval40
#SBATCH --gres=gpu:8
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.out
set -uo pipefail
source ~/code/pear_local/env.sh; cd ~/code/pear
T0=$(date +%s); echo "host=$(hostname) start=$(date -Is) model=$MODEL"
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
ROLLOUT_NNODES=1 EPISODES=$HOME/code/pear_local/libero_40.json EVAL_STEP=0 RUN=$WORK_DIR/runs/eval40_$SLURM_JOB_ID \
  bash examples/eval/eval_libero.sh
echo "[$(date -Is)] eval rc=$? sec=$(( $(date +%s)-s ))"
cat $WORK_DIR/runs/eval40_$SLURM_JOB_ID/val_dumps/eval_0_summary.json 2>/dev/null
$WORK_DIR/envs/verl/bin/ray stop --force > /dev/null 2>&1; kill $RAY_PID 2> /dev/null
echo "end=$(date -Is) total_sec=$(( $(date +%s)-T0 ))"

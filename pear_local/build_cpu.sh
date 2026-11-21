#!/bin/bash
# Build the PEAR LIBERO + verl environments on a CPU node (free cpu QOS), for H100 (CUDA_ARCH=9.0). The GPU check
# (install.sh libero_run) runs later on a GPU node and finds these built.
set -uo pipefail
source ~/code/pear_local/env.sh; cd ~/code/pear
export MAX_JOBS=${MAX_JOBS:-24}
for t in libero numactl verl; do
  s=$(date +%s); echo "[$(date -Is)] install $t"
  bash scripts/install.sh $t > $WORK_DIR/logs/install_$t.log 2>&1; rc=$?
  echo "[$(date -Is)] $t rc=$rc sec=$(( $(date +%s)-s ))"; [ $rc -eq 0 ] || exit $rc
done

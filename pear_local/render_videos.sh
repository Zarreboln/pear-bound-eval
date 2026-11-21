#!/bin/bash
# Render every replay spec of a run (render_run.py) in parallel on a CPU node; 2 llvmpipe threads per replay.
R=$1; export LP_NUM_THREADS=2; date -Is
ls $R/videos/specs/*.args | xargs -P 15 -I{} bash -c "~/code/pear_local/replay.sh \$(cat {}) > {}.log 2>&1; echo \"\$(date -Is) {} rc=\$?\""
date -Is

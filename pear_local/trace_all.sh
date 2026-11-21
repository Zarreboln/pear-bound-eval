#!/bin/bash
# Goal-predicate traces for every episode of a run (CPU node, parallel).
R=$1; date -Is
ls $R/videos/specs/*.actions | xargs -n1 basename | sed "s/.actions//" | xargs -P 30 -I{} bash -c "~/code/pear_local/run_libero_py.sh trace_goals.py $R/videos/specs {} $R/analysis/goals/{}.json > $R/analysis/goals/{}.log 2>&1; echo {} rc=\$?"
date -Is

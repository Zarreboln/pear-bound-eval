#!/bin/bash
# Runs a script of pear_local in the LIBERO simulator environment with the rendering variables of LiberoEnv.setup_env
# (Mesa software EGL, no GPU): run_libero_py.sh <script> <args...>
set -euo pipefail
source ~/code/pear_local/env.sh
MESA=$WORK_DIR/envs/mesa
export LIBERO_CONFIG_PATH=$WORK_DIR/envs/libero/etc/libero
export LD_LIBRARY_PATH=$MESA/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export __EGL_VENDOR_LIBRARY_FILENAMES=$MESA/share/glvnd/egl_vendor.d/50_mesa.json
export LIBGL_DRIVERS_PATH=$MESA/lib/dri GALLIUM_DRIVER=llvmpipe LIBGL_ALWAYS_SOFTWARE=1 LP_NUM_THREADS=${LP_NUM_THREADS:-4}
export XDG_RUNTIME_DIR=${TMPDIR:-/tmp}/xdg_$USER CUDA_VISIBLE_DEVICES=
export PYTHONPATH=$HOME/code/pear
exec $WORK_DIR/envs/libero/bin/python "$HOME/code/pear_local/$1" "${@:2}"

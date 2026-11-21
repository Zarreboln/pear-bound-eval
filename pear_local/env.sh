# source me: PEAR eval settings (set WORK_DIR to the directory holding the environments, models and runs)
export WORK_DIR=${WORK_DIR:-$HOME/pear_work}
export MODEL_FILE=$HOME/code/pear_local/qwen3_8_27b_pear_ckpt.sh
export MODEL=$WORK_DIR/models/Qwen3.8-27B-PEAR
export GPU=h100 CUDA_ARCH=9.0

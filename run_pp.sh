#!/bin/bash
# Run Pipeline Parallel training on 8 Ascend NPU cards

# Load CANN environment
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate npu_train

# Use proxy for external network access + HuggingFace mirror
export https_proxy=http://127.0.0.1:7890
export http_proxy=http://127.0.0.1:7890
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DOWNLOAD_TIMEOUT=600

# PP config
# For 14B model with 4-way pipeline parallel
MODEL_NAME="/home/sd/npu_train/models/Qwen-Qwen2.5-14B-Instruct"
OUTPUT_PATH="./output_pp"
PP_SIZE=4

# Override with command line args
if [ "$1" = "--model" ]; then
    MODEL_NAME="$2"
    shift 2
fi

if [ "$1" = "--pp_size" ]; then
    PP_SIZE="$2"
    shift 2
fi

# Launch PP training
torchrun --nproc_per_node=8 \
    train_pp.py \
    --model_name "$MODEL_NAME" \
    --batch_size 2 \
    --max_length 512 \
    --epochs 2 \
    --lr 1e-5 \
    --warmup_steps 50 \
    --log_interval 5 \
    --max_samples 5000 \
    --pp_size "$PP_SIZE" \
    --micro_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --save_path "$OUTPUT_PATH" \
    "$@"

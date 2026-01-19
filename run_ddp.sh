#!/bin/bash
# Run DDP training on 8 Ascend NPU cards

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

# DDP config
# Using torch.distributed.launch for multi-GPU training
# 8 NPU cards on a single machine

MODEL_NAME="/home/sd/npu_train/models/Qwen-Qwen2.5-1.5B-Instruct"
OUTPUT_PATH="./output_ddp"

# Override with command line args
if [ "$1" = "--model" ]; then
    MODEL_NAME="$2"
    shift 2
fi

# Launch DDP training using torchrun
# For NPU, we use the same approach as GPU but with torch_npu backend
torchrun --nproc_per_node=8 \
    train_ddp.py \
    --model_name "$MODEL_NAME" \
    --batch_size 8 \
    --max_length 256 \
    --epochs 3 \
    --lr 5e-5 \
    --warmup_steps 100 \
    --log_interval 10 \
    --max_samples 10000 \
    --gradient_accumulation_steps 1 \
    --save_path "$OUTPUT_PATH" \
    "$@"

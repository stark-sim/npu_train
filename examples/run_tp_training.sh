#!/bin/bash
# Robust launch script for distributed TP training with HCCL port management

set -e

# Setup environment
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source ~/miniconda3/etc/profile.d/conda.sh
conda activate npu_train

# Configuration
TP_SIZE=${TP_SIZE:-2}  # Default: 2-way tensor parallelism
WORLD_SIZE=${WORLD_SIZE:-2}  # Default: use 2 NPUs
MASTER_PORT=${MASTER_PORT:-29500}
MODEL_PATH=${MODEL_PATH:-"/home/sd/npu_train/models/Qwen-Qwen2.5-1.5B-Instruct"}

# HCCL environment variables
export HCCL_WHITELIST_DISABLE=1
export HCCL_INTRA_PCIE_ENABLE=1
export HCCL_INTER_PCIE_ENABLE=1

# For debugging - get accurate stack traces
export ASCEND_LAUNCH_BLOCKING=1

# Find available port
find_available_port() {
    for port in $(seq 29500 29600); do
        if ! netstat -tuln 2>/dev/null | grep -q ":$port "; then
            echo $port
            return 0
        fi
    done
    echo "No available port found!" >&2
    return 1
}

AVAILABLE_PORT=$(find_available_port)
if [ $? -ne 0 ]; then
    echo "ERROR: Could not find available port for training"
    echo "Please stop existing distributed processes first"
    echo "Run: ps aux | grep torchrun | awk '{print \$2}' | xargs -r kill -9"
    exit 1
fi

echo "========================================"
echo "NPU TP Training"
echo "========================================"
echo "TP_SIZE: $TP_SIZE"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "MASTER_PORT: $AVAILABLE_PORT"
echo "Model: $MODEL_PATH"
echo "========================================"

cd ~/npu_train

# Run training
torchrun --nproc_per_node=$WORLD_SIZE \
    --master_port=$AVAILABLE_PORT \
    train_tp_custom.py \
    --model_path "$MODEL_PATH" \
    --tp_size $TP_SIZE \
    --batch_size 2 \
    --max_length 512 \
    --epochs 1 \
    --lr 1e-4 \
    --save_path "./output_tp_1b5"

echo ""
echo "Training completed!"

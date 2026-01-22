#!/bin/bash
# Run 8-way TP training for 1.5B model

set -e

# Setup environment
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source ~/miniconda3/etc/profile.d/conda.sh
conda activate npu_train

# Configuration
TP_SIZE=8
WORLD_SIZE=8
MASTER_PORT=29601
MODEL_PATH="/home/sd/npu_train/models/Qwen-Qwen2.5-1.5B-Instruct"

# HCCL environment variables
export HCCL_WHITELIST_DISABLE=1
export HCCL_INTRA_PCIE_ENABLE=1
export HCCL_INTER_PCIE_ENABLE=1

# For debugging - get accurate stack traces
export ASCEND_LAUNCH_BLOCKING=1

echo "========================================"
echo "NPU 8-Way TP Training - 1.5B Model"
echo "========================================"
echo "TP_SIZE: $TP_SIZE"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "MASTER_PORT: $MASTER_PORT"
echo "Model: $MODEL_PATH"
echo "========================================"

cd /home/sd/npu_train

# Run training
timeout 1800 torchrun --nproc_per_node=$WORLD_SIZE \
    --master_port=$MASTER_PORT \
    train_tp_custom.py \
    --model_path "$MODEL_PATH" \
    --tp_size $TP_SIZE \
    --batch_size 2 \
    --max_length 256 \
    --epochs 1 \
    --lr 1e-4 \
    --save_path "./output_tp_1b5_8way"

echo ""
echo "Training completed!"

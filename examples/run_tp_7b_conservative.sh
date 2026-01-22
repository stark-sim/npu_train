#!/bin/bash
# Run 7B model with conservative settings

set -e

# Setup environment
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source ~/miniconda3/etc/profile.d/conda.sh
conda activate npu_train

# HCCL environment variables
export HCCL_WHITELIST_DISABLE=1
export HCCL_INTRA_PCIE_ENABLE=1
export HCCL_INTER_PCIE_ENABLE=1
export ASCEND_LAUNCH_BLOCKING=1

cd /home/sd/npu_train

echo "========================================"
echo "7B Model Conservative Settings Test"
echo "========================================"

timeout 1800 torchrun --nproc_per_node=8 --master_port=29603 \
    train_tp_custom.py \
    --model_path "/home/sd/npu_train/models/Qwen-Qwen2.5-7B-Instruct" \
    --tp_size 8 \
    --batch_size 1 \
    --max_length 64 \
    --epochs 1 \
    --lr 1e-4 \
    --save_path "./output_tp_7b_conservative"

echo ""
echo "Test completed!"

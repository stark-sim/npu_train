#!/bin/bash
# Launch script for testing custom NPU TP with 1.5B model

# Setup environment
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source ~/miniconda3/etc/profile.d/conda.sh
conda activate npu_train

# Configuration
export TP_SIZE=2
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

# Model path
MODEL_PATH="/home/sd/npu_train/models/Qwen-Qwen2.5-1.5B-Instruct"

echo "========================================"
echo "NPU Custom TP Test - 1.5B Qwen"
echo "========================================"
echo "TP_SIZE: $TP_SIZE"
echo "Model: $MODEL_PATH"
echo "========================================"

# Run test with 2 NPUs (tp_size=2)
cd ~/npu_train

torchrun --nproc_per_node=2 \
    --master_port=$MASTER_PORT \
    test_tp_custom_simple.py \
    --tp_size $TP_SIZE

echo ""
echo "Test completed!"

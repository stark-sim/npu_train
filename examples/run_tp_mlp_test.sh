#!/bin/bash
# Run TP MLP test with proper environment setup

set -e

# Setup environment
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source ~/miniconda3/etc/profile.d/conda.sh
conda activate npu_train

# HCCL environment variables
export HCCL_WHITELIST_DISABLE=1
export HCCL_INTRA_PCIE_ENABLE=1
export HCCL_INTER_PCIE_ENABLE=1

# For debugging - get accurate stack traces
export ASCEND_LAUNCH_BLOCKING=1

# Configuration
WORLD_SIZE=${WORLD_SIZE:-2}
MASTER_PORT=${MASTER_PORT:-29801}

echo "========================================"
echo "NPU TP MLP Test"
echo "========================================"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "MASTER_PORT: $MASTER_PORT"
echo "========================================"

cd /home/sd/npu_train

# Run test
torchrun --nproc_per_node=$WORLD_SIZE \
    --master_port=$MASTER_PORT \
    test_tp_mlp_only.py

echo ""
echo "Test completed!"

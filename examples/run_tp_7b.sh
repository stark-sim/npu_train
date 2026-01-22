#!/bin/bash
# Launch 8 NPU training with Tensor Parallelism using PyTorch native TP

# Load CANN environment
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate npu_train

# Network configuration (uncomment if needed)
# export https_proxy=http://127.0.0.1:7890
# export http_proxy=http://127.0.0.1:7890
# export HF_ENDPOINT=https://hf-mirror.com
# export HF_HUB_DOWNLOAD_TIMEOUT=600

# Tensor Parallel configuration
export TP_SIZE=4          # Tensor parallel degree (4 for 7B model on 8 NPUs)
export WORLD_SIZE=8         # Total number of NPUs
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

# Training configuration
MODEL_PATH="/home/sd/npu_train/models/AI-ModelScope/qwen2.5-7b-instruct"
BATCH_SIZE=2
MAX_LENGTH=2048
EPOCHS=1
LR=1e-5
SAVE_PATH="./output_tp_native"

echo "======================================================================"
echo "Tensor Parallel Training - PyTorch Native TP"
echo "======================================================================"
echo "Model: $MODEL_PATH"
echo "TP Size: $TP_SIZE"
echo "World Size: $WORLD_SIZE"
echo "Batch Size: $BATCH_SIZE"
echo "Max Length: $MAX_LENGTH"
echo "======================================================================"

# Launch with torchrun
torchrun \
    --nproc_per_node=$WORLD_SIZE \
    --master_port=$MASTER_PORT \
    train_tp_7b.py \
    --model_path "$MODEL_PATH" \
    --tp_size $TP_SIZE \
    --batch_size $BATCH_SIZE \
    --max_length $MAX_LENGTH \
    --epochs $EPOCHS \
    --lr $LR \
    --warmup_steps 10 \
    --log_interval 5 \
    --save_path "$SAVE_PATH" \
    --use_amp

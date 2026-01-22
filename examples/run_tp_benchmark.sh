#!/bin/bash
# NPU TP 性能基准测试脚本

# 设置 CANN 环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 模型路径
MODEL_PATH="/home/sd/npu_train/models/AI-ModelScope/qwen2.5-1.5b-instruct"

echo "========================================"
echo "NPU TP 性能基准测试"
echo "========================================"
echo "模型: $MODEL_PATH"
echo "========================================"

# 测试配置
BATCH_SIZE=1
SEQ_LEN=512
NUM_STEPS=20

echo ""
echo "测试 1: TP=1,2,4 对比 (4卡)"
echo "----------------------------------------"
torchrun --nproc_per_node=4 \
    tests/benchmark_tp_performance.py \
    --model_path "$MODEL_PATH" \
    --batch_size $BATCH_SIZE \
    --seq_len $SEQ_LEN \
    --num_steps $NUM_STEPS \
    --tp_sizes 1,2,4

echo ""
echo "测试 2: 8-way TP (8卡)"
echo "----------------------------------------"
torchrun --nproc_per_node=8 \
    tests/benchmark_tp_performance.py \
    --model_path "$MODEL_PATH" \
    --batch_size $BATCH_SIZE \
    --seq_len $SEQ_LEN \
    --num_steps $NUM_STEPS \
    --tp_sizes 8

echo ""
echo "========================================"
echo "基准测试完成"
echo "========================================"

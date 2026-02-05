#!/bin/bash
#
# 真实 LLM 训练启动脚本
#
# 使用方法:
#   chmod +x run_real_training.sh
#   ./run_real_training.sh
#
# 环境要求:
#   - 8 卡华为 Ascend 910A NPU
#   - Python 3.11 + torch-npu 2.5.1
#   - 数据集已下载到 /home/sd/npu_train/datasets/
#

set -e

# ==================== 配置参数 ====================

# 模型路径
MODEL_PATH="/home/sd/npu_train/models/deepseek-ai-DeepSeek-V2-Lite"

# 数据集路径
DATASET_PATH="/home/sd/npu_train/datasets/wikitext-103"

# 输出路径
OUTPUT_PATH="./output_real_llm"

# 训练参数
NUM_STEPS=200000         # 总训练步数 (2-3天)
BATCH_SIZE=2             # 每卡批次大小
MAX_LENGTH=512           # 最大序列长度
LR=1e-5                  # 学习率
WARMUP_STEPS=2000        # 预热步数
TP_SIZE=4                # 张量并行大小

# 评估和保存
EVAL_INTERVAL=5000       # 评估间隔
SAVE_INTERVAL=10000      # 保存间隔

# 分布式配置
NPROC_PER_NODE=8         # NPU 数量

# ==================== 环境变量 ====================

# HCCL 通信超时 (对于大模型很重要)
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=7200

# NPU 内存配置
export TORCH_NPU_ALLOC_CONF="max_split_size_mb:128"

# 启用 NPU 算子融合
export NPU_FUSION_ENABLE=1

# ==================== 打印配置 ====================

echo "=========================================="
echo "真实 LLM 训练配置"
echo "=========================================="
echo "模型: $MODEL_PATH"
echo "数据集: $DATASET_PATH"
echo "输出: $OUTPUT_PATH"
echo "------------------------------------------"
echo "训练步数: $NUM_STEPS"
echo "批次大小: $BATCH_SIZE"
echo "序列长度: $MAX_LENGTH"
echo "学习率: $LR"
echo "TP Size: $TP_SIZE"
echo "NPU 数量: $NPROC_PER_NODE"
echo "=========================================="
echo ""

# ==================== 检查数据集 ====================

if [ ! -d "$DATASET_PATH" ]; then
    echo "错误: 数据集目录不存在: $DATASET_PATH"
    echo ""
    echo "请先下载数据集:"
    echo "  python examples/download_real_dataset.py --dataset wikitext"
    echo ""
    exit 1
fi

# ==================== 启动训练 ====================

echo "启动训练..."
echo ""

torchrun \
    --nproc_per_node=$NPROC_PER_NODE \
    --master_port=29500 \
    examples/train_real_llm.py \
    --model_path "$MODEL_PATH" \
    --dataset_path "$DATASET_PATH" \
    --output_path "$OUTPUT_PATH" \
    --tp_size $TP_SIZE \
    --batch_size $BATCH_SIZE \
    --max_length $MAX_LENGTH \
    --num_steps $NUM_STEPS \
    --lr $LR \
    --warmup_steps $WARMUP_STEPS \
    --eval_interval $EVAL_INTERVAL \
    --save_interval $SAVE_INTERVAL

echo ""
echo "=========================================="
echo "训练完成!"
echo "=========================================="

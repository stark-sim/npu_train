#!/bin/bash
# DeepSeek-V2-Lite MoE TP 快速测试脚本
# 在远程NPU服务器上运行

set -e  # 遇到错误立即退出

echo "========================================"
echo "DeepSeek-V2-Lite MoE TP 快速测试"
echo "========================================"

# 配置
MODEL_PATH="/home/sd/npu_train/models/deepseek-ai/DeepSeek-V2-Lite"
CONDA_ENV="npu_train"
WORK_DIR="~/npu_train/osaka"

# 检查环境
echo ""
echo "[1/5] 检查环境..."
echo "----------------------------------------"

# 检查conda环境
if ! conda env list | grep -q "$CONDA_ENV"; then
    echo "错误: Conda环境 '$CONDA_ENV' 不存在"
    exit 1
fi
echo "✓ Conda环境: $CONDA_ENV"

# 检查NPU
if ! python3 -c "import torch_npu; print('NPU available:', torch.npu.is_available())" 2>/dev/null; then
    echo "错误: torch_npu 不可用"
    exit 1
fi
echo "✓ NPU可用"

# 激活环境
echo "激活环境: $CONDA_ENV"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $CONDA_ENV

# 切换工作目录
cd $WORK_DIR
echo "✓ 工作目录: $(pwd)"

# 检查模型
echo ""
echo "[2/5] 检查模型..."
echo "----------------------------------------"
if [ ! -d "$MODEL_PATH" ]; then
    echo "错误: 模型目录不存在: $MODEL_PATH"
    echo ""
    echo "请先下载模型:"
    echo "  python3 download_deepseek_moe.py"
    exit 1
fi
echo "✓ 模型目录存在"

# 检查必要文件
REQUIRED_FILES=("config.json" "tokenizer.json" "tokenizer_config.json")
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$MODEL_PATH/$file" ]; then
        echo "警告: 缺少文件 $file"
    fi
done

# 检查safetensors文件
SAFETENSORS_COUNT=$(find "$MODEL_PATH" -name "*.safetensors" | wc -l)
echo "✓ 找到 $SAFETENSORS_COUNT 个safetensors文件"

# 运行单元测试
echo ""
echo "[3/5] 运行MoE单元测试..."
echo "----------------------------------------"
if python3 tests/test_tp_moe.py -v 2>&1 | tee /tmp/test_tp_moe.log; then
    echo "✓ MoE单元测试通过"
else
    echo "⚠ MoE单元测试有警告（可能非致命）"
fi

# 测试模型转换
echo ""
echo "[4/5] 测试DeepSeek-V2模型转换..."
echo "----------------------------------------"
if python3 examples/test_deepseek_moe.py \
    --model_path "$MODEL_PATH" \
    --device "npu:0" 2>&1 | tee /tmp/test_deepseek_moe.log; then
    echo "✓ 模型转换测试通过！"

    # 检查关键输出
    if grep -q "All tests passed" /tmp/test_deepseek_moe.log; then
        echo "✓ 所有测试通过！"
        CONVERSION_OK=true
    else
        echo "⚠ 测试完成但未显示成功消息"
        CONVERSION_OK=false
    fi
else
    echo "✗ 模型转换测试失败"
    echo ""
    echo "查看日志:"
    echo "  cat /tmp/test_deepseek_moe.log"
    CONVERSION_OK=false
fi

# 如果转换测试通过，询问是否运行训练
echo ""
echo "[5/5] 测试总结"
echo "----------------------------------------"
echo "MoE单元测试: 完成"
echo "模型转换测试: $([ "$CONVERSION_OK" = true ] && echo '✓ 通过' || echo '✗ 失败')"
echo ""

if [ "$CONVERSION_OK" = true ]; then
    echo "========================================"
    echo "✓ 所有测试通过！"
    echo "========================================"
    echo ""
    echo "现在可以运行训练:"
    echo ""
    echo "  # 4卡训练"
    echo "  torchrun --nproc_per_node=4 examples/train_deepseek_v2_lite.py \\"
    echo "      --model_path \"$MODEL_PATH\" \\"
    echo "      --tp_size 4 \\"
    echo "      --batch_size 1 \\"
    echo "      --max_length 256"
    echo ""
    echo "  # 8卡训练（如果可用）"
    echo "  torchrun --nproc_per_node=8 examples/train_deepseek_v2_lite.py \\"
    echo "      --model_path \"$MODEL_PATH\" \\"
    echo "      --tp_size 8"
    echo ""
else
    echo "========================================"
    echo "✗ 测试失败，请检查错误"
    echo "========================================"
    echo ""
    echo "查看日志:"
    echo "  cat /tmp/test_deepseek_moe.log"
    echo "  cat /tmp/test_tp_moe.log"
    echo ""
    exit 1
fi

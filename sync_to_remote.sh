#!/bin/bash
# 同步代码到远程NPU服务器的脚本

REMOTE_USER="sd"
REMOTE_HOST="bms-hit-haom-jbgs"
REMOTE_DIR="~/npu_train/osaka"

echo "========================================"
echo "同步代码到远程NPU服务器"
echo "========================================"
echo "目标: ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}"
echo ""

# 需要同步的文件列表
FILES=(
    "npu_parallel/tp_moe.py"
    "npu_parallel/convert_model.py"
    "npu_parallel/__init__.py"
    "npu_parallel/supported_models.py"
    "npu_parallel/tp_layers.py"
    "npu_parallel/tp_attention.py"
    "examples/train_tp_moe.py"
    "examples/train_deepseek_v2_lite.py"
    "examples/test_deepseek_moe.py"
    "tests/test_tp_moe.py"
    "download_deepseek_moe.py"
)

# 同步文件
for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "同步: $file"
        scp "$file" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/$file"
    else
        echo "警告: 文件不存在 - $file"
    fi
done

echo ""
echo "========================================"
echo "同步完成!"
echo "========================================"
echo ""
echo "接下来在远程服务器执行:"
echo ""
echo "1. 首先测试MoE转换:"
echo "   cd ~/npu_train/osaka"
echo "   conda activate npu_train"
echo "   python3 examples/test_deepseek_moe.py"
echo ""
echo "2. 如果测试通过，运行训练:"
echo "   torchrun --nproc_per_node=4 examples/train_deepseek_v2_lite.py \\"
echo "       --model_path \"/home/sd/npu_train/models/deepseek-ai/DeepSeek-V2-Lite\" \\"
echo "       --tp_size 4 \\"
echo "       --batch_size 1 \\"
echo "       --max_length 256"
echo ""

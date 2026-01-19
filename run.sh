#!/bin/bash
# Run training on Ascend NPU

# Load CANN environment
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate npu_train

# Use proxy for external network access + HuggingFace mirror
export https_proxy=http://127.0.0.1:7890
export http_proxy=http://127.0.0.1:7890
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DOWNLOAD_TIMEOUT=600

# Training config for ~30min on single NPU
# GPT-2 small: 124M params, wikitext-2: ~36k train samples
python3 train.py \
    --model_name /home/sd/npu_train/models/AI-ModelScope/gpt2 \
    --dataset wikitext \
    --dataset_config wikitext-2-raw-v1 \
    --batch_size 8 \
    --max_length 256 \
    --epochs 3 \
    --lr 5e-5 \
    --warmup_steps 100 \
    --log_interval 20 \
    --max_samples 5000 \
    --save_path ./output \
    "$@"

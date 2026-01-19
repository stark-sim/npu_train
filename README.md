# NPU Training Project

Small-scale LLM training on Huawei Ascend 910A NPU using PyTorch + HuggingFace.

## Environment

- Python 3.11
- PyTorch 2.8.0
- torch-npu 2.8.0
- transformers 4.57.x
- CANN 8.1.RC1

## Quick Start

```bash
# Make run script executable
chmod +x run.sh

# Run training (~30 minutes)
./run.sh
```

## Training Configuration

- **Model**: GPT-2 (124M parameters)
- **Dataset**: WikiText-2 (~5000 samples)
- **Batch size**: 8
- **Sequence length**: 256
- **Epochs**: 3
- **Learning rate**: 5e-5

## Custom Training

```bash
# Use different model
./run.sh --model_name distilgpt2

# Adjust batch size
./run.sh --batch_size 16

# More epochs
./run.sh --epochs 5
```

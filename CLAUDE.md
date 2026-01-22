# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python-based machine learning project for training small-scale LLMs on Huawei Ascend 910A NPU using PyTorch + HuggingFace transformers.

## Common Commands

### Training
```bash
# Quick start (default: GPT-2, ~30 minutes)
chmod +x run.sh && ./run.sh

# Custom training
./run.sh --model_name distilgpt2
./run.sh --batch_size 16 --epochs 5
```

### Model Download/Management
```bash
# Download Qwen models via ModelScope
python3 download_models.py

# Resume incomplete downloads
python3 simple_resume_download.py

# Fix corrupted downloads (safetensors integrity check)
python3 fix_downloads.py

# Download with aria2c (parallel, better resume)
python3 download_with_aria2.py

# Test model loading
python3 load_example.py

# Generate model manifest with PP/TP suggestions
python3 model_manifest.py
```

### Verification
```bash
# Check NPU availability
python3 -c "import torch_npu; print(torch.npu.is_available())"

# Verify dependencies
python3 -c "import torch, transformers; print(torch.__version__)"
```

## Architecture

### Training Flow
1. **run.sh**: Sets up CANN environment, activates conda `npu_train`, configures proxy/HF mirror, launches `train.py`
2. **train.py**: Main training loop with NPU device setup, model loading via HuggingFace, synthetic data generation (9 repeated text templates), forward/backward passes with gradient clipping

### Model Download Utilities
- `download_models.py`: Primary ModelScope downloader
- `download_with_aria2.py`: Multi-threaded download with resume
- `download_with_wget.py`: Sequential download with resume
- `simple_resume_download.py`: Retry-based fallback
- `fix_downloads.py`: Safetensors integrity checker and repair

## Environment Setup

The project requires:
- Python 3.11
- PyTorch 2.5.1 + torch-npu 2.5.1
- transformers 4.57.x
- CANN 8.1.RC1 (`/usr/local/Ascend/ascend-toolkit/set_env.sh`)
- Conda environment: `npu_train` at `~/miniconda3/etc/profile.d/conda.sh`

## Important Paths

- Model storage: `/home/sd/npu_train/models`
- Models are loaded from paths like `/home/sd/npu_train/models/AI-ModelScope/gpt2`
- Default model save path: `./output`

## Network Configuration

- HTTP proxy: `http://127.0.0.1:7890`
- HuggingFace mirror: `https://hf-mirror.com`
- HF download timeout: `600s`

## NPU-Specific Notes

- Use `torch.npu.set_device(local_rank)` instead of CUDA
- Device selection: `torch.device(f"npu:{local_rank}")`
- HCCL timeout: `HCCL_CONNECT_TIMEOUT=1200` (set in train.py)
- All tensor operations use NPU backend via `torch_npu`

## Training Configuration Defaults

| Parameter | Value |
|-----------|-------|
| model | GPT-2 (124M params) |
| dataset | Synthetic (5000 samples) |
| batch_size | 8 |
| max_length | 256 |
| epochs | 3 |
| learning_rate | 5e-5 |
| warmup_steps | 100 |

## Data Strategy

Training uses **synthetic data** (9 repeated text templates cycled for `max_samples`) to avoid network issues with HuggingFace datasets download. The data is tokenized and padded to `max_length`.

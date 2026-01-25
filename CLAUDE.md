# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python-based machine learning project for training LLMs on Huawei Ascend 910A NPU using PyTorch + HuggingFace transformers. It implements custom **Tensor Parallelism (TP)** for distributed training across multiple NPUs, along with Data Parallel (DDP) and Pipeline Parallel (PP) support.

## Common Commands

### Training
```bash
# Single NPU training (default: GPT-2, ~30 minutes)
chmod +x run.sh && ./run.sh

# Data Parallel training (8 NPUs)
chmod +x run_ddp.sh && ./run_ddp.sh

# Pipeline Parallel training (8 NPUs, 4-stage pipeline)
chmod +x run_pp.sh && ./run_pp.sh

# Tensor Parallel training (custom TP implementation)
python examples/train_tp_custom.py --model_path "/path/to/model" --tp_size 4
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

# Run TP tests
python3 tests/test_tp_conversion.py
python3 tests/test_tp_mlp_only.py
python3 tests/test_hccl_ops.py
```

## Architecture

### Training Scripts
| Script | Purpose | NPUs |
|--------|---------|------|
| `train.py` | Single NPU baseline | 1 |
| `train_ddp.py` | Data Parallel (replicated model) | 8 |
| `train_pp.py` | Pipeline Parallel (layer split) | 8 |
| `examples/train_tp*.py` | Tensor Parallel (weight split) | 2-8 |

### npu_parallel Module

Custom Tensor Parallelism implementation based on Megatron-LM patterns, adapted for NPU/HCCL.

**tp_layers.py**: Core TP building blocks
- `ColumnParallelLinear`: Splits weights by columns (output dimension), uses all-gather
- `RowParallelLinear`: Splits weights by rows (input dimension), uses all-reduce
- `AllGatherFromTensor`: Custom autograd function for HCCL compatibility
- `TPProcessGroup`: Manages hybrid TP+DDP process groups

**tp_attention.py**: TP-aware transformer components
- `TPQKVParallel`: Combined Q/K/V projection (column parallel)
- `TPOutputParallel`: Attention output projection (row parallel)
- `TPAttention`: Complete multi-head attention with TP
- `TPMLP`: SwiGLU feed-forward with TP

**convert_model.py**: Model conversion utilities
- `convert_to_tp()`: Auto-detects and converts HuggingFace models to TP
- Supports: Qwen, Llama, Mistral, Gemma, Phi, Yi, DeepSeek, Baichuan, GPT-2
- Two architecture patterns: `qwen_style` (separate Q,K,V) and `gpt2_style` (combined QKV)

### Communication Patterns
- **all_gather**: For column parallel (combine partial outputs from each rank)
- **all_reduce**: For row parallel (sum partial results from each rank)
- Backend: HCCL (Huawei Collective Communication Library)

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

## Supported Models for TP

The `npu_parallel` module supports converting HuggingFace models to Tensor Parallelism:

| Model Family | Status | Architecture Pattern |
|--------------|--------|---------------------|
| Qwen/Qwen2/Qwen2.5 | ✅ Tested | qwen_style (separate Q,K,V) |
| GPT-2/GPT-Neo | ✅ Tested | gpt2_style (combined QKV) |
| Llama/Llama2/Llama3 | ⚠️ Should work | qwen_style |
| Mistral/Mixtral | ⚠️ Should work | qwen_style |
| Gemma/Gemma2 | ⚠️ Should work | qwen_style |
| Phi-2/Phi-3 | ⚠️ Should work | qwen_style |
| Yi, DeepSeek, Baichuan2 | ⚠️ Should work | qwen_style |

See `npu_parallel/supported_models.py` for full compatibility reference.

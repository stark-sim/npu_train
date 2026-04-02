# NPU Training Project

> **⚠️ PROJECT STATUS: ARCHIVED (2026-04-02)**  
> This project has been completed and archived. See [ARCHIVE_STATUS.md](ARCHIVE_STATUS.md) for details.  
> Tag: `v910a-complete-20260402` | Branch: `stark-sim/real-moe-smoke`

LLM training on Huawei Ascend 910A NPU with support for Data Parallel (DDP), Pipeline Parallel (PP), custom Tensor Parallelism (TP), and DeepSeek MoE.

## Environment

- Python 3.11
- PyTorch 2.5.1 + torch-npu 2.5.1
- transformers 4.57.x
- CANN 8.1.RC1

## Quick Start

```bash
# Single NPU training
chmod +x run.sh && ./run.sh

# CPU smoke tests (no NPU required)
python tests/test_npu_compat_layer.py
python tests/test_npu_compat_log_analyze.py
python tests/test_tp_attention_compat.py
```

## Training Modes

### Single NPU
```bash
./run.sh --model_name distilgpt2 --batch_size 16 --epochs 5
```

### Data Parallel (DDP) - 8 NPUs
Replicates model across all NPUs, synchronizes gradients.
```bash
chmod +x run_ddp.sh && ./run_ddp.sh
```

### Pipeline Parallel (PP) - 8 NPUs
Splits model layers across NPUs (e.g., 4-stage pipeline).
```bash
chmod +x run_pp.sh && ./run_pp.sh --pp_size 4
```

### Tensor Parallel (TP) - Custom Implementation
Splits weights across NPUs using custom `npu_parallel` module (Megatron-LM style).
```bash
# Single NPU TP smoke test
python examples/train_tp_custom.py \
  --model_path "/path/to/model" \
  --tp_size 1 --max_steps 2 --skip_save

# 4-card TP training (on 910A)
torchrun --nproc_per_node=4 examples/train_tp_custom.py \
  --model_path "/path/to/model" \
  --tp_size 4 --max_steps 100

# With compatibility report
python examples/train_tp_custom.py \
  --model_path "/path/to/model" \
  --tp_size 4 --max_steps 10 \
  --compat_report_file compat_report.json
```

### DeepSeek MoE with TP
```bash
torchrun --nproc_per_node=4 examples/train_tp_moe.py \
  --model_path "/path/to/DeepSeek-V2-Lite" \
  --tp_size 4 --max_steps 2 --skip_save
```

## npu_parallel Module

Custom Tensor Parallelism implementation for Ascend NPU/HCCL:

| Component | Description |
|-----------|-------------|
| `tp_layers.py` | `ColumnParallelLinear`, `RowParallelLinear` with autograd-aware collectives |
| `tp_attention.py` | `TPAttention`, `TPMLP` with SwiGLU |
| `tp_moe.py` | `TPMoERouter`, expert parallelism, DeepSeek compatibility |
| `convert_model.py` | Auto-convert HuggingFace models to TP |
| `checkpoint_utils.py` | TP checkpoint save/load with optimizer state |
| `npu_compat.py` | NPU compatibility layer for low-CANN environments |

Supported models: Qwen/Qwen2/Qwen2.5, Llama, Mistral, Gemma, Phi, Yi, DeepSeek, Baichuan, GPT-2.

### NPU Compatibility Layer

For Ascend 910A + lower CANN environments where certain operators may fail:

```python
# Policy: fallback (default), warn, or strict
python examples/train_tp_moe.py \
  --model_path "/path/to/model" \
  --tp_size 4 \
  --compat_policy fallback \
  --compat_report_file report.json

# Analyze logs for new error signatures
python tools/npu_compat_log_analyze.py logs/

# Benchmark raw vs safe paths
python tools/npu_compat_benchmark.py --device npu:0
```

## TP Checkpoint Tooling

```bash
# Inspect checkpoint
python tools/tp_checkpoint.py /path/to/checkpoint --inspect

# Export merged checkpoint (single file)
python tools/tp_checkpoint.py /path/to/checkpoint --export merged.bin

# Reshard to different TP size
python tools/tp_checkpoint.py /path/to/checkpoint \
  --reshard /path/to/new_checkpoint \
  --new_tp_size 2

# Resume training
python examples/train_tp_custom.py \
  --model_path "/path/to/model" \
  --tp_size 4 \
  --resume_from /path/to/checkpoint
```

## Model Management

```bash
# Download models via ModelScope
python3 download_models.py

# Download with aria2c (parallel, resume-capable)
python3 download_with_aria2.py

# Fix corrupted downloads (safetensors integrity)
python3 fix_downloads.py

# Generate model manifest with PP/TP suggestions
python3 model_manifest.py
```

## Verification

```bash
# Check NPU availability
python3 -c "import torch_npu; print(torch.npu.is_available())"

# Run tests
python3 tests/test_tp_conversion.py
python3 tests/test_tp_mlp_only.py
python3 tests/test_hccl_ops.py

# Compatibility layer tests
python3 tests/test_npu_compat_layer.py
python3 tests/test_npu_compat_log_analyze.py
python3 tests/test_tp_attention_compat.py

# Checkpoint tests
python3 tests/test_tp_checkpoint_tool.py
python3 tests/test_tp_checkpoint_resume.py
```

## Configuration

| Setting | Value |
|---------|-------|
| Model storage | `/home/sd/npu_train/models` |
| HTTP proxy | `http://127.0.0.1:7890` |
| HF mirror | `https://hf-mirror.com` |
| HCCL timeout | `1200s` |

## Documentation

- [Project Status Summary (CN)](docs/project-status/stage-results-short.zh.md)
- [Project Status Summary (EN)](docs/project-status/stage-results-short.en.md)
- [Completion Report](docs/project-status/COMPLETION_REPORT.md)
- [Storage Offset Diagnosis](docs/project-status/storage_offset_diagnosis.md)

## Memory Bank

This project uses [Memory Bank](memory-bank/) for cross-session context continuity. Key files:
- [Active Context](memory-bank/activeContext.md) - Current work and decisions
- [Progress](memory-bank/progress.md) - Completed work and known issues
- [System Patterns](memory-bank/systemPatterns.md) - Architecture decisions
- [Tech Context](memory-bank/techContext.md) - Development environment

## Project Status

**Phase**: Ascend 910A development complete  
**Key Achievements**:
- ✅ NPU compatibility layer for low-CANN environments
- ✅ TP checkpoint save/load/reshard/resume
- ✅ DeepSeek-V2-Lite MoE TP training
- ✅ Real-device validation on 8-card 910A
- ✅ Comprehensive diagnostic tooling

See [docs/project-status/](docs/project-status/) for detailed status.

# NPU Training Project

LLM training on Huawei Ascend 910A NPU with support for Data Parallel (DDP), Pipeline Parallel (PP), and custom Tensor Parallelism (TP).

## Environment

- Python 3.11
- PyTorch 2.5.1 + torch-npu 2.5.1
- transformers 4.57.x
- CANN 8.1.RC1

## Quick Start

```bash
# Single NPU training (~30 minutes)
chmod +x run.sh && ./run.sh
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
python examples/train_tp_custom.py --model_path "/path/to/model" --tp_size 4
```

## npu_parallel Module

Custom Tensor Parallelism implementation for Ascend NPU/HCCL:

- **tp_layers.py**: `ColumnParallelLinear`, `RowParallelLinear`
- **tp_attention.py**: `TPAttention`, `TPMLP` with SwiGLU
- **convert_model.py**: Auto-convert HuggingFace models to TP

Supported models: Qwen/Qwen2/Qwen2.5, Llama, Mistral, Gemma, Phi, Yi, DeepSeek, Baichuan, GPT-2.

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
```

## Configuration

| Setting | Value |
|---------|-------|
| Model storage | `/home/sd/npu_train/models` |
| HTTP proxy | `http://127.0.0.1:7890` |
| HF mirror | `https://hf-mirror.com` |
| HCCL timeout | `1200s` |

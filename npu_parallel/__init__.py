"""
NPU Parallel: Tensor Parallelism implementation for Ascend NPU using torch-npu

This package provides tensor parallelism components for training large language models
on Ascend 910A NPUs with PyTorch + torch-npu.

Components:
- tp_layers: ColumnParallelLinear, RowParallelLinear
- tp_attention: TP-aware attention mechanism
- convert_model: Utilities to convert HuggingFace models to TP
- supported_models: List of compatible model families

Supported Model Families:
- Qwen/Qwen2/Qwen2.5 ✅ (tested)
- Llama/Llama2/Llama3 ⚠️ (should work)
- Mistral/Mixtral ⚠️ (should work)
- Gemma/Gemma2 ⚠️ (should work)
- Phi-2/Phi-3 ⚠️ (should work)
- Yi ⚠️ (should work)
- DeepSeek ⚠️ (should work)
- Baichuan2 ⚠️ (should work)
- GPT-2 ✅ (tested)
"""

from .tp_layers import ColumnParallelLinear, RowParallelLinear, TPProcessGroup
from .convert_model import convert_to_tp, sync_gradients_tp

__version__ = "0.1.0"
__all__ = [
    "ColumnParallelLinear",
    "RowParallelLinear",
    "TPProcessGroup",
    "convert_to_tp",
    "sync_gradients_tp",
]

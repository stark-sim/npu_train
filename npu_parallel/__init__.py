"""
NPU Parallel: Tensor Parallelism implementation for Ascend NPU using torch-npu

This package provides tensor parallelism components for training large language models
on Ascend 910A NPUs with PyTorch + torch-npu.

Components:
- tp_layers: ColumnParallelLinear, RowParallelLinear
- tp_attention: TP-aware attention mechanism
- tp_moe: MoE (Mixture-of-Experts) support with expert parallelism
- convert_model: Utilities to convert HuggingFace models to TP
- supported_models: List of compatible model families

Supported Model Families:
- Qwen/Qwen2/Qwen2.5 ✅ (tested)
- Llama/Llama2/Llama3 ⚠️ (should work)
- Mistral/Mixtral-8x7B ⚠️ (MoE now supported!)
- DeepSeek-V2/V3 ⚠️ (MoE now supported!)
- Gemma/Gemma2 ⚠️ (should work)
- Phi-2/Phi-3 ⚠️ (should work)
- Yi ⚠️ (should work)
- Qwen2MoE ⚠️ (MoE now supported!)
- Baichuan2 ⚠️ (should work)
- GPT-2 ✅ (tested)
"""

from .tp_layers import ColumnParallelLinear, RowParallelLinear, TPProcessGroup
from .tp_moe import TPMoELayer, TPDeepSeekMoE
from .convert_model import convert_to_tp, sync_gradients_tp
from .checkpoint_utils import load_tp_rank_checkpoint, save_tp_rank_checkpoint, write_tp_state_dict_shards
from .npu_compat import (
    analyze_error_messages,
    analyze_log_file,
    analyze_log_text,
    build_outcome_objective,
    build_signature_update_plan,
    classify_runtime_error,
    compatibility_report,
    get_compat_policy,
    get_fallback_stats,
    get_perf_counters,
    known_error_signatures,
    recommended_action,
    render_signature_patch_template,
    reset_fallback_stats,
    runtime_info,
    safe_has_any_tokens,
    safe_nonzero,
    safe_softmax,
    safe_topk,
    set_compat_policy,
    supports_op,
)

__version__ = "0.2.0"
__all__ = [
    "ColumnParallelLinear",
    "RowParallelLinear",
    "TPProcessGroup",
    "TPMoELayer",
    "TPDeepSeekMoE",
    "convert_to_tp",
    "sync_gradients_tp",
    "save_tp_rank_checkpoint",
    "load_tp_rank_checkpoint",
    "write_tp_state_dict_shards",
    "runtime_info",
    "analyze_error_messages",
    "analyze_log_text",
    "analyze_log_file",
    "build_outcome_objective",
    "build_signature_update_plan",
    "classify_runtime_error",
    "compatibility_report",
    "get_compat_policy",
    "set_compat_policy",
    "get_fallback_stats",
    "get_perf_counters",
    "known_error_signatures",
    "recommended_action",
    "render_signature_patch_template",
    "reset_fallback_stats",
    "supports_op",
    "safe_topk",
    "safe_softmax",
    "safe_has_any_tokens",
    "safe_nonzero",
]

"""
NPU Tensor Parallelism - Supported Models

This module lists all supported models and provides quick reference for usage.
"""

SUPPORTED_MODELS = {
    # Qwen Series (Alibaba)
    "qwen": {
        "models": ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-14B", "Qwen/Qwen2.5-32B", "Qwen/Qwen2-72B"],
        "architecture": "GPT + SwiGLU",
        "tested": True,
        "notes": "Fully tested, recommended"
    },

    # Llama Series (Meta)
    "llama": {
        "models": ["meta-llama/Llama-2-7b", "meta-llama/Llama-2-13b", "meta-llama/Meta-Llama-3-8B", "meta-llama/Meta-Llama-3.1-8B"],
        "architecture": "GPT + SwiGLU",
        "tested": False,
        "notes": "Should work, same architecture as Qwen"
    },

    # Mistral Series
    "mistral": {
        "models": ["mistralai/Mistral-7B-v0.1", "mistralai/Mistral-7B-v0.3", "mistralai/Mixtral-8x7B-v0.1"],
        "architecture": "GPT + SwiGLU + MoE (Mixtral)",
        "tested": False,
        "notes": "Dense models work. MoE (Mixtral-8x7B) now supported with TP!"
    },

    # Gemma Series (Google)
    "gemma": {
        "models": ["google/gemma-2-9b", "google/gemma-2-27b"],
        "architecture": "GPT + SwiGLU",
        "tested": False,
        "notes": "Should work"
    },

    # Phi Series (Microsoft)
    "phi": {
        "models": ["microsoft/phi-2", "microsoft/Phi-3-mini-4k-instruct"],
        "architecture": "GPT + SwiGLU",
        "tested": False,
        "notes": "Should work"
    },

    # Yi Series (01.AI)
    "yi": {
        "models": ["01-ai/Yi-6B", "01-ai/Yi-34B"],
        "architecture": "GPT + SwiGLU",
        "tested": False,
        "notes": "Should work"
    },

    # DeepSeek Series
    "deepseek": {
        "models": ["deepseek-ai/deepseek-coder-6.7b", "deepseek-ai/DeepSeek-V2-Lite", "deepseek-ai/DeepSeek-V2-Chat"],
        "architecture": "GPT + SwiGLU + MoE (DeepSeek-V2/V3)",
        "tested": False,
        "notes": "Dense models work. MoE (DeepSeek-V2) now supported with TP! Use train_deepseek_v2_lite.py for V2-Lite."
    },

    # Qwen2MoE Series
    "qwen2moe": {
        "models": ["Qwen/Qwen1.5-MoE-A2.7B", "Qwen/Qwen2-57B-A14B"],
        "architecture": "GPT + SwiGLU + MoE",
        "tested": False,
        "notes": "MoE now supported with TP!"
    },

    # Baichuan Series
    "baichuan": {
        "models": ["baichuan-inc/Baichuan2-7B-Base"],
        "architecture": "GPT + SwiGLU",
        "tested": False,
        "notes": "Should work"
    },

    # InternLM Series
    "internlm": {
        "models": ["internlm/internlm-7b", "internlm/internlm2-7b"],
        "architecture": "GPT + SwiGLU",
        "tested": False,
        "notes": "Should work"
    },
}

ARCHITECTURE_PATTERNS = {
    "qwen_style": {
        "attention": "Separate Q, K, V projections (Column Parallel)",
        "attention_out": "O projection (Row Parallel)",
        "mlp": "Gate + Up projections (Column Parallel)",
        "mlp_out": "Down projection (Row Parallel)",
        "models": ["Qwen", "Llama", "Mistral", "Gemma", "Phi", "Yi", "DeepSeek", "Baichuan", "InternLM"]
    },
    "gpt2_style": {
        "attention": "Combined QKV projection (Column Parallel)",
        "attention_out": "C_proj projection (Row Parallel)",
        "mlp": "C_fc projection (Column Parallel)",
        "mlp_out": "C_proj projection (Row Parallel)",
        "models": ["GPT-2", "GPT-Neo"]
    },
    "moe_style": {
        "attention": "Separate Q, K, V projections (Column Parallel)",
        "attention_out": "O projection (Row Parallel)",
        "moe": "Router (top-k) + Sharded Experts (Expert Parallelism)",
        "models": ["DeepSeek-V2", "DeepSeek-V3", "Mixtral-8x7B", "Qwen2MoE"]
    },
}


def print_supported_models():
    """Print all supported models"""
    print("=" * 70)
    print("NPU Tensor Parallelism - Supported Models")
    print("=" * 70)

    for family, info in SUPPORTED_MODELS.items():
        status = "✅ TESTED" if info["tested"] else "⚠️  SHOULD WORK"
        print(f"\n{family.upper()} - {status}")
        print(f"  Architecture: {info['architecture']}")
        print(f"  Notes: {info['notes']}")
        print(f"  Examples:")
        for model in info["models"][:3]:
            print(f"    - {model}")


def print_usage_examples():
    """Print usage examples"""
    print("\n" + "=" * 70)
    print("Usage Examples")
    print("=" * 70)

    examples = [
        ("Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B-Instruct", 8),
        ("Llama-3-8B", "meta-llama/Meta-Llama-3-8B", 4),
        ("Mistral-7B", "mistralai/Mistral-7B-v0.3", 2),
        ("Gemma-2-9B", "google/gemma-2-9b", 4),
    ]

    for name, model_path, tp_size in examples:
        print(f"\n# {name} ({tp_size}-way TP)")
        print(f'python train_tp_custom.py \\')
        print(f'    --model_path "{model_path}" \\')
        print(f'    --tp_size {tp_size} \\')
        print(f'    --batch_size 2 \\')
        print(f'    --max_length 256')


def check_model_compatibility(model_name: str) -> dict:
    """
    Check if a model is compatible with NPU TP

    Args:
        model_name: Model name or class name

    Returns:
        Dict with compatibility info
    """
    model_name_lower = model_name.lower()

    for family, info in SUPPORTED_MODELS.items():
        if family in model_name_lower:
            return {
                "compatible": True,
                "family": family,
                "architecture": info["architecture"],
                "tested": info["tested"],
                "notes": info["notes"]
            }

    # Check for GPT-2 style
    if "gpt2" in model_name_lower or "neo" in model_name_lower:
        return {
            "compatible": True,
            "family": "gpt2",
            "architecture": "GPT + Combined QKV",
            "tested": True,
            "notes": "GPT-2 style architecture"
        }

    return {
        "compatible": "unknown",
        "family": None,
        "architecture": "unknown",
        "tested": False,
        "notes": "Unknown model, may work if using GPT + SwiGLU pattern"
    }


if __name__ == "__main__":
    print_supported_models()
    print_usage_examples()

    # Example compatibility check
    print("\n" + "=" * 70)
    print("Compatibility Check Examples")
    print("=" * 70)

    test_models = [
        "Qwen2ForCausalLM",
        "LlamaForCausalLM",
        "MistralForCausalLM",
        "GPT2LMHeadModel",
        "SomeUnknownModel"
    ]

    for model in test_models:
        result = check_model_compatibility(model)
        print(f"\n{model}:")
        print(f"  Compatible: {result['compatible']}")
        print(f"  Architecture: {result['architecture']}")
        print(f"  Notes: {result['notes']}")

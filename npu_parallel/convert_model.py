"""
Convert HuggingFace Models to Tensor Parallelism

This module provides utilities to convert standard HuggingFace transformer models
to use tensor parallelism on NPU/HCCL.

Supported Architectures (all use GPT-style + SwiGLU pattern):
- Qwen/Qwen2/Qwen2.5
- Llama/Llama2/Llama3/Llama3.1/Llama3.2
- Mistral/Mixtral
- Gemma/Gemma2
- Phi-2/Phi-3
- Yi
- DeepSeek
- Baichuan2
- Qwen2MoE (expert routing not yet supported)
- GPT-2/GPT-Neo (older architecture with combined QKV)

Usage:
    from transformers import AutoModelForCausalLM
    from npu_parallel import convert_to_tp

    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    model = convert_to_tp(model, tp_size=4, rank=dist.get_rank())
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional

from .tp_layers import ColumnParallelLinear, RowParallelLinear
from .tp_attention import TPAttention, TPMLP
from .tp_moe import TPMoELayer, convert_moe_layer_to_tp, convert_deepseek_v2_moe_to_tp


def convert_deepseek_v2_attention_to_tp(
    attention: nn.Module,
    tp_size: int,
    rank: int,
) -> nn.Module:
    """
    Convert DeepSeek-V2 MLA (Multi-head Latent Attention) to TP

    DeepSeek-V2 uses compressed KV cache with MLA:
    - q_proj: Q projection (standard)
    - kv_a_proj_with_mqa: Compressed KV projection
    - kv_b_proj: KV decompression
    - o_proj: Output projection

    For simplicity, we only TP the output projection (row parallel)
    and keep Q, KV projections replicated.
    """
    # Output projection: Row parallel
    if hasattr(attention, 'o_proj'):
        o_proj = attention.o_proj
        in_features = o_proj.in_features
        out_features = o_proj.out_features
        weight_dtype = o_proj.weight.dtype
        device = o_proj.weight.device

        row_linear = RowParallelLinear(
            in_features,
            out_features,
            tp_size=tp_size,
            rank=rank,
            bias=False,
            input_is_parallel=False,  # Input is full size from attention
            dtype=weight_dtype,
            device=device,
        )
        # Copy weight (row parallel splits input dimension, which is intermediate_size)
        in_per_rank = in_features // tp_size
        start_idx = rank * in_per_rank
        end_idx = start_idx + in_per_rank
        row_linear.weight.data.copy_(o_proj.weight[:, start_idx:end_idx])

        attention.o_proj = row_linear

    return attention


def convert_deepseek_v2_block_to_tp(
    decoder_layer: nn.Module,
    tp_size: int,
    rank: int,
) -> nn.Module:
    """
    Convert a DeepSeek-V2 decoder block to TP

    DeepSeek-V2 block structure:
        - self_attn: DeepseekV2Attention (MLA)
        - mlp: DeepseekV2MLP or DeepseekV2MoE

    Args:
        decoder_layer: DeepseekV2DecoderLayer
        tp_size: Tensor parallel size
        rank: Current rank

    Returns:
        Modified decoder layer
    """
    # Convert attention (MLA) - only o_proj for now
    if hasattr(decoder_layer, 'self_attn'):
        decoder_layer.self_attn = convert_deepseek_v2_attention_to_tp(
            decoder_layer.self_attn,
            tp_size,
            rank
        )

    # Convert MLP/MoE
    if hasattr(decoder_layer, 'mlp'):
        mlp = decoder_layer.mlp
        mlp_type = type(mlp).__name__

        if 'DeepseekV2MoE' in mlp_type:
            # MoE layer - shard experts
            if tp_size > 1:
                # Get dimensions
                if hasattr(mlp, 'experts') and len(mlp.experts) > 0:
                    expert = mlp.experts[0]
                    hidden_size = expert.gate_proj.in_features
                    intermediate_size = expert.gate_proj.out_features

                decoder_layer.mlp = convert_deepseek_v2_moe_to_tp(
                    mlp, tp_size, rank,
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                )

        elif 'DeepseekV2MLP' in mlp_type:
            # Dense MLP - apply standard TP
            if tp_size > 1:
                # Gate projection: Column parallel
                if hasattr(mlp, 'gate_proj'):
                    gate_proj = mlp.gate_proj
                    mlp.gate_proj = convert_linear_to_tp(
                        gate_proj, tp_size, rank,
                        column=True, gather_output=False
                    )

                # Up projection: Column parallel
                if hasattr(mlp, 'up_proj'):
                    up_proj = mlp.up_proj
                    mlp.up_proj = convert_linear_to_tp(
                        up_proj, tp_size, rank,
                        column=True, gather_output=False
                    )

                # Down projection: Row parallel
                if hasattr(mlp, 'down_proj'):
                    down_proj = mlp.down_proj
                    mlp.down_proj = convert_linear_to_tp(
                        down_proj, tp_size, rank,
                        column=False, input_is_parallel=True
                    )

    return decoder_layer


def convert_deepseek_v2_model_to_tp(
    model: nn.Module,
    tp_size: int,
    rank: int,
) -> nn.Module:
    """
    Convert a DeepSeek-V2 model to tensor parallelism

    Args:
        model: DeepseekV2ForCausalLM
        tp_size: Tensor parallel size
        rank: Current rank

    Returns:
        TP-converted model
    """
    # Convert each decoder layer
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        for i, layer in enumerate(model.model.layers):
            model.model.layers[i] = convert_deepseek_v2_block_to_tp(layer, tp_size, rank)

    # LM head: column parallel with gather
    if hasattr(model, "lm_head"):
        lm_head = model.lm_head
        model.lm_head = convert_linear_to_tp(
            lm_head, tp_size, rank,
            column=True, gather_output=True
        )

    return model



def convert_linear_to_tp(
    linear: nn.Linear,
    tp_size: int,
    rank: int,
    column: bool = True,
    gather_output: bool = True,
    input_is_parallel: bool = False,
) -> nn.Module:
    """
    Convert a standard nn.Linear to TP version

    Args:
        linear: Original linear layer
        tp_size: Tensor parallel size
        rank: Current rank (0 to tp_size-1)
        column: True for column parallel, False for row parallel
        gather_output: For column parallel, whether to all-gather output
        input_is_parallel: For row parallel, whether input is already split

    Returns:
        TP version of the linear layer with copied weights
    """
    in_features = linear.in_features
    out_features = linear.out_features
    bias = linear.bias is not None
    weight_dtype = linear.weight.dtype
    device = linear.weight.device

    if column:
        tp_layer = ColumnParallelLinear(
            in_features,
            out_features,
            tp_size=tp_size,
            rank=rank,
            bias=bias,
            gather_output=gather_output,
            dtype=weight_dtype,
            device=device,
        )
        # PyTorch weight shape: [out_features, in_features]
        # Column parallel splits output dimension, so slice rows
        out_per_rank = out_features // tp_size
        start_idx = rank * out_per_rank
        end_idx = start_idx + out_per_rank
        tp_layer.weight.data.copy_(linear.weight[start_idx:end_idx, :])
        if bias:
            tp_layer.bias.data.copy_(linear.bias[start_idx:end_idx])
    else:
        tp_layer = RowParallelLinear(
            in_features,
            out_features,
            tp_size=tp_size,
            rank=rank,
            bias=bias,
            input_is_parallel=input_is_parallel,
            dtype=weight_dtype,
            device=device,
        )
        # PyTorch weight shape: [out_features, in_features]
        # Row parallel splits input dimension, so slice columns
        in_per_rank = in_features // tp_size
        start_idx = rank * in_per_rank
        end_idx = start_idx + in_per_rank
        tp_layer.weight.data.copy_(linear.weight[:, start_idx:end_idx])
        if bias and rank == tp_size - 1:
            tp_layer.bias.data.copy_(linear.bias)

    return tp_layer


def replace_layer(module: nn.Module, target_name: str, new_layer: nn.Module):
    """
    Replace a layer in a module by name

    Args:
        module: Parent module containing the layer
        target_name: Dot-separated path to the layer (e.g., "model.layers.0.self_attn")
        new_layer: New layer to replace with
    """
    parts = target_name.split(".")
    parent = module

    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)

    last_part = parts[-1]
    if last_part.isdigit():
        parent[int(last_part)] = new_layer
    else:
        setattr(parent, last_part, new_layer)


def convert_qwen_block_to_tp(
    decoder_layer: nn.Module,
    tp_size: int,
    rank: int,
) -> nn.Module:
    """
    Convert a Qwen2/Qwen2.5 decoder block to TP

    Qwen decoder block structure:
        - self_attn:
            - q_proj, k_proj, v_proj: Column parallel (with gather for Q)
            - o_proj: Row parallel
        - mlp:
            - For dense models: gate_proj, up_proj, down_proj (SwiGLU)
            - For MoE models: experts, gate (DeepSeek-V2 style)

    Args:
        decoder_layer: QwenDecoderLayer or similar
        tp_size: Tensor parallel size
        rank: Current rank

    Returns:
        Modified decoder layer
    """
    # Convert attention projections
    # For Qwen with separate Q, K, V projections using HuggingFace SDPA:
    # We need to gather all outputs so the standard attention code works
    # The parallelism comes from weight splitting, not activation splitting
    q_proj = decoder_layer.self_attn.q_proj
    decoder_layer.self_attn.q_proj = convert_linear_to_tp(q_proj, tp_size, rank, column=True, gather_output=True)

    k_proj = decoder_layer.self_attn.k_proj
    decoder_layer.self_attn.k_proj = convert_linear_to_tp(k_proj, tp_size, rank, column=True, gather_output=True)

    v_proj = decoder_layer.self_attn.v_proj
    decoder_layer.self_attn.v_proj = convert_linear_to_tp(v_proj, tp_size, rank, column=True, gather_output=True)

    # Output projection: Row parallel
    # Since Q, K, V are gathered, attention output is full size
    # Set input_is_parallel=False so RowParallelLinear will split the input
    o_proj = decoder_layer.self_attn.o_proj
    decoder_layer.self_attn.o_proj = convert_linear_to_tp(o_proj, tp_size, rank, column=False, input_is_parallel=False)

    # Convert MLP - Check if this is a MoE layer
    if hasattr(decoder_layer, 'mlp'):
        mlp = decoder_layer.mlp

        # Check for MoE structure (DeepSeek-V2, Mixtral, Qwen2MoE patterns)
        is_moe = False

        # Pattern 1: DeepSeek-V2 style (DeepseekV2MoE with experts, gate, shared_experts)
        # Check class name for more specific detection
        mlp_class_name = type(mlp).__name__
        if 'DeepseekV2MoE' in mlp_class_name or (hasattr(mlp, 'experts') and hasattr(mlp, 'gate')):
            is_moe = True
            if tp_size > 1:
                # DeepSeek-V2 MoE structure:
                # - mlp.experts: ModuleList of DeepseekV2MLP (64 experts)
                # - mlp.gate: MoEGate (router)
                # - mlp.shared_experts: DeepseekV2MLP (shared experts, optional)

                # Get dimensions from experts
                if hasattr(mlp, 'experts') and len(mlp.experts) > 0:
                    expert = mlp.experts[0]
                    # DeepseekV2MLP has gate_proj, up_proj, down_proj
                    if hasattr(expert, 'gate_proj'):
                        hidden_size = expert.gate_proj.in_features
                        intermediate_size = expert.gate_proj.out_features

                # For DeepSeek-V2, we shard experts across TP ranks
                # Each rank gets num_experts // tp_size experts
                from .tp_moe import convert_deepseek_v2_moe_to_tp
                decoder_layer.mlp = convert_deepseek_v2_moe_to_tp(
                    mlp, tp_size, rank,
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                )

        # Pattern 2: Mixtral style (block_sparse_moe.experts, block_sparse_moe.gate)
        elif hasattr(mlp, 'block_sparse_moe'):
            is_moe = True
            block_moe = mlp.block_sparse_moe
            if hasattr(block_moe, 'experts') and hasattr(block_moe, 'gate'):
                # Access through mlp.block_sparse_moe
                if tp_size > 1:
                    hidden_size = block_moe.experts[0].w1.in_features
                    intermediate_size = block_moe.experts[0].w1.out_features

                    # Convert the inner MoE layer
                    converted_moe = convert_moe_layer_to_tp(
                        block_moe, tp_size, rank,
                        hidden_size=hidden_size,
                        intermediate_size=intermediate_size,
                    )
                    mlp.block_sparse_moe = converted_moe

        # Pattern 3: Qwen2MoE style (mlp.num_experts, mlp.experts)
        elif hasattr(mlp, 'num_experts') and hasattr(mlp, 'experts'):
            is_moe = True
            if tp_size > 1:
                hidden_size = mlp.experts[0].gate_proj.in_features if hasattr(mlp.experts[0], 'gate_proj') else 4096
                intermediate_size = mlp.experts[0].gate_proj.out_features if hasattr(mlp.experts[0], 'gate_proj') else 1536

                decoder_layer.mlp = convert_moe_layer_to_tp(
                    mlp, tp_size, rank,
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                )

        # If not MoE, convert as standard dense MLP
        if not is_moe:
            # Gate projection: Column parallel
            if hasattr(decoder_layer.mlp, "gate_proj"):
                gate_proj = decoder_layer.mlp.gate_proj
                decoder_layer.mlp.gate_proj = convert_linear_to_tp(gate_proj, tp_size, rank, column=True, gather_output=False)

            # Up projection: Column parallel
            if hasattr(decoder_layer.mlp, "up_proj"):
                up_proj = decoder_layer.mlp.up_proj
                decoder_layer.mlp.up_proj = convert_linear_to_tp(up_proj, tp_size, rank, column=True, gather_output=False)

            # Down projection: Row parallel
            if hasattr(decoder_layer.mlp, "down_proj"):
                down_proj = decoder_layer.mlp.down_proj
                decoder_layer.mlp.down_proj = convert_linear_to_tp(down_proj, tp_size, rank, column=False, input_is_parallel=True)

    return decoder_layer


def convert_qwen_model_to_tp(
    model: nn.Module,
    tp_size: int,
    rank: int,
) -> nn.Module:
    """
    Convert a Qwen2/Qwen2.5 model to tensor parallelism

    Args:
        model: Qwen2ForCausalLM or similar
        tp_size: Tensor parallel size
        rank: Current rank

    Returns:
        TP-converted model
    """
    # Check if model is on correct device
    device = next(model.parameters()).device

    # Convert each decoder layer
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        for i, layer in enumerate(model.model.layers):
            model.model.layers[i] = convert_qwen_block_to_tp(layer, tp_size, rank)

    # Handle embedding layer (row parallel)
    if hasattr(model.model, "embed_tokens"):
        embed_tokens = model.model.embed_tokens
        # Embedding: split vocab across TP ranks (row parallel on first dimension)
        vocab_size = embed_tokens.num_embeddings
        embedding_dim = embed_tokens.embedding_dim
        vocab_per_rank = vocab_size // tp_size

        # Note: For training, we typically keep embedding on all ranks
        # For inference, we can shard it. Keeping it simple for now.

    # Handle LM head (same as embedding, usually tied)
    if hasattr(model, "lm_head"):
        lm_head = model.lm_head
        # LM head: column parallel with gather for loss computation
        model.lm_head = convert_linear_to_tp(
            lm_head,
            tp_size=tp_size,
            rank=rank,
            column=True,
            gather_output=True,  # Gather for full logits in loss computation
        )

    return model


def convert_gpt2_block_to_tp(
    decoder_layer: nn.Module,
    tp_size: int,
    rank: int,
) -> nn.Module:
    """
    Convert a GPT-2 decoder block to TP

    GPT-2 decoder block structure:
        - attn:
            - c_attn (combined QKV): Column parallel
            - c_proj (output): Row parallel
        - mlp:
            - c_fc (up): Column parallel
            - c_proj (down): Row parallel

    Args:
        decoder_layer: GPT2Block
        tp_size: Tensor parallel size
        rank: Current rank

    Returns:
        Modified decoder layer
    """
    # Convert QKV projection (combined): Column parallel
    if hasattr(decoder_layer.attn, "c_attn"):
        c_attn = decoder_layer.attn.c_attn
        decoder_layer.attn.c_attn = convert_linear_to_tp(c_attn, tp_size, rank, column=True, gather_output=False)

    # Output projection: Row parallel
    if hasattr(decoder_layer.attn, "c_proj"):
        c_proj = decoder_layer.attn.c_proj
        decoder_layer.attn.c_proj = convert_linear_to_tp(c_proj, tp_size, rank, column=False, input_is_parallel=True)

    # Convert MLP
    # Up projection: Column parallel
    if hasattr(decoder_layer.mlp, "c_fc"):
        c_fc = decoder_layer.mlp.c_fc
        decoder_layer.mlp.c_fc = convert_linear_to_tp(c_fc, tp_size, rank, column=True, gather_output=False)

    # Down projection: Row parallel
    if hasattr(decoder_layer.mlp, "c_proj"):
        c_proj = decoder_layer.mlp.c_proj
        decoder_layer.mlp.c_proj = convert_linear_to_tp(c_proj, tp_size, rank, column=False, input_is_parallel=True)

    return decoder_layer


def convert_gpt2_model_to_tp(
    model: nn.Module,
    tp_size: int,
    rank: int,
) -> nn.Module:
    """
    Convert a GPT-2 model to tensor parallelism

    Args:
        model: GPT2LMHeadModel
        tp_size: Tensor parallel size
        rank: Current rank

    Returns:
        TP-converted model
    """
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        for i, block in enumerate(model.transformer.h):
            model.transformer.h[i] = convert_gpt2_block_to_tp(block, tp_size, rank)

    # LM head: column parallel
    if hasattr(model, "lm_head"):
        lm_head = model.lm_head
        model.lm_head = convert_linear_to_tp(lm_head, tp_size, rank, column=True, gather_output=True)

    return model


def convert_to_tp(
    model: nn.Module,
    tp_size: int,
    rank: int,
) -> nn.Module:
    """
    Convert a HuggingFace model to tensor parallelism

    Automatically detects model type and applies appropriate conversion.
    Supports both dense models and MoE models (DeepSeek-V2, Mixtral, Qwen2MoE).

    Args:
        model: HuggingFace model (AutoModelForCausalLM)
        tp_size: Tensor parallel size
        rank: Current rank (0 to tp_size-1)

    Returns:
        TP-converted model

    Example:
        >>> import torch.distributed as dist
        >>> from transformers import AutoModelForCausalLM
        >>> from npu_parallel import convert_to_tp
        >>>
        >>> dist.init_process_group(backend="hccl")
        >>> rank = dist.get_rank()
        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
        >>> model = convert_to_tp(model, tp_size=4, rank=rank)

    For MoE models:
        >>> model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-V2-Lite")
        >>> model = convert_to_tp(model, tp_size=4, rank=rank)
    """
    if tp_size <= 1:
        return model

    # Check if model is MoE
    is_moe = False
    model_type = model.__class__.__name__.lower()

    # Detect MoE by checking model structure
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        if len(model.model.layers) > 0:
            first_layer = model.model.layers[0]
            if hasattr(first_layer, "mlp"):
                # DeepSeek-V2 MoE pattern
                if hasattr(first_layer.mlp, "experts") and hasattr(first_layer.mlp, "gate"):
                    is_moe = True
                # Mixtral MoE pattern
                elif hasattr(first_layer.mlp, "block_sparse_moe"):
                    is_moe = True
                # Qwen2MoE pattern
                elif hasattr(first_layer.mlp, "num_experts"):
                    is_moe = True

    # Print MoE detection info
    if rank == 0 and is_moe:
        print(f"[MoE Detection] Model {model_type} has MoE architecture")
        print(f"[MoE Detection] TP will be applied to both attention and MoE layers")

    # Create TP process group if world_size > tp_size
    # This allows hybrid TP+DDP training
    try:
        world_size = dist.get_world_size()
    except (ValueError, RuntimeError):
        # Distributed not initialized, use tp_size as world_size
        world_size = tp_size

    tp_group = None
    if world_size > tp_size:
        # Create a new process group for TP ranks
        # Ranks 0..tp_size-1 form group 0, ranks tp_size..2*tp_size-1 form group 1, etc.
        tp_group_rank = rank // tp_size  # Which TP group this rank belongs to
        tp_ranks = [tp_group_rank * tp_size + i for i in range(tp_size)]

        # Get the current rank's position within its TP group
        tp_rank_in_group = rank % tp_size

        # Create the TP process group
        try:
            tp_group = dist.new_group(ranks=tp_ranks, backend="hccl")
        except Exception as e:
            # If new_group fails, fall back to using world group
            # This can happen in some distributed setups
            tp_group = None
    else:
        tp_rank_in_group = rank

    # Store TP group info in model for later use
    model._tp_group = tp_group
    model._tp_size = tp_size
    model._tp_rank = tp_rank_in_group if world_size > tp_size else rank
    model._is_moe = is_moe

    # Detect model type and apply appropriate conversion
    model_type = model.__class__.__name__.lower()

    # Check for DeepSeek-V2 specifically (has MLA attention + MoE)
    if "deepseek" in model_type and "v2" in model_type:
        if rank == 0:
            print(f"[DeepSeek-V2 Detection] Using specialized DeepSeek-V2 TP conversion")
        return convert_deepseek_v2_model_to_tp(model, tp_size, rank)

    # Modern LLMs with GPT-style attention + SwiGLU MLP or MoE
    # All use: separate Q,K,V projections + gate/up/down MLP structure
    qwen_style_models = ["qwen", "llama", "mistral", "mixtral", "gemma",
                         "phi", "yi", "baichuan", "internlm"]

    if any(name in model_type for name in qwen_style_models):
        return convert_qwen_model_to_tp(model, tp_size, rank)
    elif "gpt2" in model_type or "neo" in model_type:
        return convert_gpt2_model_to_tp(model, tp_size, rank)
    else:
        # Fallback: auto-detect architecture pattern
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            # Qwen/Llama/Mistral style: model.layers[i]
            for i, layer in enumerate(model.model.layers):
                model.model.layers[i] = convert_qwen_block_to_tp(layer, tp_size, rank)
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            # GPT-2 style: transformer.h[i]
            for i, block in enumerate(model.transformer.h):
                model.transformer.h[i] = convert_gpt2_block_to_tp(block, tp_size, rank)
        elif hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
            # GPT-NeoX style
            for i, layer in enumerate(model.gpt_neox.layers):
                model.gpt_neox.layers[i] = convert_qwen_block_to_tp(layer, tp_size, rank)

        # LM head: column parallel
        if hasattr(model, "lm_head"):
            lm_head = model.lm_head
            model.lm_head = convert_linear_to_tp(lm_head, tp_size, rank, column=True, gather_output=True)

    return model


def sync_gradients_tp(model: nn.Module, tp_size: int):
    """
    Synchronize gradients across TP ranks

    After backward pass, gradients need to be synchronized across TP ranks
    before the optimizer step.

    Args:
        model: TP-converted model
        tp_size: Tensor parallel size
    """
    if tp_size <= 1:
        return

    for param in model.parameters():
        if param.grad is not None:
            grad = param.grad
            original_dtype = grad.dtype

            # Cast to float32 if bfloat16 for HCCL compatibility
            if original_dtype == torch.bfloat16:
                grad = grad.to(torch.float32)

            # All-reduce gradients and average
            dist.all_reduce(grad, op=dist.ReduceOp.SUM)

            # Cast back to original dtype if needed
            if original_dtype == torch.bfloat16:
                grad = grad.to(original_dtype)

            param.grad = grad / tp_size


def calculate_tp_memory(model: nn.Module, tp_size: int) -> dict:
    """
    Calculate memory usage per rank with tensor parallelism

    Args:
        model: Model to analyze
        tp_size: Tensor parallel size

    Returns:
        Dictionary with memory statistics
    """
    total_params = sum(p.numel() for p in model.parameters())
    params_per_rank = total_params // tp_size

    # Estimate memory (assuming fp16)
    bytes_per_param = 2  # fp16 = 2 bytes
    model_memory = params_per_rank * bytes_per_param

    # Estimate activations (rough estimate: 2x model memory)
    activation_memory = model_memory * 2

    total_memory = model_memory + activation_memory

    return {
        "total_params": total_params,
        "params_per_rank": params_per_rank,
        "model_memory_gb": model_memory / 1e9,
        "activation_memory_gb": activation_memory / 1e9,
        "total_memory_gb": total_memory / 1e9,
    }

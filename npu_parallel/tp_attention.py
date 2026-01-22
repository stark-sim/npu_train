"""
Tensor Parallelism Attention Implementation for NPU

Implements TP-compatible attention mechanism for transformer models.
Based on Megatron-LM patterns adapted for NPU/HCCL.

Key Components:
- TPQKVParallel: Query, Key, Value projections (column parallel)
- TPOutputParallel: Output projection (row parallel)
- TPAttention: Complete TP-aware attention module

TP Strategy for Attention:
    Q, K, V projections: Column Parallel (split output dimension)
        Input [B, S, H] → Q [B, S, H], K [B, S, H], V [B, S, H]
        Each rank: [B, S, H/tp] @ [H, H/tp] → [B, S, H/tp]
        All-gather: [B, S, H/tp] → [B, S, H]

    Output projection: Row Parallel (split input dimension)
        Input [B, S, H] → [B, S, H/tp] per rank
        Each rank: [B, S, H/tp] @ [H/tp, H] → [B, S, H]
        All-reduce: sum across ranks → [B, S, H]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from .tp_layers import ColumnParallelLinear, RowParallelLinear


class TPQKVParallel(nn.Module):
    """
    Combined Query-Key-Value projections with Tensor Parallelism

    Q, K, V projections are all column parallel - each rank computes
    a portion of the Q, K, V matrices.

    For GPT/Qwen style models:
        hidden_size = num_heads * head_dim
        qkv_out = hidden_size * 3
        Each rank outputs qkv_out / tp_size
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        tp_size: int = 1,
        rank: int = 0,
        bias: bool = True,
        dtype: torch.dtype = torch.float16,
        device: torch.device = None,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.tp_size = tp_size
        self.rank = rank

        # Head dimension (assuming all heads have same size)
        self.head_dim = hidden_size // num_heads

        # For TP, we split heads across ranks
        self.num_heads_per_rank = num_heads // tp_size

        # Output size is 3 * hidden_size (Q, K, V concatenated)
        self.out_features = hidden_size * 3
        self.out_per_rank = self.out_features // tp_size

        # Single projection for Q, K, V (more efficient than 3 separate projections)
        # Weight shape: [hidden_size, 3*hidden_size/tp_size]
        weight = torch.empty(
            hidden_size,
            self.out_per_rank,
            dtype=dtype,
            device=device,
        )
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        self.weight = nn.Parameter(weight)

        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_per_rank, dtype=dtype, device=device))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward: x @ W → output (partial QKV)

        Args:
            x: [batch, seq_len, hidden_size]

        Returns:
            qkv: [batch, seq_len, 3*hidden_size/tp_size]
        """
        return F.linear(x, self.weight, self.bias)


class TPOutputParallel(nn.Module):
    """
    Attention output projection with Tensor Parallelism (Row Parallel)

    The output projection takes the attention output and projects it back
    to the hidden dimension. This is row parallel since the input is
    already split across TP ranks.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        tp_size: int = 1,
        rank: int = 0,
        bias: bool = True,
        dtype: torch.dtype = torch.float16,
        device: torch.device = None,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.tp_size = tp_size
        self.rank = rank
        self.num_heads_per_rank = num_heads // tp_size

        # Output: each rank has hidden_size/tp_size inputs, hidden_size outputs
        in_per_rank = hidden_size // tp_size

        weight = torch.empty(
            in_per_rank,
            hidden_size,
            dtype=dtype,
            device=device,
        )
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        self.weight = nn.Parameter(weight)

        if bias:
            if rank == tp_size - 1:
                self.bias = nn.Parameter(torch.zeros(hidden_size, dtype=dtype, device=device))
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward: local computation → all-reduce

        Args:
            x: [batch, seq_len, hidden_size/tp_size]

        Returns:
            out: [batch, seq_len, hidden_size]
        """
        out = F.linear(x, self.weight, self.bias)

        if self.tp_size > 1:
            dist.all_reduce(out, op=dist.ReduceOp.SUM)

        return out


class TPAttention(nn.Module):
    """
    Complete Tensor Parallel Attention Module

    Implements multi-head attention with tensor parallelism:
    1. Q, K, V projections: Column parallel (split output heads)
    2. Scaled dot-product attention: Local computation per rank
    3. Output projection: Row parallel

    Memory Layout:
        For tp_size=4, num_heads=32:
        - Each rank handles 8 heads
        - QKV projection output: [B, S, 3*hidden_size/4]
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        tp_size: int = 1,
        rank: int = 0,
        max_position_embeddings: int = 8192,
        rope_base: int = 10000,
        rope_theta: float = 1.0,
        use_rope: bool = True,
        dtype: torch.dtype = torch.float16,
        device: torch.device = None,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.tp_size = tp_size
        self.rank = rank
        self.head_dim = hidden_size // num_heads
        self.num_heads_per_rank = num_heads // tp_size
        self.use_rope = use_rope

        # QKV projection (column parallel)
        self.qkv_proj = ColumnParallelLinear(
            hidden_size,
            hidden_size * 3,
            tp_size=tp_size,
            rank=rank,
            bias=False,
            gather_output=False,  # Keep partial output for TP attention
            dtype=dtype,
            device=device,
        )

        # Output projection (row parallel)
        self.o_proj = RowParallelLinear(
            hidden_size,
            hidden_size,
            tp_size=tp_size,
            rank=rank,
            bias=False,
            input_is_parallel=True,  # Input is already split from TP attention
            dtype=dtype,
            device=device,
        )

        # For models with bias
        if hasattr(self.qkv_proj, "bias"):
            nn.init.zeros_(self.qkv_proj.bias)

        # Rotary position embedding parameters
        if use_rope:
            self.rope_base = rope_base
            self.rope_theta = rope_theta
            self._init_rope(max_position_embeddings)

    def _init_rope(self, max_position_embeddings):
        """Initialize rotary position embedding frequencies"""
        inv_freq = 1.0 / (self.rope_base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute frequencies
        t = torch.arange(max_position_embeddings, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        self.register_buffer("freqs", freqs, persistent=False)

    def _apply_rope(self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor) -> tuple:
        """
        Apply rotary position embeddings

        Args:
            q, k: [batch, seq_len, num_heads, head_dim]
            positions: [batch, seq_len] or [seq_len]

        Returns:
            q, k with RoPE applied
        """
        # Get rotary embeddings for positions
        if positions.dim() == 1:
            freqs = self.freqs[positions]
        else:
            freqs = self.freqs[positions].unsqueeze(1)  # Add seq_len dim

        # Apply rotary embeddings
        freqs = freqs.view(*freqs.shape[:-1], self.head_dim // 2, 2)
        cos = freqs[..., 0]
        sin = freqs[..., 1]

        # Rotate q and k
        q_shape = q.shape
        q = q.view(q_shape[0], q_shape[1], q_shape[2], -1, 2)
        q_rot = torch.stack([-q[..., 1], q[..., 0]], dim=-1)
        q = (q * cos.unsqueeze(2).unsqueeze(3) + q_rot * sin.unsqueeze(2).unsqueeze(3)).view(q_shape)

        k_shape = k.shape
        k = k.view(k_shape[0], k_shape[1], k_shape[2], -1, 2)
        k_rot = torch.stack([-k[..., 1], k[..., 0]], dim=-1)
        k = (k * cos.unsqueeze(2).unsqueeze(3) + k_rot * sin.unsqueeze(2).unsqueeze(3)).view(k_shape)

        return q, k

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        positions: torch.Tensor = None,
    ) -> tuple:
        """
        Forward pass

        Args:
            x: [batch, seq_len, hidden_size]
            attention_mask: [batch, 1, 1, seq_len] or [batch, seq_len, seq_len]
            positions: [batch, seq_len] or [seq_len] - position indices

        Returns:
            attn_output: [batch, seq_len, hidden_size]
            attn_weights: [batch, num_heads, seq_len, seq_len] (optional)
        """
        batch, seq_len, _ = x.shape

        # QKV projection: [B, S, 3*H/tp]
        qkv = self.qkv_proj(x)

        # Split Q, K, V
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        # Reshape for attention: [B, S, num_heads/tp, head_dim]
        q = q.view(batch, seq_len, self.num_heads_per_rank, self.head_dim)
        k = k.view(batch, seq_len, self.num_heads_per_rank, self.head_dim)
        v = v.view(batch, seq_len, self.num_heads_per_rank, self.head_dim)

        # Apply RoPE if enabled
        if self.use_rope and positions is not None:
            q, k = self._apply_rope(q, k, positions)

        # Compute attention scores
        # Transpose for efficiency: [B, num_heads, S, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Attention: Q @ K^T / sqrt(d)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply attention mask if provided
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        # Softmax
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # Attention output: attn_weights @ V
        attn_output = torch.matmul(attn_weights, v)

        # Transpose back: [B, S, num_heads, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch, seq_len, -1)  # [B, S, H/tp]

        # Output projection (row parallel)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


class TPMLP(nn.Module):
    """
    Tensor Parallel MLP Layer (Feed-Forward Network)

    Pattern: Column Parallel → Activation → Row Parallel

    For GPT/Qwen style models:
        up_proj + gate_proj: Column parallel
        down_proj: Row parallel
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        tp_size: int = 1,
        rank: int = 0,
        activation: str = "swiglu",
        dtype: torch.dtype = torch.float16,
        device: torch.device = None,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.tp_size = tp_size
        self.rank = rank
        self.activation = activation

        # Gate projection (column parallel)
        self.gate_proj = ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            tp_size=tp_size,
            rank=rank,
            bias=False,
            gather_output=False,
            dtype=dtype,
            device=device,
        )

        # Up projection (column parallel)
        self.up_proj = ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            tp_size=tp_size,
            rank=rank,
            bias=False,
            gather_output=False,
            dtype=dtype,
            device=device,
        )

        # Down projection (row parallel)
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            tp_size=tp_size,
            rank=rank,
            bias=False,
            input_is_parallel=True,
            dtype=dtype,
            device=device,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward: SiGLU pattern

        Args:
            x: [batch, seq_len, hidden_size]

        Returns:
            out: [batch, seq_len, hidden_size]
        """
        # Gate: [B, S, intermediate_size/tp]
        gate = self.gate_proj(x)

        # Up: [B, S, intermediate_size/tp]
        up = self.up_proj(x)

        # SiGLU activation
        if self.activation == "swiglu":
            activated = F.silu(gate) * up
        elif self.activation == "gelu":
            activated = F.gelu(gate) * up
        else:
            activated = F.relu(gate) * up

        # Down projection: [B, S, hidden_size]
        out = self.down_proj(activated)

        return out

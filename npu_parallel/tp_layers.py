"""
Tensor Parallelism Core Layer Implementations for NPU

Based on Megatron-LM TP patterns adapted for NPU/HCCL.

Key Concepts:
- Column Parallel: Split weight by columns, each rank computes partial output, then all-gather
- Row Parallel: Split weight by rows, each rank computes on partial input, then all-reduce

Communication Pattern:
    Column Parallel (e.g., Q projection):
        Input [B, S, H] → Broadcast → [B, S, H/tp] @ [H/tp, H] → AllGather → [B, S, H]

    Row Parallel (e.g., Output projection):
        Input [B, S, H] → Split → [B, S, H/tp] @ [H/tp, H] → AllReduce → [B, S, H]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class AllGatherFromTensor(torch.autograd.Function):
    """
    Custom autograd function for all_gather that works with NPU/HCCL.

    Forward: all_gather the input tensor across all ranks
    Backward: scatter the gradient back to each rank
    """

    @staticmethod
    def forward(ctx, input, group, tp_size):
        ctx.group = group
        ctx.tp_size = tp_size  # Use provided tp_size instead of get_world_size

        input_size = input.numel()
        output = torch.empty(
            input_size * ctx.tp_size,
            dtype=input.dtype,
            device=input.device
        )

        dist.all_gather_into_tensor(output, input, group=group)

        # Save original shape for backward
        ctx.input_shape = input.shape

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Split gradient evenly among ranks
        tp_size = ctx.tp_size
        grad_size = grad_output.numel() // tp_size
        rank = dist.get_rank(group=ctx.group)

        # Each rank takes its portion
        start_idx = rank * grad_size
        end_idx = start_idx + grad_size
        grad_input = grad_output[start_idx:end_idx]

        # Reshape to original input shape
        grad_input = grad_input.view(ctx.input_shape)

        return grad_input, None, None


def all_gather_forward(input_tensor, group=None, tp_size=None):
    """
    Wrapper for AllGatherFromTensor autograd function.

    Args:
        input_tensor: Input tensor to gather
        group: Process group (uses WORLD if None)
        tp_size: Tensor parallel size (uses group world size if None)

    Returns:
        Gathered tensor from all ranks
    """
    if group is None:
        group = dist.group.WORLD
    if tp_size is None:
        tp_size = dist.get_world_size(group=group)

    return AllGatherFromTensor.apply(input_tensor, group, tp_size)


class ColumnParallelLinear(nn.Module):
    """
    Column Parallel Linear Layer

    Splits the weight matrix column-wise across TP ranks.
    Each rank holds weight: [in_features, out_features / tp_size]

    Forward: x @ W_local → all_gather → full output

    Example:
        For tp_size=4, a Linear(4096, 4096) becomes:
        - Each rank: Linear(4096, 1024)
        - After forward: all_gather combines to [B, S, 4096]
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tp_size: int = 1,
        rank: int = 0,
        bias: bool = True,
        gather_output: bool = True,
        dtype: torch.dtype = torch.float16,
        device: torch.device = None,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.tp_size = tp_size
        self.rank = rank
        self.gather_output = gather_output

        # Validate dimensions
        assert out_features % tp_size == 0, (
            f"out_features ({out_features}) must be divisible by tp_size ({tp_size})"
        )

        self.out_per_rank = out_features // tp_size

        # Initialize weight with standard initialization
        # PyTorch F.linear expects weight shape: [out_features, in_features]
        weight = torch.empty(self.out_per_rank, in_features, dtype=dtype, device=device)
        # Kaiming initialization
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        self.weight = nn.Parameter(weight)

        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_per_rank, dtype=dtype, device=device))
        else:
            self.register_parameter("bias", None)

        # Process group for TP operations
        self.tp_group = None  # Will be set during initialization

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with column parallelism

        Args:
            x: Input tensor [batch, seq_len, in_features]

        Returns:
            Output tensor [batch, seq_len, out_features] (if gather_output=True)
                     [batch, seq_len, out_features/tp_size] (if gather_output=False)
        """
        # Local computation: [B, S, in_features] @ [in_features, out/tp] = [B, S, out/tp]
        out = F.linear(x, self.weight, self.bias)

        if not self.gather_output:
            return out

        # All-gather to collect outputs from all ranks
        # out shape: [B, S, out_features/tp_size]
        # out_all shape: [B, S, out_features]
        if self.tp_size > 1:
            batch, seq_len, _ = x.shape

            # Flatten and make contiguous
            out_flat = out.contiguous().view(-1)

            # Use the process group if set, otherwise use default
            group = self.tp_group if self.tp_group is not None else dist.group.WORLD

            # Perform all_gather with custom autograd, passing tp_size explicitly
            gathered_tensor = all_gather_forward(out_flat, group, self.tp_size)

            # Reshape to [batch, seq_len, out_features]
            # gathered_tensor: [batch * seq_len * out_features]
            out_all = gathered_tensor.view(batch, seq_len, self.out_features)

            return out_all

        return out

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, " \
               f"tp_size={self.tp_size}, gather_output={self.gather_output}"


class RowParallelLinear(nn.Module):
    """
    Row Parallel Linear Layer

    Splits the weight matrix row-wise across TP ranks.
    Each rank holds weight: [in_features / tp_size, out_features]

    Forward: split input → x @ W_local → all_reduce → full output

    Example:
        For tp_size=4, a Linear(4096, 4096) becomes:
        - Each rank: Linear(1024, 4096)
        - Input split: [B, S, 4096] → [B, S, 1024] per rank
        - After forward: all_reduce sums results
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tp_size: int = 1,
        rank: int = 0,
        bias: bool = True,
        input_is_parallel: bool = False,
        dtype: torch.dtype = torch.float16,
        device: torch.device = None,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.tp_size = tp_size
        self.rank = rank
        self.input_is_parallel = input_is_parallel

        # Validate dimensions
        assert in_features % tp_size == 0, (
            f"in_features ({in_features}) must be divisible by tp_size ({tp_size})"
        )

        self.in_per_rank = in_features // tp_size

        # Initialize weight with standard initialization
        # PyTorch F.linear expects weight shape: [out_features, in_features]
        # Here each rank has in_per_rank input features
        weight = torch.empty(out_features, self.in_per_rank, dtype=dtype, device=device)
        # Kaiming initialization
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        self.weight = nn.Parameter(weight)

        if bias:
            # Bias is only on the last rank (for consistency with original model)
            if rank == tp_size - 1:
                self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype, device=device))
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("bias", None)

        # Process group for TP operations
        self.tp_group = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with row parallelism

        Args:
            x: Input tensor [batch, seq_len, in_features]
                 If input_is_parallel=True: [batch, seq_len, in_features/tp_size]

        Returns:
            Output tensor [batch, seq_len, out_features]
        """
        # Split input if not already parallel
        if not self.input_is_parallel and self.tp_size > 1:
            # x: [B, S, in_features] → [B, S, in_features/tp_size]
            # Split along the last dimension
            chunk_size = self.in_per_rank
            start_idx = self.rank * chunk_size
            x = x[..., start_idx:start_idx + chunk_size]

        # Local computation: [B, S, in/tp] @ [in/tp, out] = [B, S, out]
        out = F.linear(x, self.weight, self.bias)

        # All-reduce to sum results from all ranks
        # Note: HCCL may not support bfloat16, so cast to float32 if needed
        if self.tp_size > 1:
            group = self.tp_group if self.tp_group is not None else dist.group.WORLD
            original_dtype = out.dtype
            if original_dtype == torch.bfloat16:
                out = out.to(torch.float32)
            dist.all_reduce(out, op=dist.ReduceOp.SUM, group=group)
            if original_dtype == torch.bfloat16:
                out = out.to(original_dtype)

        return out

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, " \
               f"tp_size={self.tp_size}, input_is_parallel={self.input_is_parallel}"


class TPProcessGroup:
    """
    Helper class to manage TP process groups

    This class creates and manages process groups for tensor parallelism.
    It handles both pure TP and hybrid TP+DDP scenarios.
    """

    def __init__(self, tp_size: int, dp_size: int = 1, backend: str = "hccl"):
        self.tp_size = tp_size
        self.dp_size = dp_size
        self.backend = backend

        # Get current world size and rank
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        # Validate configuration
        assert tp_size * dp_size == self.world_size, (
            f"tp_size ({tp_size}) * dp_size ({dp_size}) must equal world_size ({self.world_size})"
        )

        # Create process groups
        self._setup_groups()

    def _setup_groups(self):
        """Setup TP and DDP process groups"""
        # For hybrid TP+DDP:
        # TP group: ranks within the same data parallel group
        # DDP group: ranks with the same position in their TP groups

        # TP groups: each has dp_size groups of tp_size ranks
        for dp_idx in range(self.dp_size):
            tp_ranks = [dp_idx * self.tp_size + i for i in range(self.tp_size)]
            tp_group = dist.new_group(tp_ranks, backend=self.backend)

            # Save this rank's TP group
            if self.rank in tp_ranks:
                self.tp_group = tp_group
                self.tp_rank = tp_ranks.index(self.rank)

        # DDP groups: each has tp_size groups of dp_size ranks
        for tp_idx in range(self.tp_size):
            dp_ranks = [tp_idx + i * self.tp_size for i in range(self.dp_size)]
            dp_group = dist.new_group(dp_ranks, backend=self.backend)

            # Save this rank's DDP group
            if self.rank in dp_ranks:
                self.dp_group = dp_group
                self.dp_rank = dp_ranks.index(self.rank)

    def get_tp_group(self):
        """Get the tensor parallel process group for this rank"""
        return self.tp_group

    def get_dp_group(self):
        """Get the data parallel process group for this rank"""
        return self.dp_group

    def get_tp_rank(self):
        """Get the rank within the TP group"""
        return self.tp_rank

    def get_dp_rank(self):
        """Get the rank within the DDP group"""
        return self.dp_rank

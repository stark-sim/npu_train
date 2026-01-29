"""
Tensor Parallelism for Mixture-of-Experts (MoE) Models on NPU

This module implements TP support for MoE layers in models like:
- DeepSeek-V2/V3 (DeepSeekMoE)
- Mixtral-8x7B (MixtralSparseMoeBlock)
- Qwen2MoE

Key Concepts:
- Expert Parallelism (EP): Sharding experts across TP ranks
- Router: Top-k expert selection (replicated or parallel)
- AllToAll: Communication for token dispatch to expert-owning ranks

Architecture Pattern (DeepSeek-V2):
    Input → Router → Top-k selection → Dispatch to experts → Expert computation → Combine → Output

Each rank holds num_experts / tp_size experts.
Tokens are routed to the rank that owns their selected experts.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Optional, Tuple


class AllToAllFromTensor(torch.autograd.Function):
    """
    Custom autograd function for all_to_all that works with NPU/HCCL.

    This is used for MoE token dispatch:
    - Forward: Send tokens to expert-owning ranks, receive tokens for local experts
    - Backward: Send gradients back to source ranks

    AllToAll splits data and scatters to different ranks, then gathers data from all ranks.
    Each rank i sends split[i] to rank j, and receives split[j] from rank i.

    For MoE:
    - Each rank computes which experts it needs (based on router output)
    - Tokens are dispatched to ranks that own those experts
    - After expert computation, outputs are combined back
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor, output_split_sizes: list,
                input_split_sizes: list, group) -> torch.Tensor:
        """
        Forward pass: all_to_all communication

        Args:
            input: Input tensor to scatter
            output_split_sizes: List of sizes to expect from each rank
            input_split_sizes: List of sizes to send to each rank
            group: Process group for communication

        Returns:
            Gathered tensor from all ranks
        """
        ctx.group = group
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes
        ctx.input_shape = input.shape

        # Calculate total output size
        output_size = sum(output_split_sizes)

        # Prepare output tensor
        output = torch.empty(output_size, dtype=input.dtype, device=input.device)

        # Split input for sending
        input_splits = torch.split(input, input_split_sizes, dim=0)

        # Perform all_to_all
        # Note: HCCL all_to_all may have limitations, fallback implementation provided
        try:
            dist.all_to_all_single(
                output,
                input,
                output_split_sizes=output_split_sizes,
                input_split_sizes=input_split_sizes,
                group=group
            )
        except (TypeError, RuntimeError) as e:
            # Fallback: use all_to_all with list-based API
            # Some HCCL versions may not support the _single variant
            output_list = [torch.empty(size, dtype=input.dtype, device=input.device)
                          for size in output_split_sizes]
            dist.all_to_all(output_list, list(input_splits), group=group)
            output = torch.cat(output_list, dim=0)

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Backward pass: reverse the all_to_all operation"""
        # Split gradient according to output_split_sizes
        grad_splits = torch.split(grad_output, ctx.output_split_sizes, dim=0)

        # Prepare input gradient tensor
        grad_input = torch.empty(sum(ctx.input_split_sizes),
                                 dtype=grad_output.dtype,
                                 device=grad_output.device)

        # Reverse all_to_all
        try:
            dist.all_to_all_single(
                grad_input,
                grad_output,
                output_split_sizes=ctx.input_split_sizes,
                input_split_sizes=ctx.output_split_sizes,
                group=ctx.group
            )
        except (TypeError, RuntimeError):
            # Fallback
            grad_input_list = [torch.empty(size, dtype=grad_output.dtype, device=grad_output.device)
                              for size in ctx.input_split_sizes]
            dist.all_to_all(grad_input_list, list(grad_splits), group=ctx.group)
            grad_input = torch.cat(grad_input_list, dim=0)

        return grad_input.view(ctx.input_shape), None, None, None


def all_to_all_forward(input_tensor: torch.Tensor,
                       output_split_sizes: list,
                       input_split_sizes: list,
                       group=None) -> torch.Tensor:
    """
    Wrapper for AllToAllFromTensor autograd function.

    Args:
        input_tensor: Input tensor to scatter
        output_split_sizes: List of sizes to receive from each rank
        input_split_sizes: List of sizes to send to each rank
        group: Process group (uses WORLD if None)

    Returns:
        Gathered tensor from all ranks
    """
    if group is None:
        group = dist.group.WORLD

    return AllToAllFromTensor.apply(input_tensor, output_split_sizes,
                                    input_split_sizes, group)


class TPMoEExperts(nn.Module):
    """
    Sharded Expert MLPs with Tensor Parallelism

    Each rank holds num_experts // tp_size experts.
    Experts use column/row parallelism within their MLP structure.

    Structure per expert:
        gate_proj (column parallel) + up_proj (column parallel) → activation → down_proj (row parallel)
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        tp_size: int = 1,
        rank: int = 0,
        num_local_experts: int = None,
        dtype: torch.dtype = torch.float16,
        device: torch.device = None,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.tp_size = tp_size
        self.rank = rank

        # Calculate number of experts per rank
        if num_local_experts is None:
            self.num_local_experts = num_experts // tp_size
        else:
            self.num_local_experts = num_local_experts

        # Expert range for this rank
        self.expert_start_idx = rank * self.num_local_experts
        self.expert_end_idx = self.expert_start_idx + self.num_local_experts

        # Create experts as ModuleList
        # Each expert is a standard MLP (SwiGLU activation)
        self.experts = nn.ModuleList([
            self._create_expert_mlp(dtype, device)
            for _ in range(self.num_local_experts)
        ])

    def _create_expert_mlp(self, dtype, device):
        """Create a single expert MLP"""
        return nn.ModuleDict({
            'gate_proj': nn.Linear(
                self.hidden_size,
                self.intermediate_size,
                bias=False,
                dtype=dtype,
                device=device
            ),
            'up_proj': nn.Linear(
                self.hidden_size,
                self.intermediate_size,
                bias=False,
                dtype=dtype,
                device=device
            ),
            'down_proj': nn.Linear(
                self.intermediate_size,
                self.hidden_size,
                bias=False,
                dtype=dtype,
                device=device
            ),
        })

    def forward(self, x: torch.Tensor, expert_indices: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for local experts

        Args:
            x: Input tokens [batch_tokens, hidden_size]
            expert_indices: Expert indices for each token [batch_tokens]
                           Only indices in [expert_start_idx, expert_end_idx) are processed

        Returns:
            Expert outputs for processed tokens
        """
        batch_tokens = x.shape[0]
        device = x.device

        # Output buffer
        output = torch.zeros_like(x)

        # Process each local expert
        for local_idx, expert in enumerate(self.experts):
            global_expert_idx = self.expert_start_idx + local_idx

            # Find tokens assigned to this expert
            expert_mask_float = (expert_indices == global_expert_idx).float()

            # Check if any token is assigned using sum (NPU compatible)
            if expert_mask_float.sum() == 0:
                continue

            # Create weighted input (all tokens, but only assigned contribute)
            expert_input = x * expert_mask_float.unsqueeze(-1)

            # Expert MLP computation (SwiGLU)
            gate = F.silu(expert['gate_proj'](expert_input))
            up = expert['up_proj'](expert_input)
            expert_output = expert['down_proj'](gate * up)

            # Accumulate to output using the mask
            output = output + expert_output * expert_mask_float.unsqueeze(-1)

        return output

    def extra_repr(self):
        return (f"hidden_size={self.hidden_size}, "
                f"intermediate_size={self.intermediate_size}, "
                f"num_experts={self.num_experts}, "
                f"num_local_experts={self.num_local_experts}, "
                f"tp_size={self.tp_size}")


class TPMoERouter(nn.Module):
    """
    Top-k Router for MoE with Tensor Parallelism

    The router can be either:
    1. Replicated across all ranks (same weights)
    2. Column parallel (each rank computes partial logits)

    For simplicity, we start with replicated router.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int = 2,
        dtype: torch.dtype = torch.float16,
        device: torch.device = None,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k

        # Router gate: projects hidden states to expert logits
        self.gate = nn.Linear(
            hidden_size,
            num_experts,
            bias=False,
            dtype=dtype,
            device=device
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route tokens to experts

        Args:
            x: Input tokens [batch, seq_len, hidden_size]

        Returns:
            router_logits: Raw logits [batch, seq_len, num_experts]
            routing_weights: Normalized weights for top-k experts
            expert_indices: Indices of top-k experts
        """
        # Flatten input for routing
        batch, seq_len, hidden = x.shape
        x_flat = x.view(-1, hidden)

        # Compute router logits
        router_logits = self.gate(x_flat)  # [batch*seq, num_experts]

        # Top-k selection
        routing_weights, expert_indices = torch.topk(
            F.softmax(router_logits, dim=-1),
            k=self.top_k,
            dim=-1
        )

        # Normalize weights (they sum to k, not 1, due to topk)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        # Reshape back
        router_logits = router_logits.view(batch, seq_len, -1)

        return router_logits, routing_weights, expert_indices

    def compute_aux_loss(
        self,
        router_logits: torch.Tensor,
        expert_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute auxiliary load balancing loss

        This encourages uniform expert utilization:
        - Minimize variance of expert assignment
        - Penalize experts that are over/under utilized

        Args:
            router_logits: Router output logits [batch*seq, num_experts]
            expert_indices: Selected expert indices [batch*seq, top_k]

        Returns:
            Auxiliary loss scalar
        """
        batch_tokens, num_experts = router_logits.shape

        # Compute expert mask (one-hot)
        expert_mask = F.one_hot(expert_indices, num_experts)  # [batch*seq, top_k, num_experts]
        expert_mask = expert_mask.sum(dim=1)  # [batch*seq, num_experts]

        # Expert utilization (fraction of tokens assigned to each expert)
        expert_util = expert_mask.sum(dim=0) / batch_tokens  # [num_experts]

        # Target: uniform distribution
        target = torch.full_like(expert_util, 1.0 / num_experts)

        # Load balancing loss: mean squared error from uniform
        aux_loss = F.mse_loss(expert_util, target)

        return aux_loss

    def extra_repr(self):
        return f"hidden_size={self.hidden_size}, num_experts={self.num_experts}, top_k={self.top_k}"


class TPMoELayer(nn.Module):
    """
    Complete MoE Layer with Tensor Parallelism

    Combines router, expert sharding, and all-to-all communication.

    Flow:
        Input → Router (top-k) → Token dispatch → Local experts → Result combine → Output

    For TP:
        - Experts are sharded across ranks
        - Tokens are dispatched via all-to-all to expert-owning ranks
        - Results are combined back via all-to-all
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int = 2,
        tp_size: int = 1,
        rank: int = 0,
        num_local_experts: int = None,
        dtype: torch.dtype = torch.float16,
        device: torch.device = None,
        aux_loss_coef: float = 0.01,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.tp_size = tp_size
        self.rank = rank
        self.aux_loss_coef = aux_loss_coef

        # Router (replicated for now)
        self.router = TPMoERouter(
            hidden_size,
            num_experts,
            top_k=top_k,
            dtype=dtype,
            device=device,
        )

        # Local experts (sharded)
        self.experts = TPMoEExperts(
            hidden_size,
            intermediate_size,
            num_experts,
            tp_size=tp_size,
            rank=rank,
            num_local_experts=num_local_experts,
            dtype=dtype,
            device=device,
        )

        # Process group
        self.tp_group = None

        # Track last auxiliary loss
        self.last_aux_loss = torch.zeros(1, device=device)

    def set_tp_group(self, group):
        """Set the TP process group"""
        self.tp_group = group

    def forward(
        self,
        x: torch.Tensor,
        return_aux_loss: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through MoE layer

        Args:
            x: Input tensor [batch, seq_len, hidden_size]
            return_aux_loss: Whether to return auxiliary loss

        Returns:
            output: MoE layer output [batch, seq_len, hidden_size]
            aux_loss: Auxiliary load balancing loss (optional)
        """
        batch, seq_len, hidden = x.shape
        original_shape = x.shape

        # Flatten input
        x_flat = x.view(-1, hidden)

        # Step 1: Route tokens to experts
        router_logits, routing_weights, expert_indices = self.router(x)

        # Compute auxiliary loss
        aux_loss = self.router.compute_aux_loss(
            router_logits.view(-1, self.num_experts),
            expert_indices
        )
        self.last_aux_loss = aux_loss.detach()

        # Step 2: Determine token-to-rank assignment
        # For each token, which rank owns its assigned experts?
        if self.tp_size > 1:
            # Map expert indices to owning ranks
            expert_to_rank = expert_indices // (self.num_experts // self.tp_size)

            # Expand for top-k: each token has k assignments
            # expert_indices: [batch*seq, top_k]
            # routing_weights: [batch*seq, top_k]

            # Flatten for processing
            expert_indices_flat = expert_indices.view(-1)  # [batch*seq*top_k]
            routing_weights_flat = routing_weights.view(-1)  # [batch*seq*top_k]

            # Create mapping from token position to (expert, weight) pairs
            token_positions = torch.arange(
                x_flat.shape[0],
                device=x.device
            ).repeat_interleave(self.top_k)  # [batch*seq*top_k]

            # Step 3: Dispatch tokens to expert-owning ranks
            # For simplicity, we use a non-optimized approach:
            # All-rank all-to-all based on expert ownership

            # Determine which tokens go to which rank
            num_experts_per_rank = self.num_experts // self.tp_size

            # Count tokens per rank
            tokens_per_rank = []
            indices_per_rank = []
            weights_per_rank = []

            for r in range(self.tp_size):
                # Find experts owned by this rank
                rank_expert_start = r * num_experts_per_rank
                rank_expert_end = rank_expert_start + num_experts_per_rank

                # Mask for tokens assigned to this rank's experts
                rank_mask = (expert_indices_flat >= rank_expert_start) & \
                           (expert_indices_flat < rank_expert_end)

                tokens_for_rank = token_positions[rank_mask]
                experts_for_rank = expert_indices_flat[rank_mask]
                weights_for_rank = routing_weights_flat[rank_mask]

                tokens_per_rank.append(tokens_for_rank)
                indices_per_rank.append(experts_for_rank - rank_expert_start)  # Local expert index
                weights_per_rank.append(weights_for_rank)

            # Step 4: Prepare local expert computation
            # We need to gather tokens assigned to our local experts
            local_tokens_indices = tokens_per_rank[self.rank]
            local_expert_indices = indices_per_rank[self.rank]
            local_weights = weights_per_rank[self.rank]

            if local_tokens_indices.shape[0] > 0:
                # Gather tokens for local experts
                local_input = x_flat[local_tokens_indices]

                # Compute local expert outputs
                local_output = self.experts(local_input, local_expert_indices)

                # Weight by routing weights
                local_output = local_output * local_weights.unsqueeze(-1)
            else:
                # No tokens for local experts
                local_output = torch.zeros(
                    0, hidden,
                    dtype=x.dtype,
                    device=x.device
                )

            # Step 5: Gather outputs from all ranks
            # For simplicity, we do an all-gather of sizes then all-to-all
            # This is not optimal but works for correctness

            # Gather output sizes from all ranks
            output_sizes = [len(tokens_per_rank[r]) * hidden for r in range(self.tp_size)]
            input_sizes = [len(local_tokens_indices) * hidden for r in range(self.tp_size)]

            # Flatten local output
            local_output_flat = local_output.view(-1)

            # All-to-all to gather results
            gathered = all_to_all_forward(
                local_output_flat,
                output_split_sizes=[s // hidden for s in input_sizes],
                input_split_sizes=[s // hidden for s in output_sizes],
                group=self.tp_group if self.tp_group else dist.group.WORLD
            )

            # Reshape and scatter back to original positions
            # This requires careful index management
            # For now, we use a simpler scatter approach

            # Initialize output
            output_flat = torch.zeros_like(x_flat)

            # Scatter gathered outputs back
            gather_idx = 0
            for r in range(self.tp_size):
                count = len(tokens_per_rank[r])
                if count > 0:
                    # Get the portion of gathered output from rank r
                    start_idx = sum(input_sizes[:r]) // hidden
                    end_idx = start_idx + count
                    rank_output = gathered[start_idx:end_idx]

                    # Scatter back to original positions
                    original_positions = tokens_per_rank[r]
                    output_flat[original_positions] += rank_output

        else:
            # Single rank: no all-to-all needed
            # Direct expert computation
            expert_indices_flat = expert_indices.view(-1)  # [batch*seq*top_k]
            routing_weights_flat = routing_weights.view(-1)  # [batch*seq*top_k]

            token_positions = torch.arange(
                x_flat.shape[0],
                device=x.device
            ).repeat_interleave(self.top_k)

            output_flat = torch.zeros_like(x_flat)

            # Process each unique expert assignment
            unique_experts = torch.unique(expert_indices_flat)

            for expert_idx in unique_experts:
                # Find all tokens assigned to this expert
                expert_mask_float = (expert_indices_flat == expert_idx).float()

                # Check if any token is assigned using sum (NPU compatible)
                if expert_mask_float.sum() == 0:
                    continue

                # Use boolean mask for indexing
                expert_mask_bool = expert_mask_float.bool()
                expert_tokens = token_positions[expert_mask_bool]
                expert_weights = routing_weights_flat[expert_mask_bool]

                # Get expert input
                expert_input = x_flat[expert_tokens]

                # Compute expert output (using local expert if it's our range)
                if self.experts.expert_start_idx <= expert_idx < self.experts.expert_end_idx:
                    local_expert_idx = expert_idx - self.experts.expert_start_idx
                    expert_output = self.experts.experts[local_expert_idx]['gate_proj'](expert_input)
                    expert_output = F.silu(expert_output) * self.experts.experts[local_expert_idx]['up_proj'](expert_input)
                    expert_output = self.experts.experts[local_expert_idx]['down_proj'](expert_output)
                else:
                    # This expert is on another rank - in single rank mode, this shouldn't happen
                    continue

                # Accumulate weighted output
                output_flat[expert_tokens] += expert_output * expert_weights.unsqueeze(-1)

        # Reshape output
        output = output_flat.view(original_shape)

        if return_aux_loss:
            return output, aux_loss * self.aux_loss_coef
        return output

    def extra_repr(self):
        return (f"hidden_size={self.hidden_size}, "
                f"num_experts={self.num_experts}, "
                f"top_k={self.top_k}, "
                f"tp_size={self.tp_size}, "
                f"aux_loss_coef={self.aux_loss_coef}")


class TPDeepSeekMoE(nn.Module):
    """
    DeepSeek-V2 style MoE with TP support

    DeepSeekMoE has:
    - Shared experts (process all tokens)
    - Routed experts (process tokens via routing)

    This combines both and handles the specific DeepSeek architecture.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_shared_experts: int = 1,
        num_routed_experts: int = 64,
        top_k: int = 6,
        tp_size: int = 1,
        rank: int = 0,
        dtype: torch.dtype = torch.float16,
        device: torch.device = None,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.tp_size = tp_size
        self.rank = rank

        # Shared experts (always active)
        self.shared_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, intermediate_size, bias=False),
                nn.SiLU(),
                nn.Linear(intermediate_size, hidden_size, bias=False),
            )
            for _ in range(num_shared_experts)
        ])

        # Routed MoE layer
        self.routed_moe = TPMoELayer(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_routed_experts,
            top_k=top_k,
            tp_size=tp_size,
            rank=rank,
            dtype=dtype,
            device=device,
        )

    def forward(self, x: torch.Tensor, return_aux_loss: bool = False):
        """
        Forward pass for DeepSeek-style MoE

        Args:
            x: Input tensor [batch, seq_len, hidden_size]
            return_aux_loss: Whether to return auxiliary loss

        Returns:
            output: Combined output from shared + routed experts
            aux_loss: Auxiliary loss (optional)
        """
        # Shared experts
        shared_output = torch.zeros_like(x)
        for expert in self.shared_experts:
            shared_output += expert(x)

        # Routed experts
        routed_output = self.routed_moe(x, return_aux_loss=return_aux_loss)

        if return_aux_loss:
            routed_out, aux_loss = routed_output
            return shared_output + routed_out, aux_loss

        return shared_output + routed_output


def convert_moe_layer_to_tp(
    moe_layer: nn.Module,
    tp_size: int,
    rank: int,
    hidden_size: int,
    intermediate_size: int,
) -> nn.Module:
    """
    Convert a standard MoE layer to TP version

    This function handles different MoE architectures:
    - DeepSeek-V2: mlp.experts, mlp.gate
    - Mixtral: block_sparse_moe.experts, block_sparse_moe.gate

    Args:
        moe_layer: Original MoE layer
        tp_size: Tensor parallel size
        rank: Current rank
        hidden_size: Model hidden size
        intermediate_size: MLP intermediate size

    Returns:
        TP-converted MoE layer
    """
    # Detect MoE type and extract configuration
    if hasattr(moe_layer, 'num_experts'):
        num_experts = moe_layer.num_experts
    elif hasattr(moe_layer, 'num_local_experts'):
        num_experts = moe_layer.num_local_experts
    else:
        # Try to count from ModuleList
        if hasattr(moe_layer, 'experts') and isinstance(moe_layer.experts, nn.ModuleList):
            num_experts = len(moe_layer.experts)
        else:
            raise ValueError("Cannot determine number of experts in MoE layer")

    # Detect top_k
    if hasattr(moe_layer, 'top_k'):
        top_k = moe_layer.top_k
    else:
        top_k = 2  # Default

    device = next(moe_layer.parameters()).device
    dtype = next(moe_layer.parameters()).dtype

    # Create TP MoE layer
    tp_moe = TPMoELayer(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        top_k=top_k,
        tp_size=tp_size,
        rank=rank,
        dtype=dtype,
        device=device,
    )

    # Copy weights from original layer
    # Router weights
    if hasattr(moe_layer, 'gate'):
        tp_moe.router.gate.weight.data.copy_(moe_layer.gate.weight)
    elif hasattr(moe_layer, 'router') and hasattr(moe_layer.router, 'gate'):
        tp_moe.router.gate.weight.data.copy_(moe_layer.router.gate.weight)

    # Expert weights
    # Copy only the experts that belong to this rank
    num_local = num_experts // tp_size
    start_idx = rank * num_local
    end_idx = start_idx + num_local

    if hasattr(moe_layer, 'experts'):
        for i, expert in enumerate(moe_layer.experts):
            global_idx = start_idx + i
            if global_idx >= len(moe_layer.experts):
                break

            original_expert = moe_layer.experts[global_idx]

            # Copy gate_proj
            if hasattr(original_expert, 'gate_proj'):
                tp_moe.experts.experts[i]['gate_proj'].weight.data.copy_(
                    original_expert.gate_proj.weight
                )

            # Copy up_proj
            if hasattr(original_expert, 'up_proj'):
                tp_moe.experts.experts[i]['up_proj'].weight.data.copy_(
                    original_expert.up_proj.weight
                )

            # Copy down_proj
            if hasattr(original_expert, 'down_proj'):
                tp_moe.experts.experts[i]['down_proj'].weight.data.copy_(
                    original_expert.down_proj.weight
                )

    return tp_moe


def convert_deepseek_v2_moe_to_tp(
    moe_layer: nn.Module,
    tp_size: int,
    rank: int,
    hidden_size: int,
    intermediate_size: int,
) -> nn.Module:
    """
    Convert DeepSeek-V2 MoE layer to TP version

    DeepSeek-V2 MoE structure:
    - experts: ModuleList of DeepseekV2MLP (64 experts)
    - gate: MoEGate (router with weight shape [num_experts, hidden_size])
    - shared_experts: DeepseekV2MLP (optional)

    Args:
        moe_layer: DeepseekV2MoE layer
        tp_size: Tensor parallel size
        rank: Current rank
        hidden_size: Model hidden size
        intermediate_size: MLP intermediate size

    Returns:
        Modified DeepseekV2MoE layer with TP support
    """
    from .tp_layers import ColumnParallelLinear, RowParallelLinear

    # Get the number of experts
    num_experts = len(moe_layer.experts)
    num_local_experts = num_experts // tp_size

    print(f"[Rank {rank}] Converting DeepSeek-V2 MoE: {num_experts} experts -> {num_local_experts} local experts")

    # For TP on DeepSeek-V2:
    # 1. Shard experts: each rank keeps num_experts // tp_size experts
    # 2. Keep router replicated (for simplicity, could also be sharded)
    # 3. For each expert, apply column/row parallelism to its MLP

    # Calculate which experts belong to this rank
    expert_start_idx = rank * num_local_experts
    expert_end_idx = expert_start_idx + num_local_experts

    # Create new ModuleList with only local experts
    # But wrap each expert's projections with TP layers
    local_experts = nn.ModuleList()

    for i in range(num_local_experts):
        global_idx = expert_start_idx + i
        if global_idx >= num_experts:
            break

        original_expert = moe_layer.experts[global_idx]

        # Create TP-wrapped expert
        # The expert MLP structure:
        # - gate_proj: Column parallel
        # - up_proj: Column parallel
        # - down_proj: Row parallel

        # Copy the original weights to TP layers
        gate_proj = ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            tp_size=tp_size,
            rank=rank,
            bias=False,
            gather_output=False,
            dtype=original_expert.gate_proj.weight.dtype,
            device=original_expert.gate_proj.weight.device,
        )
        # Copy weight: PyTorch linear is [out, in], column parallel splits out dim
        out_per_rank = intermediate_size // tp_size
        start = rank * out_per_rank
        end = start + out_per_rank
        gate_proj.weight.data.copy_(original_expert.gate_proj.weight[start:end, :])

        up_proj = ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            tp_size=tp_size,
            rank=rank,
            bias=False,
            gather_output=False,
            dtype=original_expert.up_proj.weight.dtype,
            device=original_expert.up_proj.weight.device,
        )
        up_proj.weight.data.copy_(original_expert.up_proj.weight[start:end, :])

        down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            tp_size=tp_size,
            rank=rank,
            bias=False,
            input_is_parallel=True,
            dtype=original_expert.down_proj.weight.dtype,
            device=original_expert.down_proj.weight.device,
        )
        # Copy weight: Row parallel splits in dim
        in_per_rank = intermediate_size // tp_size
        start = rank * in_per_rank
        end = start + in_per_rank
        down_proj.weight.data.copy_(original_expert.down_proj.weight[:, start:end])

        # Create expert module with TP layers
        class TPExpert(nn.Module):
            def __init__(self, gate, up, down, act_fn):
                super().__init__()
                self.gate_proj = gate
                self.up_proj = up
                self.down_proj = down
                self.act_fn = act_fn

            def forward(self, x):
                gate = self.act_fn(self.gate_proj(x))
                up = self.up_proj(x)
                return self.down_proj(gate * up)

        tp_expert = TPExpert(
            gate_proj, up_proj, down_proj,
            original_expert.act_fn
        )
        local_experts.append(tp_expert)

    # Replace the experts list
    moe_layer.experts = local_experts

    # Update experts_per_rank if exists
    if hasattr(moe_layer, 'experts_per_rank'):
        moe_layer.experts_per_rank = num_local_experts

    # Note: We keep the gate (router) replicated for now
    # The gate computes which experts each token should go to
    # Since experts are sharded, we need to remap expert indices in forward pass

    # For shared_experts, also apply TP
    if hasattr(moe_layer, 'shared_experts'):
        shared = moe_layer.shared_experts

        # Create TP-wrapped shared expert
        gate_proj = ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            tp_size=tp_size,
            rank=rank,
            bias=False,
            gather_output=False,
            dtype=shared.gate_proj.weight.dtype,
            device=shared.gate_proj.weight.device,
        )
        out_per_rank = intermediate_size // tp_size
        start = rank * out_per_rank
        end = start + out_per_rank
        gate_proj.weight.data.copy_(shared.gate_proj.weight[start:end, :])

        up_proj = ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            tp_size=tp_size,
            rank=rank,
            bias=False,
            gather_output=False,
            dtype=shared.up_proj.weight.dtype,
            device=shared.up_proj.weight.device,
        )
        up_proj.weight.data.copy_(shared.up_proj.weight[start:end, :])

        down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            tp_size=tp_size,
            rank=rank,
            bias=False,
            input_is_parallel=True,
            dtype=shared.down_proj.weight.dtype,
            device=shared.down_proj.weight.device,
        )
        in_per_rank = intermediate_size // tp_size
        start = rank * in_per_rank
        end = start + in_per_rank
        down_proj.weight.data.copy_(shared.down_proj.weight[:, start:end])

        # Create TP shared expert
        class TPSharedExpert(nn.Module):
            def __init__(self, gate, up, down, act_fn):
                super().__init__()
                self.gate_proj = gate
                self.up_proj = up
                self.down_proj = down
                self.act_fn = act_fn

            def forward(self, x):
                gate = self.act_fn(self.gate_proj(x))
                up = self.up_proj(x)
                return self.down_proj(gate * up)

        moe_layer.shared_experts = TPSharedExpert(
            gate_proj, up_proj, down_proj,
            shared.act_fn
        )

    # Store TP info
    moe_layer._tp_size = tp_size
    moe_layer._tp_rank = rank
    moe_layer._expert_start_idx = expert_start_idx
    moe_layer._num_local_experts = num_local_experts

    # IMPORTANT: Replace the forward method to avoid NPU-incompatible operations
    # The original DeepSeek-V2 MoE forward uses torch.nonzero which is not supported on NPU
    # We use a custom forward that avoids this operation
    original_forward = moe_layer.forward

    def npu_compatible_forward(self, hidden_states):
        """
        NPU-compatible forward pass for DeepSeek-V2 MoE

        Avoids torch.nonzero which causes error code 500002 on NPU.
        Uses masked operations instead.

        Note: DeepSeek-V2 MoEGate returns (router_logits, expert_indices, combined_aux_loss)
        where router_logits is [batch, seq_len, top_k] and expert_indices is [batch, seq_len, top_k]
        """
        batch_size, sequence_length, hidden_dim = hidden_states.shape

        # Call gate to get routing decisions
        # gate returns: (router_logits [batch, seq_len, top_k], expert_indices [batch, seq_len, top_k], aux_loss)
        gate_output = self.gate(hidden_states)
        if isinstance(gate_output, tuple) and len(gate_output) >= 2:
            router_logits = gate_output[0]  # [batch, seq_len, top_k] - scores for selected experts
            expert_indices = gate_output[1]   # [batch, seq_len, top_k] - indices of selected experts
        else:
            # Fallback: treat single output as router_logits and run topk
            router_logits = gate_output if not isinstance(gate_output, tuple) else gate_output[0]
            flat_router_logits = router_logits.view(-1, router_logits.size(-1))
            topk_weights, expert_indices = flat_router_logits.topk(getattr(self.gate, 'n_grouped_experts', getattr(self.gate, 'top_k', 6)), dim=-1)
            router_logits = topk_weights.view(batch_size, sequence_length, -1)
            expert_indices = expert_indices.view(batch_size, sequence_length, -1)

        # Flatten for processing
        flat_router_logits = router_logits.view(-1, router_logits.size(-1))  # [batch*seq, top_k]
        flat_expert_indices = expert_indices.view(-1, expert_indices.size(-1))  # [batch*seq, top_k]
        flat_hidden_states = hidden_states.view(-1, hidden_dim)  # [batch*seq, hidden_dim]

        # Normalize weights (convert to float first for NPU compatibility)
        topk_weights = flat_router_logits.float().softmax(dim=-1)  # [batch*seq, top_k]

        # Initialize output
        output = torch.zeros_like(flat_hidden_states)

        # Process each local expert using matrix operations instead of loops
        # This is more efficient on NPU and avoids dynamic indexing issues
        for local_idx in range(self._num_local_experts):
            global_expert_idx = self._expert_start_idx + local_idx

            # Create mask for tokens assigned to this expert across all top_k selections
            # Use multiplication and sum instead of any() to avoid NPU issues
            expert_mask_float = (flat_expert_indices == global_expert_idx).float()  # [batch*seq, top_k]
            tokens_for_expert = expert_mask_float.sum(dim=-1)  # [batch*seq]

            # Check if any token is assigned to this expert
            if tokens_for_expert.max().item() == 0:
                continue

            # Get weights for this expert's tokens (sum across top_k for tokens that selected this expert)
            token_weights = topk_weights * expert_mask_float
            token_weights = token_weights.sum(dim=-1, keepdim=True)  # [batch*seq, 1]

            # Create weighted input - multiply by mask to zero out non-selected tokens
            # Use einsum-like operation with masking
            expert_input_flat = flat_hidden_states * tokens_for_expert.unsqueeze(-1)

            # Reshape for batch processing (all tokens at once)
            # The expert will process all tokens, but only assigned tokens will contribute
            expert_output_all = self.experts[local_idx](expert_input_flat)

            # Scale by normalized token weights and accumulate
            # Only add contribution from tokens that actually selected this expert
            output = output + expert_output_all * token_weights

        # Add shared experts if present
        if hasattr(self, 'shared_experts'):
            shared_output = self.shared_experts(flat_hidden_states)
            output = output + shared_output

        # Reshape back
        output = output.view(batch_size, sequence_length, hidden_dim)

        return output

    # Bind the new forward method
    moe_layer.forward = npu_compatible_forward.__get__(moe_layer, type(moe_layer))

    return moe_layer

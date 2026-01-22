#!/usr/bin/env python3
"""
Simple TP Test - Only MLP layers first
"""

import os
import sys
import torch
import torch_npu
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def setup():
    """Setup NPU and distributed"""
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    print(f"[Rank {rank}] Initializing HCCL...")
    dist.init_process_group(backend="hccl")
    torch.npu.set_device(rank)

    device = torch.device(f"npu:{rank}")
    print(f"[Rank {rank}] NPU device: {device}, World size: {world_size}")

    return rank, world_size, device


def test_basic_all_gather(rank, device):
    """Test basic all_gather"""
    print(f"[Rank {rank}] Testing all_gather...")

    # Create a tensor
    tensor = torch.ones(2, 4, 4, device=device) * (rank + 1)
    print(f"[Rank {rank}] Input tensor shape: {tensor.shape}, sum: {tensor.sum()}")

    # All-gather
    tensor_flat = tensor.view(-1)
    gathered = [torch.empty_like(tensor_flat) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor_flat)

    # Concatenate
    result = torch.cat(gathered, dim=0)
    print(f"[Rank {rank}] Gathered result shape: {result.shape}, sum: {result.sum()}")

    # Expected: 2*4*4 * (1+2) = 32 * 3 = 96 for tp_size=2
    expected_sum = 32 * sum(range(1, dist.get_world_size() + 1))
    assert result.sum() == expected_sum, f"all_gather failed: {result.sum()} != {expected_sum}"
    print(f"[Rank {rank}] all_gather: PASSED")
    return True


def test_column_parallel_linear(rank, device, tp_size):
    """Test ColumnParallelLinear"""
    from npu_parallel import ColumnParallelLinear

    print(f"[Rank {rank}] Testing ColumnParallelLinear...")

    # Create layer
    layer = ColumnParallelLinear(
        in_features=512,
        out_features=2048,
        tp_size=tp_size,
        rank=rank,
        bias=False,
        gather_output=True,
        dtype=torch.bfloat16,
        device=device,
    )

    # Forward pass
    x = torch.randn(2, 128, 512, dtype=torch.bfloat16, device=device)
    out = layer(x)

    print(f"[Rank {rank}] Input shape: {x.shape}")
    print(f"[Rank {rank}] Output shape: {out.shape}")

    # Output should be full size after gather
    assert out.shape == (2, 128, 2048), f"Output shape mismatch: {out.shape}"
    print(f"[Rank {rank}] ColumnParallelLinear: PASSED")
    return True


def test_row_parallel_linear(rank, device, tp_size):
    """Test RowParallelLinear"""
    from npu_parallel import RowParallelLinear

    print(f"[Rank {rank}] Testing RowParallelLinear...")

    # Create layer with full in_features, but internally it uses in_per_rank
    layer = RowParallelLinear(
        in_features=2048,  # Full input size (before splitting)
        out_features=512,
        tp_size=tp_size,
        rank=rank,
        bias=False,
        input_is_parallel=True,  # Input is already split across ranks
        dtype=torch.bfloat16,
        device=device,
    )

    # Forward pass with PARALLEL input (each rank gets in_features/tp_size)
    # For rank 0: first 1024 features
    # For rank 1: last 1024 features
    in_per_rank = 2048 // tp_size
    x = torch.randn(2, 128, in_per_rank, dtype=torch.bfloat16, device=device)
    out = layer(x)

    print(f"[Rank {rank}] Input shape: {x.shape}")
    print(f"[Rank {rank}] Output shape: {out.shape}")

    # Output should be full size after all_reduce
    assert out.shape == (2, 128, 512), f"Output shape mismatch: {out.shape}"
    print(f"[Rank {rank}] RowParallelLinear: PASSED")
    return True


def test_mlp_block(rank, device, tp_size):
    """Test a simple MLP block with TP"""
    from npu_parallel import ColumnParallelLinear, RowParallelLinear
    import torch.nn as nn

    print(f"[Rank {rank}] Testing MLP block...")

    # Create a simple MLP: Linear -> GeLU -> Linear
    class SimpleMLP(nn.Module):
        def __init__(self, tp_size, rank, device):
            super().__init__()
            self.fc1 = ColumnParallelLinear(
                512, 2048, tp_size=tp_size, rank=rank,
                bias=False, gather_output=False,
                dtype=torch.bfloat16, device=device,
            )
            self.fc2 = RowParallelLinear(
                2048, 512, tp_size=tp_size, rank=rank,
                bias=False, input_is_parallel=True,
                dtype=torch.bfloat16, device=device,
            )
            self.act = nn.GELU()

        def forward(self, x):
            x = self.fc1(x)  # Column parallel, output is sharded
            x = self.act(x)
            x = self.fc2(x)  # Row parallel, all_reduce output
            return x

    mlp = SimpleMLP(tp_size, rank, device)
    mlp.eval()

    # Forward pass
    x = torch.randn(2, 128, 512, dtype=torch.bfloat16, device=device)

    with torch.no_grad():
        out = mlp(x)

    print(f"[Rank {rank}] MLP input shape: {x.shape}")
    print(f"[Rank {rank}] MLP output shape: {out.shape}")

    assert out.shape == (2, 128, 512), f"Output shape mismatch: {out.shape}"
    print(f"[Rank {rank}] MLP block: PASSED")
    return True


def main():
    print("\n" + "="*60)
    print("NPU TP Simple Test - MLP Only")
    print("="*60 + "\n")

    rank, world_size, device = setup()
    tp_size = world_size  # Use all NPUs for TP

    results = []

    # Test 1: Basic all_gather
    try:
        test_basic_all_gather(rank, device)
        results.append(("Basic all_gather", True))
    except Exception as e:
        print(f"[Rank {rank}] Basic all_gather: FAILED - {e}")
        results.append(("Basic all_gather", False))

    # Test 2: ColumnParallelLinear
    try:
        test_column_parallel_linear(rank, device, tp_size)
        results.append(("ColumnParallelLinear", True))
    except Exception as e:
        print(f"[Rank {rank}] ColumnParallelLinear: FAILED - {e}")
        import traceback
        traceback.print_exc()
        results.append(("ColumnParallelLinear", False))

    # Test 3: RowParallelLinear
    try:
        test_row_parallel_linear(rank, device, tp_size)
        results.append(("RowParallelLinear", True))
    except Exception as e:
        print(f"[Rank {rank}] RowParallelLinear: FAILED - {e}")
        import traceback
        traceback.print_exc()
        results.append(("RowParallelLinear", False))

    # Test 4: MLP Block
    try:
        test_mlp_block(rank, device, tp_size)
        results.append(("MLP Block", True))
    except Exception as e:
        print(f"[Rank {rank}] MLP Block: FAILED - {e}")
        import traceback
        traceback.print_exc()
        results.append(("MLP Block", False))

    dist.barrier()
    dist.destroy_process_group()

    if rank == 0:
        print("\n" + "="*60)
        print("Test Summary")
        print("="*60)
        for name, result in results:
            status = "PASS" if result else "FAIL"
            print(f"  {name}: {status}")

        if all(r for _, r in results):
            print("\nAll tests PASSED!")
            return 0
        else:
            print("\nSome tests FAILED!")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

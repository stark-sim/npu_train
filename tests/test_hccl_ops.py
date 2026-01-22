#!/usr/bin/env python3
"""
Test HCCL communication operations on NPU
"""

import os
import torch
import torch_npu
import torch.distributed as dist

def setup_npu():
    """Setup NPU device and distributed"""
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    dist.init_process_group(backend="hccl")
    torch.npu.set_device(rank)
    device = torch.device(f"npu:{rank}")

    return rank, world_size, device


def main():
    rank, world_size, device = setup_npu()

    print(f"[Rank {rank}] Testing HCCL operations...")

    # Test 1: Basic tensor creation
    x = torch.randn(4, 4, dtype=torch.float16, device=device)
    print(f"[Rank {rank}] Created tensor: {x.shape}, dtype: {x.dtype}")

    dist.barrier()
    print(f"[Rank {rank}] Barrier passed")

    # Test 2: All-reduce
    y = x.clone()
    dist.all_reduce(y, op=dist.ReduceOp.SUM)
    print(f"[Rank {rank}] All-reduce passed. Sum: {y.sum().item():.2f}")

    dist.barrier()

    # Test 3: Try all_gather_into_tensor
    try:
        # Create a small tensor to gather
        local = torch.ones(2, 2, dtype=torch.float16, device=device) * rank
        gathered = torch.empty(2 * world_size, 2, dtype=torch.float16, device=device)

        dist.all_gather_into_tensor(gathered, local)
        print(f"[Rank {rank}] all_gather_into_tensor passed! Gathered sum: {gathered.sum().item():.2f}")
    except AttributeError as e:
        print(f"[Rank {rank}] all_gather_into_tensor not available: {e}")
        # Try list-based all_gather
        local = torch.ones(2, 2, dtype=torch.float16, device=device) * rank
        gathered_list = [torch.empty_like(local) for _ in range(world_size)]
        dist.all_gather(gathered_list, local)
        result = torch.cat(gathered_list, dim=0)
        print(f"[Rank {rank}] list all_gather passed! Gathered sum: {result.sum().item():.2f}")
    except Exception as e:
        print(f"[Rank {rank}] all_gather_into_tensor failed: {e}")

    dist.barrier()
    print(f"[Rank {rank}] All tests completed!")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

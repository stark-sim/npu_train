#!/usr/bin/env python3
"""
DeepSeek-V2-Lite 4卡TP前向传播测试

测试完整的TP流程：加载模型 -> TP转换 -> NPU前向/反向传播
"""

import os
import time
import torch
import torch_npu
import torch.distributed as dist

# Add parent directory to path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoTokenizer
from npu_parallel import convert_to_tp, sync_gradients_tp


def setup():
    """初始化分布式环境"""
    # 获取环境变量
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # 设置HCCL超时
    os.environ.setdefault("HCCL_CONNECT_TIMEOUT", "7200")
    os.environ.setdefault("HCCL_EXEC_TIMEOUT", "7200")

    # 初始化进程组
    dist.init_process_group(backend="hccl")

    # 设置NPU设备
    torch.npu.set_device(local_rank)
    device = torch.device(f"npu:{local_rank}")

    return rank, world_size, local_rank, device


def main():
    """主测试函数"""
    rank, world_size, local_rank, device = setup()

    # 配置
    MODEL_PATH = "/home/sd/npu_train/models/deepseek-ai-DeepSeek-V2-Lite"
    TP_SIZE = 4  # 4卡TP
    BATCH_SIZE = 1
    SEQ_LEN = 32  # 短序列用于快速测试

    if rank == 0:
        print("=" * 60)
        print("DeepSeek-V2-Lite 4卡TP前向传播测试")
        print("=" * 60)
        print(f"TP Size: {TP_SIZE}")
        print(f"World Size: {world_size}")
        print(f"Batch Size: {BATCH_SIZE}")
        print(f"Seq Len: {SEQ_LEN}")

    dist.barrier()

    # 加载tokenizer (仅rank 0)
    tokenizer = None
    if rank == 0:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        print(f"\n[Rank 0] Vocab size: {len(tokenizer)}")

    dist.barrier()

    # 在CPU上加载模型
    if rank == 0:
        print(f"\n[Rank 0] 正在CPU上加载模型...")

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"[Rank 0] 总参数: {total_params:,} ({total_params/1e9:.2f}B)")
        print(f"[Rank 0] 每个TP rank的参数: {total_params // TP_SIZE:,}")

    dist.barrier()

    # TP转换
    if rank == 0:
        print(f"\n[Rank 0] 正在进行TP转换...")

    tp_rank = rank  # 纯TP，rank就是tp_rank
    model = convert_to_tp(model, tp_size=TP_SIZE, rank=tp_rank)

    if rank == 0:
        print(f"[Rank 0] TP转换完成!")

    dist.barrier()

    # 移动到NPU
    if rank == 0:
        print(f"\n[Rank 0] 正在移动模型到NPU...")

    model.to(device)

    if rank == 0:
        print(f"[Rank 0] 模型已移到NPU")

    dist.barrier()

    # 禁用cache
    model.config.use_cache = False

    # 预热编译
    if rank == 0:
        print(f"\n[Rank 0] 正在预热NPU JIT编译...")

    model.eval()
    with torch.no_grad():
        warmup_input = torch.randint(0, 100002, (BATCH_SIZE, 4), device=device)
        try:
            _ = model(input_ids=warmup_input, use_cache=False)
            if rank == 0:
                print(f"[Rank 0] 编译预热完成!")
        except Exception as e:
            if rank == 0:
                print(f"[Rank 0] 编译预热失败: {e}")

    dist.barrier()

    # 测试前向传播
    if rank == 0:
        print(f"\n[Rank 0] 正在测试前向传播...")

    model.eval()
    test_input = torch.randint(0, 100002, (BATCH_SIZE, SEQ_LEN), device=device)
    start_time = time.time()

    with torch.no_grad():
        output = model(input_ids=test_input, use_cache=False)

    elapsed = time.time() - start_time

    if rank == 0:
        print(f"[Rank 0] 前向传播成功!")
        print(f"[Rank 0] 输出shape: {output.logits.shape}")
        print(f"[Rank 0] 耗时: {elapsed:.2f}s")

    dist.barrier()

    # 测试反向传播
    if rank == 0:
        print(f"\n[Rank 0] 正在测试反向传播...")

    model.train()
    test_input = torch.randint(0, 100002, (BATCH_SIZE, SEQ_LEN // 2), device=device)
    test_labels = test_input.clone()

    outputs = model(input_ids=test_input, labels=test_labels, use_cache=False)
    loss = outputs.loss

    if rank == 0:
        print(f"[Rank 0] Loss: {loss.item():.4f}")

    # 反向传播
    loss.backward()

    if rank == 0:
        print(f"[Rank 0] 反向传播成功!")

    # 同步梯度
    sync_gradients_tp(model, TP_SIZE)

    if rank == 0:
        print(f"[Rank 0] 梯度同步完成!")

    dist.barrier()

    if rank == 0:
        print("\n" + "=" * 60)
        print("✓ 所有测试通过!")
        print("=" * 60)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

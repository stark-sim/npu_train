#!/usr/bin/env python3
"""
简单的DeepSeek-V2-Lite前向传播测试
测试模型加载和CPU/NPU前向传播，不涉及TP
"""

import os
import sys
import torch
import torch_npu

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoTokenizer, AutoModelForCausalLM


def test_basic_forward(model_path, device="npu:0"):
    """测试基本的前向传播（不使用TP）"""

    print("=" * 60)
    print("DeepSeek-V2-Lite 基本前向传播测试")
    print("=" * 60)

    # Load tokenizer
    print("\n[1/4] 加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print(f"      Vocab size: {len(tokenizer)}")

    # Load model on CPU
    print("\n[2/4] 加载模型到CPU...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"      总参数: {total_params:,} ({total_params/1e9:.2f}B)")

    # Inspect model structure
    print("\n[3/4] 检查MoE结构...")
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        first_layer = model.model.layers[0]
        if hasattr(first_layer, "mlp"):
            mlp = first_layer.mlp
            mlp_type = type(mlp).__name__
            print(f"      MLP type: {mlp_type}")

            if hasattr(mlp, "experts"):
                print(f"      专家数量: {len(mlp.experts)}")
            if hasattr(mlp, "gate"):
                print(f"      有gate: Yes")
            if hasattr(mlp, "shared_experts"):
                print(f"      有shared_experts: Yes")

    # Test forward pass on CPU
    print("\n[4/4] 测试前向传播...")
    model.eval()
    test_input = torch.randint(0, len(tokenizer), (1, 32))
    print(f"      输入shape: {test_input.shape}")

    with torch.no_grad():
        output_cpu = model(input_ids=test_input, use_cache=False)
    print(f"      CPU前向传播成功! 输出shape: {output_cpu.logits.shape}")

    # Move to NPU and test
    if torch.npu.is_available():
        print("\n[NPU] 移动模型到NPU...")
        model.to(device)
        print(f"      模型已移到 {device}")

        print("\n[NPU] 测试NPU前向传播...")
        test_input_npu = torch.randint(0, len(tokenizer), (1, 32), device=device)
        print(f"      输入shape: {test_input_npu.shape}")

        with torch.no_grad():
            output_npu = model(input_ids=test_input_npu, use_cache=False)
        print(f"      NPU前向传播成功! 输出shape: {output_npu.logits.shape}")

        print("\n[NPU] 测试NPU反向传播...")
        model.train()
        test_input_npu = torch.randint(0, len(tokenizer), (1, 16), device=device)
        test_labels = test_input_npu.clone()
        outputs = model(input_ids=test_input_npu, labels=test_labels, use_cache=False)
        loss = outputs.loss
        print(f"      Loss: {loss.item():.4f}")

        loss.backward()
        print(f"      NPU反向传播成功!")

        print("\n" + "=" * 60)
        print("✓ 所有测试通过!")
        print("=" * 60)
        return True
    else:
        print("\nNPU不可用")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="测试DeepSeek-V2-Lite基本前向传播")
    parser.add_argument("--model_path", type=str,
                        default="/home/sd/npu_train/models/deepseek-ai-DeepSeek-V2-Lite",
                        help="DeepSeek-V2-Lite模型路径")
    parser.add_argument("--device", type=str, default="npu:0",
                        help="设备 (default: npu:0)")

    args = parser.parse_args()

    success = test_basic_forward(args.model_path, args.device)
    sys.exit(0 if success else 1)

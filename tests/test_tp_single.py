#!/usr/bin/env python3
"""
Simple single-NPU test for model loading and basic TP conversion (no distributed)

This tests:
1. Model loading on single NPU
2. TP layer functionality (without actual distributed communication)
3. Forward pass
"""

import os
import sys
import time
import torch
import torch_npu

# Add current directory to path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_npu_available():
    """Test NPU availability"""
    print("="*60)
    print("Testing NPU Availability")
    print("="*60)

    print(f"torch_npu available: {torch.npu.is_available()}")
    print(f"torch_npu version: {torch_npu.__version__}")
    print(f"NPU count: {torch.npu.device_count()}")

    for i in range(torch.npu.device_count()):
        print(f"  NPU {i}: {torch.npu.get_device_name(i)}")

    return torch.npu.is_available()


def test_model_loading():
    """Test model loading on single NPU"""
    print("\n" + "="*60)
    print("Testing Model Loading on Single NPU")
    print("="*60)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = "/home/sd/npu_train/models/Qwen-Qwen2.5-1.5B-Instruct"
    print(f"Model path: {model_path}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Tokenizer loaded: vocab_size={len(tokenizer)}")

    # Load model in bfloat16
    device = torch.device("npu:0")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {total_params:,} params")

    # Memory info
    allocated = torch.npu.memory_allocated(device) / 1e9
    reserved = torch.npu.memory_reserved(device) / 1e9
    print(f"NPU Memory: allocated={allocated:.2f}GB, reserved={reserved:.2f}GB")

    return model, tokenizer, device


def test_forward_pass(model, tokenizer, device):
    """Test forward pass"""
    print("\n" + "="*60)
    print("Testing Forward Pass")
    print("="*60)

    # Prepare synthetic input
    batch_size = 2
    seq_len = 128
    input_ids = torch.randint(0, 151936, (batch_size, seq_len), device=device)
    labels = input_ids.clone()

    print(f"Input shape: {input_ids.shape}")

    # Warmup
    print("Running warmup...")
    with torch.no_grad():
        _ = model(input_ids=input_ids, labels=labels)
    torch.npu.synchronize()

    # Actual forward pass
    print("Running forward pass...")
    start_time = time.time()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels)
    torch.npu.synchronize()
    forward_time = time.time() - start_time

    loss = outputs.loss
    print(f"Forward pass: loss={loss.item():.4f}, time={forward_time:.3f}s")
    print(f"Throughput: {batch_size * seq_len / forward_time:.0f} tokens/sec")

    return loss.item()


def test_backward_pass(model, tokenizer, device):
    """Test backward pass"""
    print("\n" + "="*60)
    print("Testing Backward Pass")
    print("="*60)

    # Prepare synthetic input
    batch_size = 2
    seq_len = 128
    input_ids = torch.randint(0, 151936, (batch_size, seq_len), device=device)
    labels = input_ids.clone()

    # Forward + Backward
    model.train()
    print("Running forward + backward...")
    start_time = time.time()

    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs.loss
    loss.backward()

    torch.npu.synchronize()
    total_time = time.time() - start_time

    # Check gradients
    grad_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm += p.grad.norm().item() ** 2
    grad_norm = grad_norm ** 0.5

    print(f"Forward+Backward: loss={loss.item():.4f}, time={total_time:.3f}s")
    print(f"Gradient norm: {grad_norm:.4f}")


def test_tp_layers():
    """Test TP layer imports and basic functionality"""
    print("\n" + "="*60)
    print("Testing TP Layer Imports")
    print("="*60)

    try:
        from npu_parallel import ColumnParallelLinear, RowParallelLinear
        print("ColumnParallelLinear: OK")
        print("RowParallelLinear: OK")
    except ImportError as e:
        print(f"ERROR importing TP layers: {e}")
        return False

    # Test basic layer creation
    device = torch.device("npu:0")
    try:
        col_layer = ColumnParallelLinear(
            in_features=512,
            out_features=2048,
            tp_size=4,
            rank=0,
            bias=False,
            gather_output=True,
            dtype=torch.bfloat16,
            device=device,
        )
        print(f"ColumnParallelLinear created: in=512, out=2048, tp_size=4")

        row_layer = RowParallelLinear(
            in_features=2048,
            out_features=512,
            tp_size=4,
            rank=0,
            bias=False,
            input_is_parallel=True,
            dtype=torch.bfloat16,
            device=device,
        )
        print(f"RowParallelLinear created: in=2048, out=512, tp_size=4")

        # Test forward pass (without actual distributed communication)
        x = torch.randn(2, 128, 512, dtype=torch.bfloat16, device=device)
        print(f"Input shape: {x.shape}")

        # Column parallel (without gather for single device test)
        col_layer.gather_output = False
        out = col_layer(x)
        print(f"ColumnParallel output (no gather): {out.shape}")

        print("TP layers: OK")
        return True

    except Exception as e:
        print(f"ERROR testing TP layers: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function"""
    print("\n" + "="*60)
    print("NPU Single-Device Test - 1.5B Qwen Model")
    print("="*60 + "\n")

    results = []

    # Test 1: NPU availability
    if not test_npu_available():
        print("\nERROR: NPU not available!")
        return 1
    results.append(("NPU Available", True))

    # Test 2: TP layer imports
    results.append(("TP Layers", test_tp_layers()))

    # Test 3: Model loading
    try:
        model, tokenizer, device = test_model_loading()
        results.append(("Model Loading", True))
    except Exception as e:
        print(f"\nERROR loading model: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Model Loading", False))
        return 1

    # Test 4: Forward pass
    try:
        test_forward_pass(model, tokenizer, device)
        results.append(("Forward Pass", True))
    except Exception as e:
        print(f"\nERROR in forward pass: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Forward Pass", False))

    # Test 5: Backward pass
    try:
        test_backward_pass(model, tokenizer, device)
        results.append(("Backward Pass", True))
    except Exception as e:
        print(f"\nERROR in backward pass: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Backward Pass", False))

    # Summary
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


if __name__ == "__main__":
    sys.exit(main())

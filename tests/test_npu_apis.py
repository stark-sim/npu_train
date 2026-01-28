#!/usr/bin/env python3
"""
Research torch_npu specific optimization APIs and features.

This script tests various torch_npu features to understand what's available
for performance optimization on Ascend NPU.
"""

import sys
import torch
import torch_npu


def print_section(title):
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def test_basic_info():
    """Print basic torch_npu information"""
    print_section("Basic torch_npu Information")

    print(f"PyTorch version: {torch.__version__}")
    print(f"torch_npu version: {torch_npu.__version__}")

    npu_available = torch.npu.is_available()
    print(f"NPU available: {npu_available}")

    if npu_available:
        npu_count = torch.npu.device_count()
        print(f"NPU device count: {npu_count}")
        for i in range(npu_count):
            print(f"  NPU {i}: {torch.npu.get_device_name(i)}")

        # Get current device
        current_device = torch.npu.current_device()
        print(f"Current NPU device: {current_device}")

        # Get device capabilities
        print(f"NPU capability: {torch.npu.get_device_capability(current_device)}")
    else:
        print("NPU not available - skipping device-specific tests")


def test_npu_apis():
    """Test various torch_npu APIs"""
    print_section("torch_npu API Availability")

    # Test for npu.prepare (model compilation/optimization)
    if hasattr(torch.npu, "prepare"):
        print("✓ torch.npu.prepare available")
    else:
        print("✗ torch.npu.prepare NOT available")

    # Test for npu.compile (graph compilation)
    if hasattr(torch.npu, "compile"):
        print("✓ torch.npu.compile available")
    else:
        print("✗ torch.npu.compile NOT available")

    # Test for npu.optimize (general optimization)
    if hasattr(torch.npu, "optimize"):
        print("✓ torch.npu.optimize available")
    else:
        print("✗ torch.npu.optimize NOT available")

    # Test for JIT compilation support
    if hasattr(torch.npu, "jit"):
        print("✓ torch.npu.jit available")
    else:
        print("✗ torch.npu.jit NOT available")

    # Test for fused operations
    print("\nFused Operations:")
    fused_ops = [
        "npu_conv2d",  # Fused conv2d
        "npu_batch_norm",  # Fused batch norm
        "npu_layer_norm",  # Fused layer norm
        "npu_dropout",  # Fused dropout
        "npu_quick_gelu",  # Quick GELU activation
        "npu_mul",  # Fused multiply
        "npu_add",  # Fused add
    ]

    for op in fused_ops:
        if hasattr(torch.npu, op):
            print(f"  ✓ torch.{op} available")
        else:
            print(f"  ✗ torch.{op} NOT available")


def test_amp_features():
    """Test torch_npu AMP features"""
    print_section("torch_npu AMP Features")

    # Test GradScaler
    try:
        scaler = torch_npu.amp.GradScaler()
        print("✓ torch_npu.amp.GradScaler available")
    except Exception as e:
        print(f"✗ torch_npu.amp.GradScaler failed: {e}")

    # Test ShardedGradScaler (for distributed training)
    try:
        sharded_scaler = torch_npu.amp.ShardedGradScaler()
        print("✓ torch_npu.amp.ShardedGradScaler available")
    except Exception as e:
        print(f"✗ torch_npu.amp.ShardedGradScaler failed: {e}")

    # Test autocast
    try:
        with torch_npu.amp.autocast(dtype=torch.float16):
            x = torch.randn(10, 10).npu()
            y = x @ x.T
        print("✓ torch_npu.amp.autocast works")
    except Exception as e:
        print(f"✗ torch_npu.amp.autocast failed: {e}")


def test_communication_features():
    """Test HCCL communication features"""
    print_section("HCCL Communication Features")

    # Check if distributed is available
    if torch.distributed.is_available():
        print("✓ torch.distributed available")

        # Check for HCCL backend
        backends = torch.distributed.available_backends()
        print(f"Available backends: {backends}")

        if "hccl" in backends:
            print("✓ HCCL backend available")
        else:
            print("✗ HCCL backend NOT available")
    else:
        print("✗ torch.distributed NOT available")


def test_memory_features():
    """Test memory-related features"""
    print_section("Memory Features")

    if not torch.npu.is_available():
        print("NPU not available - skipping memory tests")
        return

    try:
        device = torch.npu.current_device()

        # Get memory stats
        allocated = torch.npu.memory_allocated(device)
        reserved = torch.npu.memory_reserved(device)
        print(f"Memory allocated: {allocated / 1024**2:.2f} MB")
        print(f"Memory reserved: {reserved / 1024**2:.2f} MB")

        # Get memory summary
        print("\nMemory summary:")
        print(torch.npu.memory_summary(device))

        # Check for memory efficiency features
        if hasattr(torch.npu, "memory_empty_cache"):
            print("✓ torch.npu.memory_empty_cache available")

        if hasattr(torch.npu, "release_memory"):
            print("✓ torch.npu.release_memory available")

    except Exception as e:
        print(f"Memory test failed: {e}")


def test_sdpa():
    """Test Scaled Dot-Product Attention"""
    print_section("Scaled Dot-Product Attention (SDPA)")

    try:
        import torch.nn.functional as F

        # Create test tensors
        batch, heads, seq, dim = 2, 8, 128, 64
        q = torch.randn(batch, heads, seq, dim, dtype=torch.float16)
        k = torch.randn(batch, heads, seq, dim, dtype=torch.float16)
        v = torch.randn(batch, heads, seq, dim, dtype=torch.float16)

        # Test SDPA
        output = F.scaled_dot_product_attention(q, k, v)
        print(f"✓ F.scaled_dot_product_attention works")
        print(f"  Output shape: {output.shape}")

        # Check which SDPA backend is available
        if hasattr(F, "sdpa_backend"):
            backends = F.sdpa_backend(q, k, v)
            print(f"  SDPA backend: {backends}")

    except Exception as e:
        print(f"✗ SDPA failed: {e}")


def test_compile():
    """Test torch.compile (PyTorch 2.0+)"""
    print_section("torch.compile")

    try:
        # Simple model
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x):
                return self.linear(x)

        model = SimpleModel()

        # Try to compile
        compiled = torch.compile(model, fullgraph=False)
        print("✓ torch.compile available")

        # Test on NPU if available
        if torch.npu.is_available():
            try:
                x = torch.randn(5, 10).npu()
                model = model.npu()
                compiled = torch.compile(model, fullgraph=False)
                output = compiled(x)
                print("✓ torch.compile works on NPU")
            except Exception as e:
                print(f"✗ torch.compile on NPU failed: {e}")

    except Exception as e:
        print(f"✗ torch.compile failed: {e}")


def main():
    """Run all tests"""
    print("torch_npu Optimization API Research")
    print("="*60)

    test_basic_info()
    test_npu_apis()
    test_amp_features()
    test_communication_features()
    test_memory_features()
    test_sdpa()
    test_compile()

    print_section("Summary")
    print("Tests complete. Review results above for available optimizations.")
    print("\nRecommended next steps based on findings:")
    print("1. If torch.compile works: Enable for models (20-30% speedup)")
    print("2. If SDPA works: Already implemented in tp_attention.py")
    print("3. If fused ops available: Consider replacing manual implementations")
    print("4. Check HCCL optimizations for communication tuning")


if __name__ == "__main__":
    main()

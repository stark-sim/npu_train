#!/usr/bin/env python3
"""
Test DeepSeek-V2 MoE TP conversion without training

This script validates that:
1. The model can be loaded
2. TP conversion works
3. The NPU-compatible forward pass runs without errors

Use this to debug issues before attempting full training.
"""

import os
import sys
import torch
import torch_npu

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from npu_parallel import convert_to_tp
from transformers import AutoTokenizer, AutoModelForCausalLM


def test_moe_conversion(model_path, device="npu:0"):
    """Test MoE model conversion to TP"""

    print("=" * 60)
    print("DeepSeek-V2 MoE TP Conversion Test")
    print("=" * 60)

    # Load tokenizer
    print("\n[1/6] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print(f"      Vocab size: {len(tokenizer)}")

    # Load model on CPU
    print("\n[2/6] Loading model on CPU...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"      Total params: {total_params:,}")

    # Inspect model structure
    print("\n[3/6] Inspecting MoE structure...")
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        first_layer = model.model.layers[0]
        if hasattr(first_layer, "mlp"):
            mlp = first_layer.mlp
            mlp_type = type(mlp).__name__
            print(f"      MLP type: {mlp_type}")

            if hasattr(mlp, "experts"):
                print(f"      Num experts: {len(mlp.experts)}")
            if hasattr(mlp, "gate"):
                print(f"      Has gate: Yes")
            if hasattr(mlp, "shared_experts"):
                print(f"      Has shared_experts: Yes")

    # Test forward pass on CPU without TP first
    print("\n[4/6] Testing forward pass on CPU (without TP)...")
    model.eval()
    test_input = torch.randint(0, len(tokenizer), (1, 16))
    with torch.no_grad():
        output = model(input_ids=test_input, use_cache=False)
    print(f"      CPU forward pass successful! Output shape: {output.logits.shape}")

    # Convert to TP (this will modify the model)
    print("\n[5/6] Testing TP conversion structure...")
    tp_size = 4
    rank = 0

    try:
        model_tp = convert_to_tp(model, tp_size=tp_size, rank=rank)
        print(f"      TP conversion successful!")
        print(f"      Note: Converted model requires distributed init for forward pass")

        # Move to NPU if available
        if torch.npu.is_available():
            print("\n[6/6] Moving model to NPU...")
            model_tp.to(device)
            print(f"      Model moved to {device}")

            print("\n[NPU] Testing with distributed initialization...")

            # Import distributed and initialize for single process
            import torch.distributed as dist

            # Set environment for single-process TP
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", "29500")
            os.environ.setdefault("WORLD_SIZE", "1")
            os.environ.setdefault("RANK", "0")

            # Initialize process group
            dist.init_process_group(
                backend="hccl" if torch.npu.is_available() else "gloo",
                world_size=1,
                rank=0
            )

            print(f"      Distributed group initialized")

            # Test forward pass with distributed
            model_tp.eval()
            test_input = torch.randint(0, len(tokenizer), (1, 4), device=device)
            with torch.no_grad():
                output = model_tp(input_ids=test_input, use_cache=False)
            print(f"      NPU forward pass successful! Output shape: {output.logits.shape}")

            # Cleanup
            dist.destroy_process_group()

        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n[ERROR] Test failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test DeepSeek-V2 MoE TP conversion")
    parser.add_argument("--model_path", type=str,
                        default="/home/sd/npu_train/models/deepseek-ai/DeepSeek-V2-Lite",
                        help="Path to DeepSeek-V2-Lite model")
    parser.add_argument("--device", type=str, default="npu:0",
                        help="Device to test on (default: npu:0)")

    args = parser.parse_args()

    success = test_moe_conversion(args.model_path, args.device)
    sys.exit(0 if success else 1)

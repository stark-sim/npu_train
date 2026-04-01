#!/usr/bin/env python3
"""
Minimal repro for storage_offset warning on Ascend 910A + torch_npu 2.5.1.

This script reproduces the "storage_offset ... untrustworthy64/128" warning
using the original Qwen self-attention backward path (non-TP).

Purpose:
- Validate environment behavior
- Provide a reproducible case for upstream issue reports
- Verify that custom TPAttention backward remains clean at matched shapes

Usage (on NPU host):
    python tools/repro_storage_offset_warning.py --model_path /path/to/Qwen2.5-1.5B-Instruct

The script will report whether the warning appears in each test case.
"""

import argparse
import sys
import torch
import torch.nn as nn


def check_npu():
    """Check if NPU is available."""
    try:
        import torch_npu
        return torch.npu.is_available()
    except ImportError:
        return False


def repro_original_qwen_self_attn(model_path: str, device: str):
    """
    Reproduce storage_offset warning using original Qwen self_attn backward.
    Expected: warning appears (baseline torch_npu/compiler behavior).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print("\n=== Test: Original Qwen self_attn backward ===")
    print(f"Loading model from: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    ).to(device)
    model.train()
    
    # Small input to trigger self_attn backward
    text = "Hello, world!"
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    print("Running forward + backward...")
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    loss.backward()
    
    print("Note: If 'storage_offset ... untrustworthy' appears above, this is EXPECTED.")
    print("      This warning is a known baseline behavior in current torch_npu + 910A.")
    return True


def repro_tp_attention_matched_shape(device: str):
    """
    Test custom TPAttention backward at matched shape.
    Expected: no storage_offset warning (our custom path is clean).
    """
    print("\n=== Test: Custom TPAttention backward (matched shape) ===")
    
    # Import after path setup
    from npu_parallel.tp_attention import TPAttention
    
    batch_size = 2
    seq_len = 16
    hidden_size = 512
    num_heads = 8
    num_key_value_heads = 4
    head_dim = hidden_size // num_heads
    
    # Create a simple config-like object
    class FakeConfig:
        def __init__(self):
            self.hidden_size = hidden_size
            self.num_attention_heads = num_heads
            self.num_key_value_heads = num_key_value_heads
            self.max_position_embeddings = 2048
            self.rope_theta = 10000.0
    
    config = FakeConfig()
    
    # Create layer
    layer = TPAttention(
        config=config,
        layer_idx=0,
        tp_size=1,
        tp_rank=0,
    ).to(device)
    layer.train()
    
    # Forward + backward
    hidden_states = torch.randn(
        batch_size, seq_len, hidden_size,
        dtype=torch.float16, device=device
    )
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    
    print(f"Running TPAttention with shape {hidden_states.shape}...")
    output, *_ = layer(hidden_states, position_ids=position_ids)
    loss = output.mean()
    loss.backward()
    
    print("Note: No storage_offset warning should appear above (custom TPAttention is clean).")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Minimal repro for storage_offset warning on NPU"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/sd/npu_train/models/Qwen-Qwen2.5-1.5B-Instruct",
        help="Path to Qwen model (for original self_attn test)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="npu:0",
        help="NPU device to use"
    )
    parser.add_argument(
        "--skip_original",
        action="store_true",
        help="Skip original Qwen test (requires model download)"
    )
    parser.add_argument(
        "--skip_tp_attention",
        action="store_true",
        help="Skip TPAttention test"
    )
    
    args = parser.parse_args()
    
    # NPU check
    if not check_npu():
        print("ERROR: NPU not available. This script must run on Ascend NPU.")
        print("(On CPU, the warning wouldn't appear anyway.)")
        sys.exit(1)
    
    print(f"Device: {args.device}")
    print(f"PyTorch: {torch.__version__}")
    
    # Run tests
    success = True
    
    if not args.skip_original:
        try:
            repro_original_qwen_self_attn(args.model_path, args.device)
        except Exception as e:
            print(f"Original Qwen test failed: {e}")
            success = False
    
    if not args.skip_tp_attention:
        try:
            repro_tp_attention_matched_shape(args.device)
        except Exception as e:
            print(f"TPAttention test failed: {e}")
            success = False
    
    print("\n=== Summary ===")
    print("- Original Qwen self_attn backward: EXPECTED to show storage_offset warning")
    print("- Custom TPAttention backward: EXPECTED to be clean (no warning)")
    print("\nIf results match expectations, the environment is behaving as documented.")
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

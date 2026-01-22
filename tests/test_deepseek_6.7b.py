#!/usr/bin/env python3
"""
Test Tensor Parallel Training with DeepSeek-Coder-6.7B model
This is a dense version (not MoE), so TP should work correctly.
"""
import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    print("=" * 60)
    print("DeepSeek-Coder-6.7B Tensor Parallel Training Test")
    print("=" * 60)

    # Check model path
    model_path = "/home/sd/npu_train/models/deepseek-coder-6.7b-base"

    if not os.path.exists(model_path):
        print(f"\nModel not found at {model_path}")
        print("Please download the model first:")
        print("  python download_deepseek_6.7b.py")
        return False

    # Check if safetensors files exist
    import glob
    safetensors_files = glob.glob(os.path.join(model_path, "*.safetensors"))
    if not safetensors_files:
        print(f"\nNo safetensors files found in {model_path}")
        print("Download may still be in progress...")
        return False

    print(f"\nModel path: {model_path}")
    print(f"Safetensors files: {len(safetensors_files)}")
    for f in safetensors_files:
        size_mb = os.path.getsize(f) / (1024*1024)
        print(f"  - {os.path.basename(f)}: {size_mb:.0f} MB")

    # Import TP modules
    from npu_parallel import convert_to_tp, sync_gradients_tp
    from npu_parallel.supported_models import check_model_compatibility

    # Check compatibility
    result = check_model_compatibility("DeepSeekForCausalLM")
    print(f"\nCompatibility Check:")
    print(f"  Compatible: {result['compatible']}")
    print(f"  Architecture: {result['architecture']}")
    print(f"  Notes: {result['notes']}")

    # Check model config
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    print(f"\nModel Config:")
    print(f"  Model type: {config.model_type}")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Num layers: {config.num_hidden_layers}")
    print(f"  Num attention heads: {config.num_attention_heads}")
    print(f"  Vocab size: {config.vocab_size}")

    # Calculate TP memory requirements
    total_params = 6.7e9  # 6.7B parameters
    for tp_size in [2, 4, 8]:
        params_per_rank = total_params / tp_size
        # bfloat16 = 2 bytes per param
        bytes_per_rank = params_per_rank * 2
        gb_per_rank = bytes_per_rank / (1024**3)
        print(f"\n  {tp_size}-way TP: ~{gb_per_rank:.1f} GB per rank")

    print("\n" + "=" * 60)
    print("To run TP training with this model:")
    print("=" * 60)
    print("\n# 2-way TP (test)")
    print("torchrun --nproc_per_node=2 train_tp_7b_simple.py \\")
    print(f"    --model_path {model_path} \\")
    print("    --tp_size 2 --batch_size 1 --max_length 64 --steps 10")
    print("\n# 4-way TP (recommended)")
    print("torchrun --nproc_per_node=4 train_tp_7b_simple.py \\")
    print(f"    --model_path {model_path} \\")
    print("    --tp_size 4 --batch_size 1 --max_length 64 --steps 10")

    print("\nNote: DeepSeek-Coder-6.7B uses standard GPT + SwiGLU architecture")
    print("      (not MoE), so TP conversion should work correctly!")

    return True

if __name__ == "__main__":
    main()

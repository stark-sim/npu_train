#!/usr/bin/env python3
"""
Download Qwen2.5-7B model using HuggingFace with resume support
Uses HF mirror for faster download.
"""

import os
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download

# Configuration
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
CACHE_DIR = "./models"
HF_ENDPOINT = "https://hf-mirror.com"  # Use mirror
HF_HUB_DOWNLOAD_TIMEOUT = 1200  # 20 minutes

def main():
    # Set HF endpoint to mirror
    os.environ["HF_ENDPOINT"] = HF_ENDPOINT
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(HF_HUB_DOWNLOAD_TIMEOUT)

    print("=" * 60)
    print("HuggingFace Download for Qwen2.5-7B")
    print("=" * 60)
    print(f"Model ID: {MODEL_ID}")
    print(f"Cache Dir: {CACHE_DIR}")
    print(f"HF Endpoint: {HF_ENDPOINT}")
    print("=" * 60)

    try:
        print("\nDownloading model with resume support...")
        print("This may take a while.\n")

        # Use snapshot_download which has built-in resume
        model_dir = snapshot_download(
            repo_id=MODEL_ID,
            cache_dir=CACHE_DIR,
            resume_download=True,  # Enable resume
            local_files_only=False,
            endpoint=HF_ENDPOINT,
        )

        print(f"\n{'=' * 60}")
        print(f"Download completed!")
        print(f"Model saved to: {model_dir}")
        print(f"{'=' * 60}")

        # Verify the model can be loaded
        print("\nVerifying model can be loaded...")
        try:
            print("Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
            print("Tokenizer loaded successfully!")

            print("Loading model...")
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                torch_dtype="auto",
                trust_remote_code=True,
                device_map="cpu",  # Load on CPU to verify
            )
            print("Model loaded successfully!")

            print(f"\nModel type: {type(model).__name__}")
            print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

            print("\n" + "=" * 60)
            print("Model verification PASSED!")
            print("=" * 60)

        except Exception as e:
            print(f"\nError during verification: {e}")
            import traceback
            traceback.print_exc()
            print("\n" + "=" * 60)
            print("WARNING: Model download may be incomplete!")
            print("=" * 60)
            sys.exit(1)

    except Exception as e:
        print(f"\nError during download: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

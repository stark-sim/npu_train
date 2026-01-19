#!/usr/bin/env python3
"""
Example: Load Qwen models for PP/TP experiments
Shows how to load models with different parallelism strategies
"""

import os
import torch
import torch_npu
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set environment for NPU
os.environ["PYTORCH_NPU_ALLOC_CONF"] = "max_split_size_mb:512"

BASE_DIR = "/home/sd/npu_train/models"

def load_model_single_gpu(model_path):
    """Load model on single GPU"""
    print(f"Loading {model_path} on single GPU...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",  # Let transformers handle device placement
        trust_remote_code=True
    )
    
    print(f"Model loaded on device: {next(model.parameters()).device}")
    print(f"Model dtype: {next(model.parameters()).dtype}")
    
    return model, tokenizer

def test_model(model, tokenizer, prompt="Hello, how are you?"):
    """Test model inference"""
    print(f"\nTesting with prompt: '{prompt}'")
    
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Response: {response}")
    
    return response

def main():
    """Test loading all models"""
    print("=== Qwen Model Loading Test ===\n")
    
    # Check NPU availability
    print(f"NPU available: {torch.npu.is_available()}")
    print(f"NPU count: {torch.npu.device_count()}")
    print()
    
    # Test each model
    models = [
        "Qwen-Qwen2.5-1.5B-Instruct",
        "Qwen-Qwen2.5-7B-Instruct", 
        "Qwen-Qwen2.5-14B-Instruct"
    ]
    
    for model_name in models:
        model_path = Path(BASE_DIR) / model_name
        
        if not model_path.exists():
            print(f"❌ Model path not found: {model_path}")
            continue
        
        try:
            # Load model
            model, tokenizer = load_model_single_gpu(model_path)
            
            # Test inference
            test_model(model, tokenizer, "What is machine learning?")
            
            # Clean up to free memory
            del model
            del tokenizer
            torch.npu.empty_cache()
            
            print(f"✅ {model_name} test completed successfully\n")
            
        except Exception as e:
            print(f"❌ Failed to load/test {model_name}: {e}\n")

if __name__ == "__main__":
    main()

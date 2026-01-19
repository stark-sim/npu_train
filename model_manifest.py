#!/usr/bin/env python3
"""
Generate model manifest for PP/TP experiments
Shows safetensors sharding and basic model info
"""

import os
import json
from pathlib import Path

BASE_DIR = "/home/sd/npu_train/models"

def get_model_info(model_dir):
    """Get basic model info and safetensors sharding"""
    model_path = Path(model_dir)
    model_name = model_path.name
    
    # Load config if available
    config_path = model_path / "config.json"
    config = {}
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except:
            pass
    
    # Get safetensors info
    safetensors_files = list(model_path.glob("*.safetensors"))
    safetensors_info = []
    
    total_size = 0
    for st_file in sorted(safetensors_files):
        size = st_file.stat().st_size
        total_size += size
        safetensors_info.append({
            "file": st_file.name,
            "size_gb": size / (1024**3),
            "size_mb": size / (1024**2)
        })
    
    return {
        "model_name": model_name,
        "original_repo": model_name.replace("Qwen-Qwen2.5-", "Qwen/Qwen2.5-").replace("-Instruct", "-Instruct"),
        "model_type": config.get("model_type", "unknown"),
        "num_hidden_layers": config.get("num_hidden_layers", "unknown"),
        "num_attention_heads": config.get("num_attention_heads", "unknown"),
        "hidden_size": config.get("hidden_size", "unknown"),
        "vocab_size": config.get("vocab_size", "unknown"),
        "safetensors_shards": len(safetensors_files),
        "safetensors_files": safetensors_info,
        "total_size_gb": total_size / (1024**3),
        "has_tokenizer": (model_path / "tokenizer.json").exists(),
        "has_config": config_path.exists()
    }

def main():
    """Generate manifest for all Qwen models"""
    print("=== Qwen Model Manifest ===\n")
    
    models = []
    for model_dir in Path(BASE_DIR).glob("Qwen-Qwen2.5-*"):
        if model_dir.is_dir():
            info = get_model_info(model_dir)
            models.append(info)
    
    # Sort by model size
    models.sort(key=lambda x: x["total_size_gb"])
    
    # Print summary
    print(f"Found {len(models)} models:\n")
    
    for model in models:
        print(f"📦 {model['model_name']}")
        print(f"   Repo: {model['original_repo']}")
        print(f"   Type: {model['model_type']}")
        print(f"   Layers: {model['num_hidden_layers']}, Heads: {model['num_attention_heads']}")
        print(f"   Hidden size: {model['hidden_size']}, Vocab: {model['vocab_size']}")
        print(f"   Safetensors: {model['safetensors_shards']} shards, {model['total_size_gb']:.2f} GB")
        
        if model['safetensors_shards'] > 1:
            print("   Shards:")
            for shard in model['safetensors_files']:
                print(f"     - {shard['file']} ({shard['size_gb']:.2f} GB)")
        
        print(f"   Tokenizer: {'✅' if model['has_tokenizer'] else '❌'}")
        print(f"   Config: {'✅' if model['has_config'] else '❌'}")
        print()
    
    # Save manifest
    manifest_path = Path(BASE_DIR) / "model_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(models, f, indent=2)
    
    print(f"Manifest saved to: {manifest_path}")
    
    # PP/TP suggestions
    print("\n=== PP/TP Suggestions ===")
    for model in models:
        shards = model['safetensors_shards']
        layers = model['num_hidden_layers']
        
        if isinstance(layers, int) and isinstance(shards, int):
            # Pipeline Parallelism suggestions
            pp_options = []
            if layers % 2 == 0:
                pp_options.append(f"PP-{layers//2}")
            if layers % 4 == 0:
                pp_options.append(f"PP-{layers//4}")
            
            # Tensor Parallelism suggestions (based on shards)
            tp_options = []
            if shards == 1:
                tp_options.append("TP-1 (single)")
            elif shards == 2:
                tp_options.append("TP-2")
            elif shards == 4:
                tp_options.append("TP-4")
            elif shards == 8:
                tp_options.append("TP-8")
            
            print(f"{model['model_name']}:")
            if pp_options:
                print(f"  Pipeline: {', '.join(pp_options)}")
            if tp_options:
                print(f"  Tensor: {', '.join(tp_options)}")
            if not pp_options and not tp_options:
                print(f"  Single GPU or custom parallelism")
            print()

if __name__ == "__main__":
    main()

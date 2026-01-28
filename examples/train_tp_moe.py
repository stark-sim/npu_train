#!/usr/bin/env python3
"""
Tensor Parallel Training for MoE Models on Ascend NPU

This script extends the custom NPU TP implementation to support MoE models:
- DeepSeek-V2/V3 (DeepSeekMoE)
- Mixtral-8x7B (MixtralSparseMoeBlock)
- Qwen2MoE

Key Features:
- Expert sharding across TP ranks
- Load balancing auxiliary loss
- All-to-all communication for token dispatch
"""

import os
import time
import argparse
import torch
import torch_npu
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)

# Import custom NPU TP implementation
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from npu_parallel import convert_to_tp, sync_gradients_tp


def setup_npu():
    """Setup NPU device and distributed"""
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Increase HCCL timeout for large models (long compile time on first run)
    # DeepSeek-V2-Lite is very large and may need extended timeout
    os.environ.setdefault("HCCL_CONNECT_TIMEOUT", "7200")  # 2 hours
    os.environ.setdefault("HCCL_EXEC_TIMEOUT", "7200")     # 2 hours

    # Set NPU optimization options for large models
    # Disable JIT compilation flags that cause issues
    os.environ.setdefault("TORCH_NPU_ENABLE_COMGR", "0")
    os.environ.setdefault("TORCH_NPU_ALLOC_CONF", "max_split_size_mb:128")

    # Enable fusion for better performance
    os.environ.setdefault("NPU_FUSION_ENABLE", "1")

    # Initialize distributed with HCCL backend
    dist.init_process_group(backend="hccl")

    # Set NPU device
    torch.npu.set_device(rank)
    device = torch.device(f"npu:{rank}")

    print(f"[Rank {rank}] NPU device: {device}")
    print(f"[Rank {rank}] World size: {world_size}")
    print(f"[Rank {rank}] HCCL timeout: 7200s")

    return rank, world_size, device


def get_args():
    parser = argparse.ArgumentParser(description="MoE TP Training on NPU")
    parser.add_argument("--model_path", type=str, default="/home/sd/npu_train/models/deepseek-ai/DeepSeek-V2-Lite",
                        help="Model path (MoE model like DeepSeek-V2-Lite, Mixtral-8x7B)")
    parser.add_argument("--tp_size", type=int, default=4,
                        help="Tensor parallel size (default: 4)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size per device (default: 1 for MoE)")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Max sequence length (default: 512)")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of epochs (default: 1)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (default: 1e-4)")
    parser.add_argument("--warmup_steps", type=int, default=5,
                        help="Warmup steps (default: 5)")
    parser.add_argument("--aux_loss_coef", type=float, default=0.01,
                        help="Auxiliary loss coefficient (default: 0.01)")
    parser.add_argument("--log_interval", type=int, default=2,
                        help="Log every N steps (default: 2)")
    parser.add_argument("--save_path", type=str, default="./output_tp_moe",
                        help="Model save path")
    return parser.parse_args()


def prepare_synthetic_data(args, tokenizer, rank):
    """Generate synthetic training data"""
    sample_texts = [
        "The quick brown fox jumps over the lazy dog. " * 10,
        "Machine learning is transforming artificial intelligence. " * 10,
        "Deep learning models can learn complex patterns. " * 10,
        "Neural networks are inspired by brain structure. " * 10,
        "Natural language processing enables text understanding. " * 10,
        "Transformers revolutionized natural language processing. " * 10,
        "Attention mechanisms focus on relevant information. " * 10,
        "Pre-trained models adapt to various tasks. " * 10,
        "Mixture of Experts models activate specialized subnetworks. " * 10,
        "Load balancing ensures uniform expert utilization. " * 10,
    ]

    max_samples = 50  # Smaller dataset for testing
    texts = []
    for i in range(max_samples):
        texts.append(sample_texts[i % len(sample_texts)])

    # Tokenize
    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=args.max_length,
        padding="max_length",
        return_tensors="pt",
    )

    input_ids = encodings["input_ids"]
    labels = input_ids.clone()

    # Create dataset
    dataset = torch.utils.data.TensorDataset(input_ids, labels)

    # Distributed sampler
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )

    return dataloader


def inspect_model_moe(model, rank):
    """Inspect model to detect and print MoE structure"""
    if rank != 0:
        return

    print("\n" + "=" * 60)
    print("Model Structure Inspection")
    print("=" * 60)

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        first_layer = model.model.layers[0]

        print(f"Layer 0 structure:")
        print(f"  - self_attn: {type(first_layer.self_attn).__name__}")

        if hasattr(first_layer, "mlp"):
            mlp = first_layer.mlp
            print(f"  - mlp: {type(mlp).__name__}")

            # Check for MoE patterns
            if hasattr(mlp, "experts"):
                print(f"    - Has 'experts': Yes ({len(mlp.experts)} experts)")
            if hasattr(mlp, "gate"):
                print(f"    - Has 'gate': Yes")
            if hasattr(mlp, "num_experts"):
                print(f"    - num_experts: {mlp.num_experts}")
            if hasattr(mlp, "block_sparse_moe"):
                print(f"    - Has 'block_sparse_moe': Yes")
                if hasattr(mlp.block_sparse_moe, "experts"):
                    print(f"      - experts: {len(mlp.block_sparse_moe.experts)}")
                if hasattr(mlp.block_sparse_moe, "num_local_experts"):
                    print(f"      - num_local_experts: {mlp.block_sparse_moe.num_local_experts}")

    print("=" * 60 + "\n")


def load_model_progressive(model_path, tp_size, rank, device, dtype=torch.float32):
    """
    Load large MoE model progressively to avoid OOM during TP conversion.

    For DeepSeek-V2-Lite (16B params), we:
    1. Load model config first
    2. Load model on CPU with memory-efficient options
    3. Convert to TP layer by layer (garbage collect after each layer)
    4. Move to device after conversion
    """
    import gc
    from transformers import AutoConfig

    if rank == 0:
        print(f"\nLoading model progressively to avoid OOM...")

    # Load config first
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    if rank == 0:
        print(f"Model config loaded: {config.__class__.__name__}")
        print(f"  - Hidden size: {config.hidden_size if hasattr(config, 'hidden_size') else 'N/A'}")
        print(f"  - Num layers: {config.num_hidden_layers if hasattr(config, 'num_hidden_layers') else 'N/A'}")
        if hasattr(config, 'num_local_experts'):
            print(f"  - Num experts: {config.num_local_experts}")

    # Load model with low CPU memory option
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")
        print(f"Expected parameters per TP rank: {total_params // tp_size:,}")

    return model


def main():
    """Main training function"""
    args = get_args()

    # Setup distributed
    rank, world_size, device = setup_npu()
    args.world_size = world_size

    # Calculate TP rank
    if world_size == args.tp_size:
        tp_rank = rank  # Pure TP
        use_ddp = False
    else:
        dp_size = world_size // args.tp_size
        dp_rank = rank // args.tp_size
        tp_rank = rank % args.tp_size
        use_ddp = True

    # Only print from rank 0
    if rank == 0:
        print(f"\n{'='*60}")
        print("Tensor Parallel Training - MoE Models")
        print(f"{'='*60}")
        print(f"Model: {args.model_path}")
        print(f"TP Size: {args.tp_size}")
        print(f"World Size: {world_size}")
        print(f"Batch Size: {args.batch_size}")
        print(f"Max Length: {args.max_length}")
        print(f"Device: NPU")
        print(f"Aux Loss Coef: {args.aux_loss_coef}")
        if use_ddp:
            print(f"Hybrid TP+DDP: TP={args.tp_size}, DP={dp_size}")
        print(f"{'='*60}\n")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model progressively for large MoE models
    model_dtype = torch.float32  # Use float32 for NPU stability

    import gc
    if rank == 0:
        print(f"Loading model on CPU first (progressive loading to avoid OOM)...")

    model = load_model_progressive(
        args.model_path,
        args.tp_size,
        rank,
        device,
        dtype=model_dtype
    )

    # Inspect model structure
    inspect_model_moe(model, rank)

    # Convert to TP BEFORE moving to device
    # This is important for MoE models - we shard experts first
    if args.tp_size > 1:
        if rank == 0:
            print(f"\nApplying tensor parallelism (tp_size={args.tp_size})...")
        model = convert_to_tp(model, tp_size=args.tp_size, rank=tp_rank)

        # Now move to device after TP conversion (each rank has smaller model)
        model.to(device)

        # Clean up CPU memory
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        if rank == 0:
            print(f"Tensor parallelism applied!")
            print(f"Rank {rank} -> TP rank {tp_rank}")

            if getattr(model, '_is_moe', False):
                print(f"MoE architecture detected and converted!")
    # Move to device after TP/DDP conversion
    if args.tp_size == 1:
        model.to(device)

    # Disable KV cache for training (fixes compatibility with newer transformers)
    model.config.use_cache = False

    # NPU compilation warmup for large models
    # This triggers JIT compilation on a small batch before actual training
    if rank == 0:
        print(f"\nRunning NPU compilation warmup...")
    model.eval()
    with torch.no_grad():
        warmup_input = torch.randint(0, tokenizer.vocab_size, (1, 8), device=device)
        try:
            _ = model(input_ids=warmup_input)
            if rank == 0:
                print(f"NPU compilation warmup complete!")
        except Exception as e:
            if rank == 0:
                print(f"Warmup failed (non-critical): {e}")
    model.train()

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    # Scheduler
    num_training_steps = 50  # Simplified for testing
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Prepare data
    dataloader = prepare_synthetic_data(args, tokenizer, rank)

    # Training settings
    model.train()

    if rank == 0:
        print(f"\nStarting training...")
        print(f"Total steps: {num_training_steps}")
        print(f"Log interval: {args.log_interval}")

    # Training loop
    step = 0
    start_time = time.time()
    total_aux_loss = 0.0

    for epoch in range(args.epochs):
        dataloader.sampler.set_epoch(epoch)

        for batch_idx, (input_ids, labels) in enumerate(dataloader):
            # Move to device
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            # Forward pass (disable KV cache for training compatibility)
            outputs = model(input_ids=input_ids, labels=labels, use_cache=False)
            loss = outputs.loss

            # Collect auxiliary losses from MoE layers
            aux_loss = torch.zeros(1, device=device)
            for name, module in model.named_modules():
                if hasattr(module, 'last_aux_loss'):
                    aux_loss += module.last_aux_loss

            total_aux_loss += aux_loss.item()

            # Combine losses
            total_loss = loss + args.aux_loss_coef * aux_loss

            # Backward pass
            total_loss.backward()

            # Sync gradients across TP ranks
            if args.tp_size > 1:
                sync_gradients_tp(model, args.tp_size)

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            # Logging
            if rank == 0 and step % args.log_interval == 0:
                lr = scheduler.get_last_lr()[0]
                loss_value = loss.item()
                aux_value = aux_loss.item()
                elapsed = time.time() - start_time
                tokens_per_sec = args.batch_size * args.max_length * (step + 1) / elapsed if step > 0 else 0
                print(f"Step {step}/{num_training_steps} | "
                      f"Loss: {loss_value:.4f} | "
                      f"Aux: {aux_value:.6f} | "
                      f"LR: {lr:.2e} | "
                      f"Tokens/sec: {tokens_per_sec:.0f}")

            # Save checkpoint
            if rank == 0 and step % 25 == 0 and step > 0:
                save_path = os.path.join(args.save_path, f"checkpoint_step_{step}")
                os.makedirs(save_path, exist_ok=True)

                save_path_rank = os.path.join(save_path, f"rank_{rank}")
                os.makedirs(save_path_rank, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(save_path_rank, "model.pt"))
                tokenizer.save_pretrained(save_path_rank)
                print(f"[Rank {rank}] Saved checkpoint to {save_path_rank}")

            step += 1
            if step >= num_training_steps:
                break

        if step >= num_training_steps:
            break

    # Final save
    if rank == 0:
        avg_aux_loss = total_aux_loss / num_training_steps
        print(f"\nAverage auxiliary loss: {avg_aux_loss:.6f}")

        final_path = os.path.join(args.save_path, "final_model")
        os.makedirs(final_path, exist_ok=True)
        save_path_rank = os.path.join(final_path, f"rank_{rank}")
        os.makedirs(save_path_rank, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(save_path_rank, "model.pt"))
        tokenizer.save_pretrained(save_path_rank)
        print(f"\n[Rank {rank}] Final model saved to {save_path_rank}")

    # Cleanup
    dist.destroy_process_group()

    if rank == 0:
        print("\nTraining completed!")


if __name__ == "__main__":
    main()

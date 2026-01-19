#!/usr/bin/env python3
"""
Pipeline Parallel training on Ascend NPU
Splits model stages across multiple NPUs for micro-batch parallelism
"""

import os
import time
import argparse
import torch
import torch_npu
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup

# Set CANN environment
os.environ.setdefault("HCCL_CONNECT_TIMEOUT", "1200")


def setup_distributed():
    """Initialize distributed training"""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))

    if world_size > 1:
        backend = "hccl"
        dist.init_process_group(backend=backend)
        torch_npu.set_device(local_rank)
    else:
        torch_npu.set_device(0)

    return local_rank, world_size, rank, torch.device(f"npu:{local_rank}")


def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_args():
    parser = argparse.ArgumentParser(description="PP Training on NPU")
    parser.add_argument("--model_name", type=str, default="/home/sd/npu_train/models/Qwen-Qwen2.5-14B-Instruct",
                        help="Model path")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size per device (default: 2)")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Max sequence length (default: 512)")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Number of epochs (default: 2)")
    parser.add_argument("--lr", type=float, default="1e-5",
                        help="Learning rate (default: 1e-5)")
    parser.add_argument("--warmup_steps", type=int, default=50,
                        help="Warmup steps (default: 50)")
    parser.add_argument("--log_interval", type=int, default=5,
                        help="Log every N steps (default: 5)")
    parser.add_argument("--save_path", type=str, default="./output_pp",
                        help="Model save path (default: ./output_pp)")
    parser.add_argument("--max_samples", type=int, default=5000,
                        help="Max training samples (default: 5000)")
    parser.add_argument("--pp_size", type=int, default=4,
                        help="Pipeline parallel size (default: 4)")
    parser.add_argument("--micro_batch_size", type=int, default=1,
                        help="Micro batch size for PP (default: 1)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2,
                        help="Gradient accumulation steps (default: 2)")
    return parser.parse_args()


def prepare_dataset(args, tokenizer, rank, world_size):
    """Prepare dataset with DistributedSampler"""
    print(f"[Rank {rank}] Preparing dataset with {args.max_samples} samples")

    sample_texts = [
        "The quick brown fox jumps over the lazy dog. " * 10,
        "Machine learning is a subset of artificial intelligence. " * 10,
        "Deep learning models can learn complex patterns from data. " * 10,
        "Neural networks are inspired by the human brain structure. " * 10,
        "Natural language processing enables computers to understand text. " * 10,
        "Transformers have revolutionized the field of NLP. " * 10,
        "Attention mechanisms allow models to focus on relevant parts. " * 10,
        "Pre-trained language models can be fine-tuned for various tasks. " * 10,
    ]

    texts = [sample_texts[i % len(sample_texts)] for i in range(args.max_samples)]

    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=args.max_length,
        padding="max_length",
        return_tensors="pt",
    )

    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings
        def __len__(self):
            return len(self.encodings["input_ids"])
        def __getitem__(self, idx):
            return {key: val[idx] for key, val in self.encodings.items()}

    dataset = SimpleDataset(tokenized)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )

    return dataset, sampler


def collate_fn(batch):
    """Custom collate function"""
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": input_ids.clone(),
    }


class PipelineParallelModel(torch.nn.Module):
    """Simple pipeline parallel wrapper for demonstration"""

    def __init__(self, model, pp_size, rank, world_size):
        super().__init__()
        self.model = model
        self.pp_size = pp_size
        self.rank = rank
        self.world_size = world_size

        # Split model layers into stages
        # This is a simplified version - real PP requires more complex implementation
        num_layers = len(model.model.layers)
        stage_size = num_layers // pp_size
        start_layer = (rank * stage_size)
        end_layer = start_layer + stage_size if rank < pp_size - 1 else num_layers

        print(f"[Rank {rank}] PP stage: layers {start_layer}-{end_layer} (total: {num_layers})")

        # Keep only this rank's layers
        self.model.model.layers = torch.nn.ModuleList(
            model.model.layers[start_layer:end_layer]
        )

        # First stage needs embedding, last stage needs LM head
        self.has_embedding = (rank == 0)
        self.has_lm_head = (rank == pp_size - 1)

        if not self.has_embedding:
            self.model.model.embed_tokens = None
        if not self.has_lm_head:
            self.model.lm_head = None

    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass for pipeline parallel"""
        hidden_states = None

        # First stage: embedding
        if self.has_embedding:
            hidden_states = self.model.model.embed_tokens(input_ids)
            hidden_states = self.model.model(drop_before_ln=False, hidden_states=hidden_states)

        # Middle stages: transformer layers
        for layer in self.model.model.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)

        # Last stage: LM head
        if self.has_lm_head:
            hidden_states = self.model.model.norm(hidden_states)
            logits = self.model.lm_head(hidden_states)

            if labels is not None:
                # Shift labels for causal LM
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100
                )
                return {"loss": loss, "logits": logits}

            return {"logits": logits}

        return {"hidden_states": hidden_states}


def train(args, local_rank, world_size, rank, device):
    """Main training function"""
    print(f"[Rank {rank}] Starting PP training, pp_size={args.pp_size}, device={device}")

    # Load tokenizer and model
    print(f"[Rank {rank}] Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    # Wrap with pipeline parallel
    if args.pp_size > 1:
        model = PipelineParallelModel(model, args.pp_size, rank, world_size)

    model = model.to(device)

    # Print model info (only rank 0)
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[Rank {rank}] Trainable parameters: {total_params:,}")
        print(f"[Rank {rank}] PP size: {args.pp_size}")

    # Prepare dataset
    train_dataset, sampler = prepare_dataset(args, tokenizer, rank, world_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs // args.gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )

    if rank == 0:
        print(f"\n{'='*50}")
        print(f"PP Training Configuration:")
        print(f"  World size: {world_size}")
        print(f"  PP size: {args.pp_size}")
        print(f"  Batch size per device: {args.batch_size}")
        print(f"  Micro batch size: {args.micro_batch_size}")
        print(f"  Effective batch size: {args.batch_size * world_size * args.gradient_accumulation_steps}")
        print(f"  Epochs: {args.epochs}")
        print(f"  Total steps: {total_steps}")
        print(f"  Learning rate: {args.lr}")
        print(f"{'='*50}\n")

    # Training loop
    model.train()
    global_step = 0
    total_loss = 0
    start_time = time.time()

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        epoch_loss = 0
        epoch_start = time.time()

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss / args.gradient_accumulation_steps

            # Backward pass
            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1

                if rank == 0 and global_step % args.log_interval == 0:
                    avg_loss = total_loss / global_step
                    elapsed = time.time() - start_time
                    samples_per_sec = (global_step * args.batch_size * world_size) / elapsed

                    print(f"Epoch {epoch+1}/{args.epochs} | "
                          f"Step {step+1}/{len(train_loader)} | "
                          f"Loss: {loss.item() * args.gradient_accumulation_steps:.4f} | "
                          f"Avg Loss: {avg_loss:.4f} | "
                          f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                          f"Speed: {samples_per_sec:.1f} samples/s")

            total_loss += loss.item() * args.gradient_accumulation_steps
            epoch_loss += loss.item() * args.gradient_accumulation_steps

        if rank == 0:
            epoch_time = time.time() - epoch_start
            avg_epoch_loss = epoch_loss / len(train_loader)
            print(f"\n>>> Epoch {epoch+1} completed in {epoch_time:.1f}s | "
                  f"Avg Loss: {avg_epoch_loss:.4f}\n")

    if rank == 0:
        total_time = time.time() - start_time
        print(f"\n{'='*50}")
        print(f"Training completed!")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Final avg loss: {total_loss/global_step:.4f}")
        print(f"{'='*50}\n")

        # Note: For PP, we only save from rank 0
        # In practice, you'd need to gather all stages or save checkpoint properly
        os.makedirs(args.save_path, exist_ok=True)
        # For PP, save only the model from rank 0 (simplified)
        # Real implementation would need to save and restore all stages
        tokenizer.save_pretrained(args.save_path)
        print(f"[Rank {rank}] Tokenizer saved to {args.save_path}")
        print(f"[Rank {rank}] Note: PP model saving requires stage coordination")

    return model


if __name__ == "__main__":
    args = get_args()
    local_rank, world_size, rank, device = setup_distributed()
    try:
        train(args, local_rank, world_size, rank, device)
    finally:
        cleanup_distributed()

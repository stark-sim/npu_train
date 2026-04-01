# TP Checkpoint Tool

## Purpose

This toolset addresses the current custom TP checkpoint gap in this repository:
- TP training scripts used to save only `rank_0`, even though the intent was to save per-rank shards.
- There was no offline tool to merge TP rank shards back into a full HuggingFace-style model.
- There was no way to re-shard an existing TP checkpoint to a different `tp_size`.
- There was no built-in resume path for TP optimizer/scheduler state.

## What Changed

- Added `npu_parallel/checkpoint_utils.py` for rank-wise TP checkpoint saving, loading, and shard writing.
- Added `tools/tp_checkpoint.py` with three subcommands:
  - `inspect`: inspect a rank-sharded TP checkpoint directory
  - `export`: merge rank shards and export a HuggingFace-compatible model directory
  - `reshard`: merge rank shards and write a new TP checkpoint at a different `tp_size`
- Added `tests/test_tp_checkpoint_tool.py` as a CPU-only merge/reshard smoke test.
- Added `tests/test_tp_checkpoint_resume.py` as a CPU-only save/load resume smoke test.
- Added `tests/test_tp_checkpoint_optimizer_reshard.py` as a CPU-only optimizer-state reshard smoke test.
- Updated TP training scripts to:
  - save one shard per rank
  - save rank-local optimizer/scheduler state
  - support `--resume_from` for TP checkpoint resume
- Added name-based optimizer-state restore so resume does not rely on parameter order being identical.

## Current Scope

Supported now:
- Rank-local TP checkpoint saving
- Rank-local optimizer/scheduler state saving
- Resume from TP checkpoints in `train_tp_custom.py` and `train_tp_moe.py`
- Name-based optimizer-state restore for the current TP scripts
- Merging column-parallel and row-parallel shards
- Replicated parameter handling
- Last-rank-only parameter handling (for row-parallel bias patterns)
- Best-effort contiguous expert remapping for evenly sharded MoE experts
- Resharding TP checkpoints to a new `tp_size`
- Best-effort optimizer-state resharding for the repository's current single-group AdamW usage

Not yet implemented:
- General optimizer-state resharding for arbitrary multi-group optimizer layouts
- Full resume tooling after a `reshard` across all possible model families
- Strong guarantees for all MoE variants beyond the current contiguous-expert assumptions

## Usage

Inspect a TP checkpoint:

```bash
python3 tools/tp_checkpoint.py inspect /path/to/checkpoint_dir
```

Export a merged HuggingFace model:

```bash
python3 tools/tp_checkpoint.py export /path/to/checkpoint_dir \
  --model-path /path/to/original/model \
  --output-dir /path/to/output_dir \
  --trust-remote-code
```

Reshard to a new TP size:

```bash
python3 tools/tp_checkpoint.py reshard /path/to/checkpoint_dir \
  --model-path /path/to/original/model \
  --output-dir /path/to/new_tp_checkpoint \
  --target-tp-size 8 \
  --copy-tokenizer \
  --include-optimizer-state \
  --trust-remote-code
```

Resume TP training:

```bash
torchrun --nproc_per_node=4 examples/train_tp_custom.py \
  --model_path /path/to/model \
  --tp_size 4 \
  --resume_from /path/to/checkpoint_step_25
```

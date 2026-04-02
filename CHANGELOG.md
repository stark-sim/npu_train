# Changelog

All notable changes to this project will be documented in this file.

## [910A Phase Complete] - 2026-04-01

### Added

#### NPU Compatibility Layer
- `npu_parallel/npu_compat.py` - Full compatibility layer with policy controls
  - `safe_softmax`, `safe_topk`, `safe_any` wrappers with automatic fallback
  - Policy modes: `fallback` (default), `warn`, `strict`
  - Fallback statistics and performance counters
  - Error signature mapping for 910A/CANN/HCCL failures
- `tools/npu_compat_report.py` - Runtime compatibility reporting
- `tools/npu_compat_benchmark.py` - Microbenchmark raw/safe/fallback paths
- `tools/npu_compat_log_analyze.py` - Offline log triage and signature mining
- `tools/repro_storage_offset_warning.py` - Minimal repro for baseline warning

#### TP Checkpoint Tooling
- `npu_parallel/checkpoint_utils.py` - Checkpoint save/load utilities
- `tools/tp_checkpoint.py` - Inspect, export, and reshard TP checkpoints
- Rank-local optimizer/scheduler state saving
- Name-based optimizer-state restore
- Optimizer-state resharding for single-group TP
- `--resume_from` support in training scripts

#### Training Features
- `--max_steps` for short smoke runs
- `--skip_save` to avoid large checkpoints during validation
- `--compat_report_file` for post-training JSON export
- `--compat_policy` for runtime policy control

#### Tests
- `tests/test_npu_compat_layer.py` - Compatibility layer smoke tests
- `tests/test_npu_compat_log_analyze.py` - Log analysis tests
- `tests/test_tp_attention_compat.py` - TP attention compatibility tests
- `tests/test_tp_checkpoint_tool.py` - Checkpoint tool tests
- `tests/test_tp_checkpoint_resume.py` - Checkpoint resume tests
- `tests/test_tp_checkpoint_optimizer_reshard.py` - Optimizer resharding tests

#### Documentation
- `docs/project-status/` - Comprehensive project documentation
  - `COMPLETION_REPORT.md` - Handoff document
  - `SUMMARY_REMOTE_20260331.md` - Remote validation summary
  - `storage_offset_diagnosis.md` - Root cause analysis
  - `stage-results.zh.md/en.md` - Full status summaries
- `memory-bank/` - Cross-session context for AI agents

### Changed

- `npu_parallel/tp_layers.py` - Autograd-aware `all_reduce` wrapper
- `npu_parallel/tp_attention.py` - Compatibility wrappers for attention paths
- `npu_parallel/tp_moe.py` - Compatibility wrappers for MoE router
- `examples/train_tp_custom.py` - Added smoke-run flags and compat report export
- `examples/train_tp_moe.py` - Added smoke-run flags and compat report export
- `CLAUDE.md` - Updated with current architecture

### Fixed

- `allreduce autograd kernel was not registered` warning fixed with autograd-aware collectives
- `storage_offset ... untrustworthy64/128` warning identified as baseline behavior, not TP regression

### Validation

Real-device validation on Ascend 910A (8-card):
- 4-card TP smoke: ✅ Completed
- 4-card MoE smoke: ✅ Completed
- 0 fallback in `softmax` and `any` paths
- Compatibility layer fully functional

### Removed

- Python cache files from git tracking

## [Previous] - 2026-02-05

### Added
- Real-data training support with HuggingFace Arrow datasets
- `run_real_training.sh` script

## [Previous] - 2026-01-29

### Added
- DeepSeek-V2-Lite MoE TP training support

## [Previous] - 2026-01-28

### Added
- TP implementation with gradient checkpointing
- DataLoader tuning optimizations

## [Previous] - 2026-01-22

### Added
- Initial Ascend NPU training baseline
- Single NPU, DDP, and PP training entry points

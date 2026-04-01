# Progress

## Completed Features
- [x] Single-NPU, DDP, and PP training entry points for Ascend NPU.
- [x] Custom TP core layers and HCCL-compatible communication helpers.
- [x] HuggingFace model conversion into TP-compatible modules.
- [x] DeepSeek-family support, including DeepSeek-V2-Lite handling.
- [x] MoE TP support with NPU-specific operator compatibility workarounds.
- [x] Real-data training support with offline HuggingFace Arrow datasets.
- [x] Checkpoint save/resume support for real-data training scripts.
- [x] Benchmark and test scripts for conversion, HCCL, NPU, and DeepSeek/MoE paths.
- [x] Project status documentation under `docs/project-status/`.
- [x] Memory Bank initialization for Claude Code and OpenAI Codex.
- [x] Rank-wise TP checkpoint saving for custom TP scripts.
- [x] Offline TP checkpoint inspect/export tool for merging rank shards.
- [x] TP checkpoint reshard support for rewriting checkpoints to a new `tp_size`.
- [x] Rank-local optimizer/scheduler state saving and resume support for TP scripts.
- [x] Optimizer-state resharding for the current single-group TP training setup.
- [x] Name-based optimizer-state restore for TP checkpoint resume.
- [x] Initial NPU compatibility layer with operation probes and safe fallback wrappers.
- [x] NPU compatibility policy controls (`fallback|warn|strict`) and fallback-stat tracking.
- [x] Compatibility policy/report integration in TP scripts and `tools/npu_compat_report.py`.
- [x] CPU smoke tests for compatibility-policy and fallback-stat behavior.
- [x] NPU compatibility report now includes error-signature patterns and recommended actions.
- [x] Per-op compatibility perf counters added (attempt/success/failure/fallback + timing).
- [x] NPU compatibility log-analysis helpers added for runtime log triage and unknown-signature candidate extraction.
- [x] Added `tools/npu_compat_log_analyze.py` with smoke test coverage.
- [x] Log-analysis output now includes an explicit outcome objective and signature-update plan.
- [x] Log-analysis now emits reviewable `_ERROR_SIGNATURES` patch templates.
- [x] `tools/npu_compat_log_analyze.py` supports recursive directory scans for offline log batches.
- [x] Critical TP/MoE/attention topk/softmax hot paths now use compatibility wrappers.
- [x] Added TP attention compatibility smoke coverage.
- [x] Added `tools/npu_compat_benchmark.py` for raw/safe/forced-fallback microbenchmarks.
- [x] Validated the current compatibility stack on a real 8-card 910A environment.
- [x] Stored remote validation artifacts under `.context/remote-npu-compat-20260331/`.
- [x] Refined signature matching and log filtering using real remote logs.
- [x] Added `--max_steps` short-smoke support to TP training scripts.
- [x] Completed a real-device 2-step TP smoke training run on 910A.
- [x] Captured a larger-shape NPU benchmark snapshot.
- [x] Cleaned the temporary remote smoke checkpoint after harvesting logs.
- [x] Added autograd-aware TP forward `all_reduce` helper.
- [x] Verified 4-card TP smoke training on 910A completes without the prior `allreduce autograd kernel` warning.
- [x] Verified 4-card MoE smoke training on 910A completes and archived the log.
- [x] Added `--skip_save` smoke-run support to TP training scripts to avoid large checkpoints during real-device validation.
- [x] Added `--compat_report_file` post-training JSON export to TP training scripts.
- [x] Validated post-training compatibility reporting on a real 4-card MoE smoke and captured zero fallback in the exercised `any` and `softmax` paths.
- [x] Confirmed the isolated `TPMoERouter` path hits `safe_topk` and `safe_softmax` on real NPU with zero fallback.
- [x] Root-caused `storage_offset ... untrustworthy64/128` warning to original Qwen backward (not TP-specific); archived findings.
- [x] Updated log-analysis to ignore benign `storage_offset` and `ne 64-bit` warnings.
- [x] Added `tools/repro_storage_offset_warning.py` minimal repro script for environment validation.
- [x] Reproduced `storage_offset ... untrustworthy64` on the original non-TP Qwen backward path, confirming it is not TP-specific.
- [x] Narrowed `storage_offset` to backward execution rather than forward-only execution on the original Qwen model.
- [x] Filtered empirically benign `storage_offset/untrustworthy` and `ne 64-bit` warnings from offline compatibility log analysis.
- [x] Shrunk the `storage_offset` repro to the original Qwen decoder-layer and `self_attn` backward path on real NPU.
- [x] Verified the custom `TPAttention` backward smoke does not emit the same `storage_offset` warning at a matched shape.

## In Progress
- [x] Final documentation and handoff for Ascend 910A phase.

## Git Commit Summary
| Commit | Description |
|--------|-------------|
| b005376 | NPU compatibility layer + checkpoint utils |
| 3c43d8f | Diagnostic tools (6 tools) |
| 007dd60 | Comprehensive test coverage (6 tests) |
| b743aa2 | Project documentation (11 docs) |
| 154015f | Memory bank for agent continuity |
| 3428834 | Autograd-aware collectives integration |
| ae8c80f | Smoke-run flags for TP training |
| ba80d06 | CLAUDE.md architecture update |
| dcc45c6 | Cleanup pycache files |

**Total**: 9 commits, 109 files tracked, 59 key source/doc/test files

## Known Issues
- Hybrid TP+DDP group management is not fully completed in the custom TP training path.
- TP checkpoint export, reshard, same-topology resume, and current single-group optimizer-state resharding exist now, but broader optimizer layouts are not yet supported.
- DeepSeek-V2 MLA attention TP remains partial in the current implementation.
- Some MoE communication paths are correctness-first and not yet fully optimized.
- Long-run robustness tooling is still basic relative to likely 910A operational failure modes.
- Fallback behavior is tracked now, but fallback performance impact is not yet quantified.
- Perf counters exist for fallback paths, but end-to-end throughput impact by workload is not yet benchmarked.
- The remaining `storage_offset ... untrustworthy64/128` warning still appears in successful runs on 910A, but current evidence shows it is a baseline backward/compiler warning rather than a TP-only regression.

## Milestones

| Milestone | Status | Target Date |
|-----------|--------|-------------|
| Initial Ascend NPU training baseline | Completed | 2026-01-22 |
| TP implementation and baseline optimization | Completed | 2026-01-28 |
| DeepSeek-V2-Lite MoE TP support | Completed | 2026-01-29 |
| Real-data training workflow | Completed | 2026-02-05 |
| Project-context and recommendation consolidation | Completed | 2026-03-26 |
| TP checkpoint tooling (phase 1: save + inspect/export) | Completed | 2026-03-26 |
| TP checkpoint tooling (phase 2: reshard) | Completed | 2026-03-26 |
| TP checkpoint tooling (phase 3: optimizer state + resume integration) | Completed | 2026-03-26 |
| TP checkpoint tooling (phase 4: optimizer-state resharding) | Completed | 2026-03-26 |
| NPU compatibility layer (phase 1: wrappers + MoE integration) | Completed | 2026-03-26 |
| NPU compatibility layer (phase 2: policy + stats + script wiring) | Completed | 2026-03-26 |
| NPU compatibility layer (phase 3: signature mapping + perf counters) | Completed | 2026-03-26 |
| NPU compatibility layer (phase 4: log-analysis tooling) | Completed | 2026-03-26 |
| NPU compatibility layer (phase 5: patch templates + broader wrapper coverage) | Completed | 2026-03-31 |
| NPU compatibility layer (phase 6: real-device validation + benchmark) | Completed | 2026-03-31 |

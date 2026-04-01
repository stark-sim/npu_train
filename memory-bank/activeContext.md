# Active Context

## Current Focus
The current focus is to keep the low-CANN 910A stack runnable on real hardware, now that both custom TP and MoE short smokes have been validated, and to turn residual device warnings into targeted diagnostics rather than broad rewrites.

## Recent Changes
- [2026-04-01] Root-caused the `storage_offset ... untrustworthy64/128` warning: traced to original Qwen self_attn backward (not TP-specific). Archived findings in `.context/remote-npu-compat-20260331/SUMMARY.md:69` and `storage_offset_diagnosis.md:1`. Custom TPAttention backward at matched shape is clean.
- [2026-04-01] Updated log-analysis to ignore benign `storage_offset` and `ne 64-bit` warnings in `npu_compat.py:54` and `:327`.
- [2026-04-01] Added smoke test coverage for ignored warning patterns in `tests/test_npu_compat_log_analyze.py:33`.
- [2026-04-01] Created `tools/repro_storage_offset_warning.py` as a minimal reproducible test for environment validation and upstream issue reporting.
- [2026-03-26] Reviewed the repository history, recent documents, and official MindSpore materials to compare current project value against official capabilities.
- [2026-03-26] Created `docs/project-status/` and added Chinese/English long and short status summaries plus a recommendation document for meaningful next work.
- [2026-03-26] Identified the most differentiated next steps as TP checkpoint merge/reshard/export, NPU compatibility-layer systematization, 910A profiling/autotune, and long-run auto-recovery.
- [2026-03-26] Implemented the first TP checkpoint milestone: fixed rank-wise shard saving for custom TP scripts and added an offline `tools/tp_checkpoint.py` inspect/export utility.
- [2026-03-26] Extended the TP checkpoint tool with `reshard` support so existing TP checkpoints can be rewritten for a new `tp_size`.
- [2026-03-26] Added rank-local optimizer/scheduler checkpointing plus `--resume_from` support for the custom TP and TP-MoE training scripts.
- [2026-03-26] Added optimizer-state resharding for the current single-group TP setup and name-based optimizer restore.
- [2026-03-26] Added the first NPU compatibility-layer module (`npu_compat`) and integrated safe fallback helpers into DeepSeek MoE TP forward.
- [2026-03-26] Extended `npu_compat` with compatibility policy modes (`fallback|warn|strict`), fallback statistics, and runtime error categorization.
- [2026-03-26] Added policy controls to `tools/npu_compat_report.py`, `examples/train_tp_custom.py`, and `examples/train_tp_moe.py`.
- [2026-03-26] Extended CPU smoke tests to cover policy behavior and fallback-stat reporting.
- [2026-03-26] Added error-signature mapping and recommendation hints in `npu_compat` reports for faster 910A/CANN failure triage.
- [2026-03-26] Added lightweight per-op perf counters (attempt/success/failure/fallback + timing) in compatibility reports.
- [2026-03-26] Added log-analysis helpers (`analyze_log_text`, `analyze_log_file`) to convert runtime logs into class counts and unknown-signature candidate tokens.
- [2026-03-26] Added CLI `tools/npu_compat_log_analyze.py` and smoke test `tests/test_npu_compat_log_analyze.py`.
- [2026-03-26] Log-analysis now emits explicit outcome objectives and a reviewable signature-update plan (no auto-apply).
- [2026-03-31] Added reviewable `_ERROR_SIGNATURES` patch-template generation from log-analysis results.
- [2026-03-31] Extended `tools/npu_compat_log_analyze.py` to scan log directories recursively for offline batch triage.
- [2026-03-31] Extended compatibility wrappers into `TPMoERouter` and `TPAttention._manual_attention`, covering the remaining critical topk/softmax hot paths.
- [2026-03-31] Added `tests/test_tp_attention_compat.py` and extended log-analysis tests for patch-template and directory-scan flows.
- [2026-03-31] Added `tools/npu_compat_benchmark.py` to benchmark raw vs safe vs forced-fallback paths on CPU/NPU.
- [2026-03-31] Synced the current tree to a remote staging directory and validated on a real 8-card 910A environment.
- [2026-03-31] Collected remote artifacts in `.context/remote-npu-compat-20260331/`, including NPU compatibility report, benchmark results, and recursive log-analysis output.
- [2026-03-31] Refined signature and log-noise filtering using real remote logs (`ERR99999`, `EI9999`, `storage_offset`, `unable to mmap`, `ReduceAny`).
- [2026-03-31] Added `--max_steps` to `examples/train_tp_custom.py` and `examples/train_tp_moe.py` for short real-device smoke runs.
- [2026-03-31] Completed a 2-step single-card TP smoke run on 910A with Qwen2.5-1.5B and captured the log under `.context/remote-npu-compat-20260331/train_tp_custom_smoke.log`.
- [2026-03-31] Ran a larger-shape NPU compatibility benchmark and recorded the result in `.context/remote-npu-compat-20260331/compat_benchmark_npu_large.json`.
- [2026-03-31] Cleaned the remote 18G smoke checkpoint and kept only small artifacts/logs.
- [2026-03-31] Replaced forward-path TP `all_reduce` with an autograd-aware helper in `npu_parallel/tp_layers.py` and `npu_parallel/tp_attention.py`.
- [2026-03-31] Re-ran the 4-card TP smoke on 910A and confirmed the prior `allreduce autograd kernel` warning disappeared.
- [2026-03-31] Captured the successful 4-card smoke log under `.context/remote-npu-compat-20260331/train_tp_custom_4card_smoke_v2.log` and cleaned the remote checkpoint.
- [2026-03-31] Completed a real 4-card MoE smoke on 910A with `examples/train_tp_moe.py`, archived the log at `.context/remote-npu-compat-20260331/train_tp_moe_4card_smoke.log`, and confirmed the run completes end-to-end.
- [2026-03-31] Added `--skip_save` to `examples/train_tp_custom.py` and `examples/train_tp_moe.py` so scarce-device smoke runs can skip large final checkpoints.
- [2026-03-31] Added `--compat_report_file` to the TP training scripts and validated it on a real 4-card MoE smoke, producing `.context/remote-npu-compat-20260331/train_tp_moe_4card_smoke_compat.json`.
- [2026-03-31] Confirmed the remaining `storage_offset ... untrustworthy128` warning appears during MoE warmup but is non-blocking in the successful 4-card run; the recorded compat report shows zero runtime fallback in the exercised `any` and `softmax` paths.
- [2026-03-31] Ran a targeted `TPMoERouter` diagnostic on real `npu:0` and confirmed both `safe_softmax` and `safe_topk` execute on the primary NPU path with zero fallback; archived at `.context/remote-npu-compat-20260331/tp_moe_router_npu_diag.json`.
- [2026-04-01] Reproduced `storage_offset ... untrustworthy64` on the original non-TP Qwen2.5-1.5B model during a single-NPU training backward step, showing the warning is not unique to this project's TP conversion.
- [2026-04-01] Narrowed the warning boundary: eval forward and train forward-only runs do not emit `storage_offset`, while backward paths do; this points to a baseline backward-graph/compiler warning under the current torch-npu stack.
- [2026-04-01] Updated offline log triage to ignore empirically benign `storage_offset/untrustworthy` and `ne 64-bit low-performance` warnings so recursive scans stay focused on actionable failures.
- [2026-04-01] Shrunk the baseline repro from full-model backward to a single original Qwen decoder layer and then to the original Qwen `self_attn` branch; both still emit `storage_offset ... untrustworthy64`.
- [2026-04-01] Verified that the local custom `TPAttention` backward smoke on real `npu:0` does not emit the same `storage_offset` warning at the matched hidden-size/sequence shape.
- [2026-02-05] Added improved real-data training support, including offline HuggingFace Arrow dataset loading and `run_real_training.sh`.
- [2026-01-29] Added DeepSeek-V2-Lite MoE TP training support and follow-up fixes for custom step counts and infinite data looping.
- [2026-01-28] Established a performance baseline for NPU TP training and introduced practical optimizations such as gradient checkpointing and DataLoader tuning.

## Active Decisions
- Keep the project on `PyTorch + torch_npu` for the current environment rather than planning immediate MindSpore migration.
- Treat the main project value as legacy-platform adaptation and runnable engineering infrastructure, not generic parallel-training capability.
- Prioritize next work that is specific to 910A operational needs.
- Keep compatibility policy default as `fallback` for production continuity, while exposing `warn` and `strict` for diagnostics.
- Prefer `--skip_save` plus `--compat_report_file` for real-device smoke runs so validation stays cheap and leaves auditable runtime evidence.
- Treat the current `storage_offset` warning as a contained baseline backward/compiler diagnostic, not a release blocker, unless a future minimal repro shows a project-specific regression.

## Next Steps
- [x] Git commit all work (11 commits, 141 files)
- [x] Create completion report at `docs/project-status/COMPLETION_REPORT.md`
- [x] Remote validation artifacts synced to `.context/remote-npu-compat-20260331/`
- [ ] Push branch to remote when ready
- Ascend 910A phase complete.
- Collect actual long-run training logs and feed them into `tools/npu_compat_log_analyze.py` to iteratively update signature patterns.
- Benchmark compatibility fallback overhead on the real device before it is reclaimed.
- Improve profiling/autotune for 910A-specific training recipes.
- Consider long-run recovery tooling before investing further in generic parallel features.

## Important Patterns & Preferences
- Memory Bank should use English and the Basic profile.
- The project actively uses Claude Code and OpenAI Codex.
- Prefer practical, environment-specific engineering work over broad framework redesign.
- Avoid positioning project value as duplication of official MindSpore capabilities.

## Current Blockers
- Official newer framework capabilities do not map cleanly to the active `910A + lower CANN` environment.
- Some custom TP and MoE paths still favor correctness-first or simplified implementations.

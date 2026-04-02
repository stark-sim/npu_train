# NPU Compatibility Layer

## Purpose

This layer centralizes operation-level compatibility handling for `Ascend 910A + lower CANN + torch_npu`.
The goal is to avoid scattering ad-hoc fallbacks across TP/MoE code paths.

## What Changed

- Added `npu_parallel/npu_compat.py`:
  - runtime inspection (`runtime_info`)
  - operation probing (`supports_op`)
  - safe fallbacks (`safe_topk`, `safe_softmax`, `safe_has_any_tokens`, `safe_nonzero`)
  - compatibility policy controls (`set_compat_policy`, `get_compat_policy`) with `fallback|warn|strict`
  - fallback statistics (`get_fallback_stats`, `reset_fallback_stats`)
  - runtime error categorization (`classify_runtime_error`)
  - known-signature mapping and recommendation outputs (`known_error_signatures`, `recommended_action`)
  - lightweight per-op performance counters (`get_perf_counters`)
- Wired the layer into DeepSeek MoE NPU-compatible forward in `npu_parallel/tp_moe.py`.
- Exported helpers from `npu_parallel/__init__.py`.
- Added/extended CPU smoke test `tests/test_npu_compat_layer.py` for policy and fallback-stat behavior.
- Added `tests/test_npu_compat_log_analyze.py` for log-analysis helpers and CLI smoke coverage.
- Added policy option to report tooling: `tools/npu_compat_report.py --policy {fallback,warn,strict}`.
- Added log-analysis tooling: `tools/npu_compat_log_analyze.py <log_files...> --top-k N --min-count M --max-updates K`.
- Added NPU microbenchmark tooling: `tools/npu_compat_benchmark.py`.
- Log-analysis output now includes an explicit outcome objective and a signature-update plan (reviewable, non-auto-apply).
- Log-analysis can now emit a reviewable patch template for `_ERROR_SIGNATURES`, and the CLI can scan directories recursively for log-like files.
- Added policy option to TP training scripts:
  - `examples/train_tp_custom.py --compat_policy ...`
  - `examples/train_tp_moe.py --compat_policy ...`

## Scope

Supported now:
- Probe selected ops (`nonzero`, `any`, `topk`, `softmax`)
- Use safe fallback wrappers without changing training scripts
- Reuse the same helpers across modules
- Select compatibility policy per run (`fallback`, `warn`, `strict`)
- Capture fallback count and latest error class in compatibility report
- Emit signature patterns and recommendation hints in compatibility report
- Track per-op attempt/success/failure/fallback counts and timing
- Analyze runtime logs into class counts and unknown-signature candidate tokens
- Emit outcome objective and reviewable update plan so the purpose is explicit
- Emit reviewable `_ERROR_SIGNATURES` patch templates from log analysis
- Recursively scan saved log directories for offline triage

Not yet implemented:
- Auto-generated compatibility report by device/CANN version
- Full signature set from real 910A/CANN failure corpus (current mapping is seed set)
- Integration across every training/utility path (current focus is MoE and TP entry points)

## Next Step

Extend this layer into a small policy engine:
- enrich error-class mapping with real-world CANN/HCCL signatures
- benchmark end-to-end fallback overhead on real 910A workloads
- add lightweight perf counters to estimate fallback impact

## Remote Validation

- [2026-03-31] Validated on a real 8-card 910A environment (`torch 2.5.1`, `torch_npu 2.5.1`).
- `tools/npu_compat_report.py --device-type npu` reported `topk/softmax/nonzero/any` as available on NPU.
- `tools/npu_compat_benchmark.py` completed on NPU and produced artifact snapshots under `.context/remote-npu-compat-20260331/`.
- A 2-step single-card TP smoke training run completed on a real 910A using `examples/train_tp_custom.py --max_steps 2`.
- A 2-step 4-card TP smoke training run also completed successfully after replacing forward-path `all_reduce` with an autograd-aware helper.
- Observed microbenchmark results on `npu:0` (small run):
  - `softmax_raw`: ~0.031 ms
  - `softmax_safe`: ~0.026 ms
  - `softmax_safe_forced_fallback`: ~1.690 ms
  - `topk_raw`: ~0.288 ms
  - `topk_safe`: ~0.268 ms
  - `tp_attention_manual_safe`: ~0.245 ms
  - `tp_moe_router_safe`: ~0.347 ms
- Observed microbenchmark results on `npu:0` (larger run):
  - `softmax_raw`: ~0.026 ms
  - `softmax_safe`: ~0.025 ms
  - `softmax_safe_forced_fallback`: ~1.650 ms
  - `topk_raw`: ~0.388 ms
  - `topk_safe`: ~0.380 ms
  - `topk_safe_forced_fallback`: ~0.198 ms
  - `tp_attention_manual_safe`: ~0.277 ms
  - `tp_moe_router_safe`: ~0.345 ms
- Remote log analysis over the existing training/log directory reduced aggregate `unknown` lines from a noisy first pass to 224 across 166 log files after filtering and signature cleanup.

## Communication Fix

- Replaced direct forward-path `dist.all_reduce` in row-parallel TP outputs with an autograd-aware helper in `npu_parallel/tp_layers.py`.
- Re-ran the 4-card TP smoke and confirmed the previous `c10d::allreduce_ ... autograd kernel was not registered` warning no longer appears.
- The remaining warning during 4-card smoke is the existing `storage_offset ... untrustworthy64` compiler warning; training still completes.

# Remote NPU Compatibility Validation Summary

Date: 2026-03-31
Remote host: bms-hit-haom-jbgs
Remote stage dir: ~/npu_train_hamburg_stage
Ascend env: source /usr/local/Ascend/ascend-toolkit/set_env.sh
Conda env: npu_train

## Runtime
- torch: 2.5.1
- torch_npu: 2.5.1
- npu_available: true
- device_count: 8
- reported ops on NPU: any/nonzero/softmax/topk = true

## Benchmark highlights on npu:0
- softmax_raw: ~0.031 ms
- softmax_safe: ~0.026 ms
- softmax_safe_forced_fallback: ~1.690 ms
- topk_raw: ~0.288 ms
- topk_safe: ~0.268 ms
- topk_safe_forced_fallback: ~0.099 ms
- tp_attention_manual_safe: ~0.245 ms
- tp_moe_router_safe: ~0.347 ms

## Log analysis
- analyzed log files: 166
- class counts: acl_runtime=66, hccl_runtime=15, memory=3, shape_or_dtype=13, timeout=311, unknown=224
- unknown count dropped significantly after filtering/real-log signature cleanup
- no high-confidence aggregate patch template remained after the latest cleanup pass

## Remote tests run
- tests/test_npu_compat_layer.py
- tests/test_npu_compat_log_analyze.py
- tests/test_tp_attention_compat.py

## Artifacts
- compat_report_npu.json
- compat_benchmark_npu_smoke.json
- compat_benchmark_npu.json
- compat_log_analysis_remote.json
- test_npu_compat_layer.txt
- test_npu_compat_log_analyze.txt
- test_tp_attention_compat.txt

## 4-card TP smoke
- command: `torchrun --nproc_per_node=4 examples/train_tp_custom.py --tp_size 4 --max_steps 2 ...`
- status: completed
- outcome: `Step 1/2` reached, final checkpoint saved, training completed
- fix validated: prior `allreduce autograd kernel was not registered` warning no longer appears after the autograd-aware all_reduce change

## 4-card MoE smoke
- command: `torchrun --nproc_per_node=4 examples/train_tp_moe.py --tp_size 4 --max_steps 2 ...`
- status: completed
- outcome: training completed, final checkpoint saved, and log archived as `train_tp_moe_4card_smoke.log`
- residual warning: `storage_offset ... untrustworthy128` still appears during warmup but did not block training

## 4-card MoE smoke (skip-save + compat report)
- command: `torchrun --nproc_per_node=4 examples/train_tp_moe.py --tp_size 4 --max_steps 1 --skip_save --compat_report_file ...`
- status: completed
- outcome: training completed without writing a large checkpoint; compat JSON archived as `train_tp_moe_4card_smoke_compat.json`
- runtime compat result: `any` attempts=832 and `softmax` attempts=52 with zero fallback; no `topk` fallback was observed in this short run

## Targeted router diagnostic
- target: `TPMoERouter` on `npu:0` with random input
- result: both `safe_softmax` and `safe_topk` executed on the NPU primary path with zero fallback
- artifact: `tp_moe_router_npu_diag.json`

## storage_offset diagnosis
- original Qwen single-NPU eval forward: no `storage_offset` warning (`qwen_orig_warn.log`)
- original Qwen single-NPU training backward: warning reproduced (`qwen_orig_train_warn.log`)
- converted Qwen `convert_to_tp(tp_size=1)` single-NPU training backward: same warning reproduced (`qwen_tp_train_warn.log`)
- conclusion: the warning is present on the original backward path too, so it is not currently evidence of a TP-specific regression
- offline triage action: ignore `storage_offset/untrustworthy` and `ne 64-bit` warnings in `tools/npu_compat_log_analyze.py` / `npu_parallel.npu_compat` filtering
- original Qwen train forward-only: no `storage_offset` warning (`qwen_train_forward_only.log`)
- original Qwen backward-only and logits-backward: warning reproduced (`qwen_backward_only.log`, `qwen_logits_backward.log`)
- original Qwen single decoder layer and original `self_attn` branch: warning reproduced (`qwen_single_layer_backward.log`, `qwen_attn_mlp_backward.log`)
- custom TPAttention backward smoke at matched shape: no `storage_offset` warning (`tp_attention_backward.log`)

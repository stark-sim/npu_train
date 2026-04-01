# Remote NPU Validation Artifacts Index

**Date**: 2026-03-31 ~ 2026-04-01  
**Remote Host**: bms-hit-haom-jbgs  
**Remote Path**: `/home/sd/npu_train_hamburg_stage/`

---

## 📁 File Inventory

### Core Documentation (Manual)
| File | Description | Size |
|------|-------------|------|
| `SUMMARY.md` | Remote validation summary | ~4KB |
| `storage_offset_diagnosis.md` | Root cause analysis of storage_offset warning | ~2KB |

### NPU Compatibility Reports
| File | Description | Size |
|------|-------------|------|
| `compat_report_npu.json` | Initial NPU compatibility probe | 2KB |
| `compat_benchmark_npu.json` | Standard shape benchmark | 4KB |
| `compat_benchmark_npu_smoke.json` | Smoke benchmark | 4KB |
| `compat_benchmark_npu_large.json` | Large shape benchmark | 5KB |
| `compat_log_analysis_remote.json` | Log analysis of 166 files | 463KB |
| `train_tp_moe_4card_smoke_compat.json` | Post-training compat report | 4KB |

### Training Logs (Real 910A Runs)
| File | Description |
|------|-------------|
| `train_tp_custom_smoke.log` | 2-step single-card TP smoke |
| `train_tp_custom_4card_smoke.log` | 4-card TP smoke (v1) |
| `train_tp_custom_4card_smoke_v2.log` | 4-card TP smoke (v2, autograd fix) |
| `train_tp_moe_4card_smoke.log` | 4-card MoE smoke |
| `train_tp_moe_4card_smoke_skip_save.log` | 4-card MoE with skip_save |

### Diagnostic Logs (storage_offset Investigation)
| File | Description | Warning? |
|------|-------------|----------|
| `qwen_orig_warn.log` | Original Qwen eval forward | ❌ No |
| `qwen_orig_train_warn.log` | Original Qwen train backward | ✅ Yes |
| `qwen_tp_train_warn.log` | Converted Qwen TP train backward | ✅ Yes |
| `qwen_train_forward_only.log` | Train forward only | ❌ No |
| `qwen_backward_only.log` | Backward only | ✅ Yes |
| `qwen_logits_backward.log` | Logits-mean backward | ✅ Yes |
| `qwen_single_layer_backward.log` | Single decoder layer backward | ✅ Yes |
| `qwen_attn_mlp_backward.log` | Attention+MLP backward | ✅ Yes |
| `tp_attention_backward.log` | Custom TPAttention backward | ❌ No |

### Test Outputs
| File | Content |
|------|---------|
| `test_npu_compat_layer.txt` | "OK" |
| `test_npu_compat_log_analyze.txt` | "OK" |
| `test_tp_attention_compat.txt` | "OK" |
| `py_compile.txt` | Empty |
| `py_compile_train_tp_custom.txt` | Empty |

### Other Artifacts
| File | Description |
|------|-------------|
| `tp_moe_router_npu_diag.json` | TPMoERouter diagnostic on npu:0 |
| `fusion_result.json` | Op fusion analysis (40KB) |
| `train_tp_custom_smoke/` | Checkpoint directory (cleaned) |

---

## 🔑 Key Findings

### storage_offset Warning
- **Status**: Baseline torch_npu/compiler behavior, NOT TP-specific
- **Evidence**: Original Qwen self_attn backward reproduces, custom TPAttention clean
- **Action**: Ignored in log triage, documented in diagnosis file

### Compatibility Layer Validation
- `safe_softmax`: 0 fallback on real NPU
- `safe_topk`: 0 fallback on real NPU
- `safe_any`: 0 fallback on real NPU

### TP Training Validation
- 4-card TP smoke: ✅ Completed
- 4-card MoE smoke: ✅ Completed
- `allreduce autograd kernel` warning: ✅ Fixed (v2 log)

---

## 📥 Sync Status

| Source | Files | Status |
|--------|-------|--------|
| Local git | SUMMARY.md, storage_offset_diagnosis.md | ✅ Tracked |
| Remote artifacts | JSON reports, logs | ✅ Synced |

**Last Sync**: 2026-04-01

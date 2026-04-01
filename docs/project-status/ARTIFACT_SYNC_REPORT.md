# 远程验证工件同步报告

**日期**: 2026-04-01  
**远程主机**: bms-hit-haom-jbgs  
**本地分支**: stark-sim/real-moe-smoke

---

## 📥 同步概述

将远程 910A 服务器上的关键验证数据和诊断文件全部回传到本地 git 仓库，确保：
1. **机器无关性**: 不再依赖特定的 910A 机器访问
2. **完整归档**: 所有验证证据和诊断结论都在版本控制中
3. **可追溯性**: 未来的开发者可以查看原始日志和报告

---

## 📁 同步文件清单 (32 个)

### 文档 (4 个)
| 文件 | 说明 |
|------|------|
| `INDEX.md` | 本目录文件索引和说明 |
| `SUMMARY.md` | 远程验证总结 |
| `storage_offset_diagnosis.md` | storage_offset 警告诊断结论 |
| `.gitignore` | 排除大文件 (compat_log_analysis_remote.json) |

### NPU 兼容性报告 (6 个)
| 文件 | 大小 | 说明 |
|------|------|------|
| `compat_report_npu.json` | 2KB | 初始兼容性探测 |
| `compat_benchmark_npu.json` | 4KB | 标准形状基准 |
| `compat_benchmark_npu_smoke.json` | 4KB | 冒烟测试基准 |
| `compat_benchmark_npu_large.json` | 5KB | 大形状基准 |
| `compat_log_analysis_remote.json` | 463KB | 166 个日志文件分析 |
| `train_tp_moe_4card_smoke_compat.json` | 4KB | MoE 训练后兼容性报告 |

### 训练日志 - 真实 910A 运行 (5 个)
| 文件 | 大小 | 说明 |
|------|------|------|
| `train_tp_custom_smoke.log` | 4KB | 单卡 TP 2-step 冒烟 |
| `train_tp_custom_4card_smoke.log` | 6KB | 4卡 TP 冒烟 (v1) |
| `train_tp_custom_4card_smoke_v2.log` | 5KB | 4卡 TP 冒烟 (v2, autograd 修复) |
| `train_tp_moe_4card_smoke.log` | 14KB | 4卡 MoE 冒烟 |
| `train_tp_moe_4card_smoke_skip_save.log` | 14KB | 4卡 MoE (skip_save 模式) |

### 诊断日志 - storage_offset 调查 (10 个)
| 文件 | 说明 | Warning |
|------|------|---------|
| `qwen_orig_warn.log` | 原始 Qwen eval forward | ❌ 无 |
| `qwen_orig_train_warn.log` | 原始 Qwen train backward | ✅ 有 |
| `qwen_tp_train_warn.log` | 转换后 Qwen TP train backward | ✅ 有 |
| `qwen_train_forward_only.log` | 仅 train forward | ❌ 无 |
| `qwen_backward_only.log` | 仅 backward | ✅ 有 |
| `qwen_logits_backward.log` | Logits-mean backward | ✅ 有 |
| `qwen_single_layer_backward.log` | 单层 decoder backward | ✅ 有 |
| `qwen_attn_mlp_backward.log` | Attention+MLP backward | ✅ 有 |
| `tp_attention_backward.log` | 自定义 TPAttention backward | ❌ 无 |

### 其他文件 (7 个)
| 文件 | 说明 |
|------|------|
| `tp_moe_router_npu_diag.json` | TPMoERouter npu:0 诊断 |
| `fusion_result.json` | 算子融合分析 (40KB) |
| `test_npu_compat_layer.txt` | 测试结果: OK |
| `test_npu_compat_log_analyze.txt` | 测试结果: OK |
| `test_tp_attention_compat.txt` | 测试结果: OK |
| `py_compile.txt` | 空 |
| `py_compile_train_tp_custom.txt` | 空 |

---

## 🔑 关键证据链

### 1. storage_offset 警告结论
```
原始 Qwen eval forward:     ❌ 无警告
原始 Qwen train forward:    ❌ 无警告
原始 Qwen train backward:   ✅ 有警告 ← 复现成功
原始 Qwen self_attn backward: ✅ 有警告 ← 最小复现
自定义 TPAttention backward: ❌ 无警告 ← 我们的实现干净

结论: 这是 torch_npu 2.5.1 + 910A baseline 行为，非 TP 回归
```

### 2. 兼容性层验证
```
real-device 4-card MoE smoke:
- safe_softmax: 52 attempts, 0 fallback
- safe_any: 832 attempts, 0 fallback
- training: completed successfully
```

### 3. Autograd 修复验证
```
train_tp_custom_4card_smoke.log (v1):     有 allreduce autograd kernel 警告
train_tp_custom_4card_smoke_v2.log (v2):  ❌ 警告消失 ← 修复成功
```

---

## 📊 同步统计

| 指标 | 数值 |
|------|------|
| 新增文件 | 32 个 |
| 新增代码行 | 13,540+ 行 |
| 总提交数 | 12 个 (本轮) |
| 总跟踪文件 | 141 个 |

---

## ✅ 检查清单

- [x] 从远程 bms-hit-haom-jbgs 同步所有关键文件
- [x] 验证文件完整性
- [x] 创建 INDEX.md 索引
- [x] 提交到本地 git
- [x] 更新 memory-bank 记录

---

**同步完成**: 所有关键验证数据现在都在本地 git 中，不再依赖远程 910A 机器。

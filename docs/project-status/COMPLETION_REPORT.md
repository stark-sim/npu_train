# Ascend 910A NPU 训练项目 - 收尾报告

**日期**: 2026-04-01  
**分支**: stark-sim/real-moe-smoke  
**状态**: 910A 阶段完成，准备交接

---

## 📋 项目概览

本项目为 **Ascend 910A + 低版本 CANN** 环境构建了可运行的 PyTorch 大模型训练基础设施，包括：

- 单卡/DDP/PP/TP 多种并行训练模式
- DeepSeek-V2-Lite MoE 模型支持
- NPU 兼容性层（处理低版本 CANN 的算子问题）
- TP Checkpoint 工具链（保存/导出/重分片/恢复）
- 真实数据训练支持（HuggingFace Arrow 格式）

---

## ✅ 已完成工作

### 核心训练功能
| 功能 | 状态 | 验证 |
|------|------|------|
| 单卡训练 (`train.py`) | ✅ | CPU/NPU 通过 |
| DDP 训练 (`train_ddp.py`) | ✅ | 多卡通过 |
| PP 训练 (`train_pp.py`) | ✅ | 多卡通过 |
| 自定义 TP 训练 (`examples/train_tp_custom.py`) | ✅ | 4卡 910A 通过 |
| MoE TP 训练 (`examples/train_tp_moe.py`) | ✅ | 4卡 910A 通过 |

### NPU 兼容性层 (`npu_parallel/npu_compat.py`)
| 功能 | 状态 | 说明 |
|------|------|------|
| 算子探测与回退 | ✅ | `safe_softmax`, `safe_topk`, `safe_any` 等 |
| 策略控制 (`fallback/warn/strict`) | ✅ | 生产用 fallback，诊断用 warn/strict |
| 统计与性能计数 | ✅ | 记录尝试/成功/失败/回退次数和耗时 |
| 错误签名映射 | ✅ | 分类 ACL/HCCL/内存/超时等错误 |
| 日志分析工具 | ✅ | `tools/npu_compat_log_analyze.py` |
| 基准测试工具 | ✅ | `tools/npu_compat_benchmark.py` |

### TP Checkpoint 工具链
| 功能 | 状态 | 文件 |
|------|------|------|
| 分片保存 | ✅ | `npu_parallel/checkpoint_utils.py` |
| 检查与导出 | ✅ | `tools/tp_checkpoint.py` |
| 重分片 | ✅ | `tools/tp_checkpoint.py --reshard` |
| 优化器状态保存 | ✅ | `examples/train_tp_*.py` |
| 优化器状态重分片 | ✅ | 支持单组 TP 布局 |
| 训练恢复 (`--resume_from`) | ✅ | 已验证 |

### DeepSeek/MoE 支持
| 功能 | 状态 | 说明 |
|------|------|------|
| DeepSeek-V2-Lite 转换 | ✅ | MLA 注意力 TP 部分支持 |
| MoE 路由 TP | ✅ | `TPMoERouter` 兼容层包装 |
| MoE 专家并行 | ✅ | 基础实现 |

---

## 🔍 关键诊断结论

### `storage_offset ... untrustworthy64/128` 警告

**结论**: 这是原始 Qwen 模型的 baseline backward/compiler 警告，**不是 TP 引入的回归**。

**证据链**:
- 原始 Qwen eval forward: ❌ 无警告
- 原始 Qwen train forward: ❌ 无警告  
- 原始 Qwen train backward: ✅ **有警告** (复现)
- 原始 Qwen `self_attn` backward: ✅ **有警告** (最小复现)
- 自定义 `TPAttention` backward (matched shape): ❌ 无警告 (我们的实现干净)

**处理**: 日志分析工具已忽略此类警告，避免污染真实故障诊断。

**文件**:
- `.context/remote-npu-compat-20260331/SUMMARY.md:69`
- `.context/remote-npu-compat-20260331/storage_offset_diagnosis.md`
- `tools/repro_storage_offset_warning.py` (最小复现脚本)

---

## 📁 关键文件清单

### 必须提交到 git 的文件

```
# 核心代码
npu_parallel/
├── __init__.py
├── tp_layers.py          # TP 基础层 (含 autograd-aware all_reduce)
├── tp_attention.py       # TP 注意力实现
├── tp_moe.py             # MoE 路由和专家
├── convert_model.py      # HuggingFace 模型转换
├── checkpoint_utils.py   # Checkpoint 工具
└── npu_compat.py         # NPU 兼容性层 ⭐

# 训练脚本
examples/
├── train_tp_custom.py    # 自定义 TP 训练
├── train_tp_moe.py       # MoE TP 训练
├── benchmark_*.py        # 基准测试
└── ...

# 工具
tools/
├── tp_checkpoint.py      # Checkpoint 检查/导出/重分片
├── npu_compat_report.py  # 兼容性报告
├── npu_compat_benchmark.py    # 兼容性基准
├── npu_compat_log_analyze.py  # 日志分析 ⭐
└── repro_storage_offset_warning.py  # 最小复现 ⭐

# 测试
tests/
├── test_npu_compat_layer.py       # 兼容性层测试
├── test_npu_compat_log_analyze.py # 日志分析测试
├── test_tp_attention_compat.py    # TP 注意力兼容测试
├── test_tp_checkpoint_*.py        # Checkpoint 测试
└── ...

# 文档
docs/project-status/
├── summary-zh.md         # 中文项目总结
├── summary-en.md         # English summary
├── recommendations.md    # 后续建议
├── SUMMARY_REMOTE_20260331.md  # 远程验证总结
└── COMPLETION_REPORT.md  # 本文件

# Memory Bank
memory-bank/
├── RULES.md
├── activeContext.md
├── progress.md
├── systemPatterns.md
├── techContext.md
└── projectbrief.md

# 配置
AGENTS.md                 # Agent 指令
CLAUDE.md                 # Claude Code 指令
download_models.py        # 模型下载
robust_download.py        # 健壮下载
download_deepseek_moe.py  # DeepSeek 下载
```

### 远程验证工件（已归档，可选提交）

```
.context/remote-npu-compat-20260331/
├── SUMMARY.md                    # 验证总结 ⭐ 建议提交
├── storage_offset_diagnosis.md   # 诊断结论 ⭐ 建议提交
├── *.json                        # 兼容性报告（可选）
└── *.log                         # 日志文件（可选，较大）
```

---

## ⚠️ 已知限制

| 限制 | 说明 | 优先级 |
|------|------|--------|
| Hybrid TP+DDP | 组管理未完全完成 | 中 |
| DeepSeek-V2 MLA | 注意力 TP 部分支持 | 中 |
| MoE 通信优化 | 正确性优先，未完全优化 | 低 |
| 长运行鲁棒性 | 基础级别，需要更多生产验证 | 中 |
| 回退性能影响 | 已追踪但未量化端到端影响 | 低 |

---

## 🚀 快速开始

### 环境要求
- Ascend 910A NPU
- CANN 8.1.RC1
- Python 3.11
- PyTorch 2.5.1 + torch_npu 2.5.1

### 验证安装
```bash
# CPU 测试
python tests/test_npu_compat_layer.py
python tests/test_npu_compat_log_analyze.py
python tests/test_tp_attention_compat.py

# NPU 单卡冒烟（在 910A 机器上）
python examples/train_tp_custom.py \
  --model_path /path/to/Qwen2.5-1.5B-Instruct \
  --tp_size 1 --max_steps 2 --skip_save

# NPU 4卡冒烟（在 910A 机器上）
torchrun --nproc_per_node=4 examples/train_tp_custom.py \
  --model_path /path/to/Qwen2.5-1.5B-Instruct \
  --tp_size 4 --max_steps 2 --skip_save \
  --compat_report_file compat_report.json
```

---

## 📝 后续建议

### 高优先级
1. **长运行恢复工具**: 为 910A 生产环境添加自动故障恢复
2. **性能分析**: 使用 ASCEND_PROFILER 进行更细致的性能调优

### 中优先级
3. **兼容性签名扩展**: 收集更多真实故障日志扩展错误签名
4. **MLA 注意力完整 TP 支持**: 完成 DeepSeek-V2 MLA 的 TP 实现

### 低优先级
5. **Hybrid TP+DDP**: 完成混合并行组管理
6. **MindSpore 迁移评估**: 如果未来升级 CANN 版本

---

## 🎯 项目价值总结

本项目为 **Ascend 910A + 低版本 CANN** 这一特定遗留环境提供了：

1. **可运行的训练基础设施**: 在官方新框架无法直接使用的情况下保持生产力
2. **工程化最佳实践**: Checkpoint 管理、兼容性层、日志分析等可复用模式
3. **诊断工具链**: 快速定位 910A 特定问题的工具集
4. **知识沉淀**: 关于 910A/CANN/torch_npu 行为模式的详细记录

---

## 📌 交接检查清单

- [x] 所有核心代码文件已整理
- [x] 关键诊断结论已归档到 `.context/`
- [x] Memory Bank 已更新到最新状态
- [x] 收尾报告已创建
- [ ] 提交到 git 并推送
- [ ] 验证远程机器仍可运行关键测试
- [ ] 备份 `.context/` 关键文件到持久存储

---

**文档创建**: 2026-04-01  
**最后更新**: 2026-04-01  
**负责人**: stark-sim

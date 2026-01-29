# DeepSeek-V2-Lite MoE TP 训练支持 - 更新摘要

## 更新时间
2025-01-28

## 目标
为 DeepSeek-V2-Lite (16B参数，MoE架构) 添加 NPU 张量并行训练支持

## 问题回顾
之前尝试训练 DeepSeek-V2-Lite 时遇到的错误：
1. **HCCL超时 (error 107020)** - 大模型JIT编译时间过长
2. **NPU操作不支持 (error 500002)** - `torch.nonzero` 等操作在NPU上不可用
3. **OOM** - 加载完整模型后内存不足
4. **BrokenPipeError** - 长时间编译导致进程崩溃

## 代码改进

### 1. `npu_parallel/tp_moe.py` - NPU兼容的MoE前向传播

**关键修改:**
- 避免使用 `torch.nonzero`（NPU error 500002）
- 避免使用 `.any()` 操作
- 使用纯矩阵运算替代动态索引
- 支持 `n_grouped_experts` 参数（DeepSeek-V2 gate使用）

```python
# 之前: 使用torch.nonzero获取专家索引
expert_mask = (expert_indices == global_expert_idx)
if expert_mask.any():  # ← NPU可能不支持
    tokens = expert_input[expert_mask]  # ← 动态索引

# 现在: 使用矩阵乘法和求和
expert_mask_float = (expert_indices == global_expert_idx).float()
tokens_for_expert = expert_mask_float.sum(dim=-1)
if tokens_for_expert.max().item() == 0:  # ← CPU上的标量比较
    continue
expert_input_flat = flat_hidden_states * tokens_for_expert.unsqueeze(-1)
expert_output_all = self.experts[local_idx](expert_input_flat)  # ← 批量处理
```

### 2. `examples/train_tp_moe.py` - 增强的训练脚本

**新增功能:**
- HCCL超时从3600s增加到7200s
- 添加NPU优化环境变量
- `load_model_progressive()` 函数
- JIT编译预热

```bash
# 新增环境变量
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=7200
export TORCH_NPU_ENABLE_COMGR=0
export TORCH_NPU_ALLOC_CONF=max_split_size_mb:128
export NPU_FUSION_ENABLE=1
```

### 3. `examples/train_deepseek_v2_lite.py` - 简化训练脚本（新建）

专门为DeepSeek-V2-Lite优化的训练脚本:
- 保守的默认参数（batch_size=1, max_length=256）
- 20步快速测试
- 详细的进度输出
- 完整的编译预热

### 4. `examples/test_deepseek_moe.py` - 转换测试脚本（新建）

不运行训练，仅验证:
- 模型能否加载
- TP转换是否成功
- CPU/NPU前向传播是否正常
- NPU反向传播是否正常

## 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `npu_parallel/tp_moe.py` | 修改 | NPU兼容的MoE实现 |
| `npu_parallel/convert_model.py` | 修改 | DeepSeek-V2转换逻辑 |
| `npu_parallel/__init__.py` | 修改 | 导出MoE组件 |
| `npu_parallel/supported_models.py` | 修改 | 更新文档 |
| `examples/train_tp_moe.py` | 修改 | 增强通用MoE训练脚本 |
| `examples/train_deepseek_v2_lite.py` | 新建 | DeepSeek专用训练脚本 |
| `examples/test_deepseek_moe.py` | 新建 | 转换测试脚本 |
| `tests/test_tp_moe.py` | 新建 | MoE单元测试 |
| `download_deepseek_moe.py` | 新建 | 模型下载脚本 |
| `sync_to_remote.sh` | 新建 | 同步脚本 |
| `test_moe_on_remote.sh` | 新建 | 远程测试脚本 |
| `TEST_GUIDE.md` | 新建 | 测试指南 |

## 使用方法

### 步骤1: 同步代码到远程服务器

```bash
# 在本地执行
./sync_to_remote.sh
```

### 步骤2: 在远程服务器运行测试

```bash
# 登录远程服务器
ssh sd@bms-hit-haom-jbgs

# 运行自动化测试
cd ~/npu_train/osaka
chmod +x test_moe_on_remote.sh
./test_moe_on_remote.sh
```

### 步骤3: 如果测试通过，运行训练

```bash
# 4卡TP训练
torchrun --nproc_per_node=4 examples/train_deepseek_v2_lite.py \
    --model_path "/home/sd/npu_train/models/deepseek-ai/DeepSeek-V2-Lite" \
    --tp_size 4 \
    --batch_size 1 \
    --max_length 256
```

## 预期结果

### 单元测试输出
```
test_alltoall_autograd_function ... ok
test_router_forward ... ok
test_aux_loss_computation ... ok
test_experts_initialization ... ok
test_expert_forward ... ok
test_moe_forward ... ok
test_moe_backward ... ok
test_forward (DeepSeek) ... ok
...
----------------------------------------------------------------------
Ran 13 tests in X.XXXs

OK
```

### 转换测试输出
```
============================================================
DeepSeek-V2 MoE TP Conversion Test
============================================================

[1/4] Loading tokenizer...
      Vocab size: 102400

[2/4] Loading model on CPU...
      Total params: 16,000,000,000

[3/4] Inspecting MoE structure...
      MLP type: DeepseekV2MoE
      Num experts: 64
      Has gate: Yes
      Has shared_experts: Yes

[4/4] Testing TP conversion...
[Rank 0] Converting DeepSeek-V2 MoE: 64 experts -> 16 local experts
      TP conversion successful!

[CPU] Testing forward pass on CPU...
      CPU forward pass successful! Output shape: torch.Size([1, 16, 102400])

[NPU] Moving model to NPU...
[NPU] Testing forward pass on NPU...
      NPU forward pass successful! Output shape: torch.Size([1, 16, 102400])

[NPU] Testing backward pass on NPU...
      NPU backward pass successful! Loss: X.XXXX

============================================================
All tests passed!
============================================================
```

## 故障排查

| 错误 | 原因 | 解决方案 |
|------|------|----------|
| HCCL timeout 107020 | JIT编译太长 | 增加HCCL_CONNECT_TIMEOUT |
| NPU error 500002 | 操作不支持 | 确保使用最新的tp_moe.py |
| OOM | 内存不足 | 减小batch_size或增加TP size |
| BrokenPipeError | 进程崩溃 | 先运行test_deepseek_moe.py预热 |

## 技术细节

### DeepSeek-V2-Lite 架构
- **总参数**: 16B
- **激活参数**: 2B (MoE，64个专家，top-6)
- **隐藏层大小**: 5120
- **注意力**: MLA (Multi-head Latent Attention)
- **专家结构**: 共享专家 + 路由专家

### TP切分策略
- **专家**: 64个专家切分到4个TP rank，每个rank 16个专家
- **注意力**: 仅输出投影(o_proj)使用TP（简化方案）
- **MLP**: gate_proj和up_proj使用列并行，down_proj使用行并行
- **LM Head**: 列并行 + gather

### NPU优化
1. **避免动态操作**: 使用固定形状的矩阵运算
2. **编译预热**: 训练前用小batch触发JIT
3. **内存管理**: CPU加载 → TP转换 → 移到NPU
4. **超时设置**: HCCL超时设为2小时

## 下一步

如果基础测试通过，可以尝试:
1. 增加batch_size和max_length
2. 使用更激进的数据类型（float16/bfloat16）
3. 添加gradient checkpointing
4. 优化all-to-all通信
5. 实现更完整的MLA attention TP

## 联系方式

如有问题，请查看:
- `TEST_GUIDE.md` - 详细测试指南
- `examples/test_deepseek_moe.py` - 测试脚本源码
- `npu_parallel/tp_moe.py` - MoE实现源码

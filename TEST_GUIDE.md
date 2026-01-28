# DeepSeek-V2-Lite MoE TP 训练测试指南

## 一、代码同步

在本地macOS执行：

```bash
# 方式1: 使用同步脚本
./sync_to_remote.sh

# 方式2: 手动同步关键文件
scp npu_parallel/tp_moe.py sd@bms-hit-haom-jbgs:~/npu_train/osaka/npu_parallel/
scp npu_parallel/convert_model.py sd@bms-hit-haom-jbgs:~/npu_train/osaka/npu_parallel/
scp examples/train_deepseek_v2_lite.py sd@bms-hit-haom-jbgs:~/npu_train/osaka/examples/
scp examples/test_deepseek_moe.py sd@bms-hit-haom-jbgs:~/npu_train/osaka/examples/
```

## 二、远程服务器测试步骤

### 1. 登录服务器

```bash
ssh sd@bms-hit-haom-jbgs
cd ~/npu_train/osaka
conda activate npu_train
```

### 2. 检查模型路径

```bash
ls -la /home/sd/npu_train/models/deepseek-ai/DeepSeek-V2-Lite/
```

如果模型不存在，先下载：

```bash
python3 download_deepseek_moe.py
```

### 3. 测试MoE转换（不运行训练）

这是最重要的验证步骤！

```bash
# 单卡测试转换
python3 examples/test_deepseek_moe.py \
    --model_path "/home/sd/npu_train/models/deepseek-ai/DeepSeek-V2-Lite" \
    --device "npu:0"
```

**预期输出：**
```
============================================================
DeepSeek-V2 MoE TP Conversion Test
============================================================

[1/4] Loading tokenizer...
      Vocab size: 102400

[2/4] Loading model on CPU...
      Total params: 16,000,000,000
      ...

[3/4] Inspecting MoE structure...
      MLP type: DeepseekV2MoE
      Num experts: 64
      Has gate: Yes
      Has shared_experts: Yes

[4/4] Testing TP conversion...
      TP conversion successful!

[CPU] Testing forward pass on CPU...
      CPU forward pass successful!

[NPU] Moving model to NPU...
[NPU] Testing forward pass on NPU...
      NPU forward pass successful!

[NPU] Testing backward pass on NPU...
      NPU backward pass successful!

============================================================
All tests passed!
============================================================
```

### 4. 如果测试通过，运行简化版训练

```bash
# 4卡TP训练，参数非常保守用于测试
torchrun --nproc_per_node=4 examples/train_deepseek_v2_lite.py \
    --model_path "/home/sd/npu_train/models/deepseek-ai/DeepSeek-V2-Lite" \
    --tp_size 4 \
    --batch_size 1 \
    --max_length 256 \
    --epochs 1 \
    --dtype float32
```

**预期输出：**
```
============================================================
DeepSeek-V2-Lite MoE TP Training
============================================================
Model: /home/sd/npu_train/models/deepseek-ai/DeepSeek-V2-Lite
TP Size: 4
World Size: 4
Batch Size: 1
Max Length: 256
Data Type: float32
============================================================

[Load] Loading model on CPU...
[Load] Total parameters: 16,000,000,000
[Load] Parameters per TP rank: 4,000,000,000

[TP] Converting model to tensor parallelism (tp_size=4)...
[Rank 0] Converting DeepSeek-V2 MoE: 64 experts -> 16 local experts
...
[TP] Conversion complete!
[TP] MoE architecture detected and converted!

[Warmup] Triggering NPU JIT compilation...
[Warmup] Compilation warmup complete!

[Train] Starting training (20 steps)...
Step 0/20 | Loss: X.XXXX | Aux: X.XXXXXX | LR: X.XXe-XX | Time: X.Xs
...
[Done] Training completed successfully!
```

## 三、故障排查

### 错误1: HCCL超时

```
RuntimeError: NPU warning, error code is 107020
```

**解决：** 增加超时时间
```bash
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=7200
```

### 错误2: NPU操作不支持 (error 500002)

```
RuntimeError: NPU warning, error code is 500002
```

**解决：** 检查是否使用了最新的tp_moe.py，我们已经替换了不兼容的操作。

### 错误3: OOM

```
RuntimeError: NPU out of memory
```

**解决：**
- 减小batch_size
- 减小max_length
- 使用float16代替float32
- 增加TP size（使用8卡）

### 错误4: BrokenPipeError

```
BrokenPipeError: [Errno 32] Broken pipe
```

**解决：** 通常是编译时间过长导致的，先运行test_deepseek_moe.py进行预热。

## 四、关键改进说明

1. **NPU兼容的前向传播**
   - 避免使用`torch.nonzero`
   - 使用纯矩阵运算替代复杂索引
   - 支持`n_grouped_experts`参数

2. **超时优化**
   - HCCL超时从3600s增加到7200s
   - 添加NPU特定的环境变量

3. **内存优化**
   - CPU先加载模型
   - 转换后再移到NPU
   - 使用`low_cpu_mem_usage=True`

4. **编译预热**
   - 在正式训练前用小batch触发JIT编译

## 五、测试命令速查

```bash
# 1. 快速验证转换
python3 examples/test_deepseek_moe.py

# 2. 4卡TP训练（保守参数）
torchrun --nproc_per_node=4 examples/train_deepseek_v2_lite.py --tp_size 4

# 3. 8卡TP训练（如果可用）
torchrun --nproc_per_node=8 examples/train_deepseek_v2_lite.py --tp_size 8

# 4. 单元测试
python3 tests/test_tp_moe.py
```

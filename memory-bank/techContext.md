# Technical Context

## Tech Stack

### Core
- Python 3.11
- PyTorch 2.5.1
- torch-npu 2.5.1
- transformers 4.57.x
- CANN 8.1.RC1
- HCCL for distributed communication

### Styling
- Not applicable; this is not a UI project

### State & Data
- HuggingFace model loading APIs
- Offline HuggingFace Arrow datasets via `datasets.load_from_disk`
- `torch.distributed` process groups and distributed samplers

### Testing
- Script-based tests under `tests/`
- Benchmark scripts for TP throughput and memory
- DeepSeek / MoE conversion and training sanity tests

### Dev Tools
- Shell scripts for local and remote execution
- Git for version history and progress tracking
- Memory Bank files for cross-session agent continuity

## Development Environment

### Prerequisites
- Ascend 910A NPU environment
- Python 3.11 runtime
- CANN environment sourced correctly
- `torch_npu` and `transformers` available in the active environment
- Access to local model paths under `/home/sd/npu_train/models`

### Setup Commands
```bash
# Single NPU
chmod +x run.sh && ./run.sh

# DDP
chmod +x run_ddp.sh && ./run_ddp.sh

# PP
chmod +x run_pp.sh && ./run_pp.sh

# Custom TP
python examples/train_tp_custom.py --model_path "/path/to/model" --tp_size 4

# Verification
python3 tests/test_tp_conversion.py
python3 tests/test_tp_mlp_only.py
python3 tests/test_hccl_ops.py
```

### Environment Variables
- `HCCL_CONNECT_TIMEOUT`
- `HCCL_EXEC_TIMEOUT`
- `HCCL_INTRA_PCIE_ENABLE`
- `HCCL_INTER_PCIE_ENABLE`
- `TORCH_NPU_ENABLE_COMGR`
- `TORCH_NPU_ALLOC_CONF`
- `NPU_FUSION_ENABLE`

## Project Structure
```text
project-root/
├── examples/         # Experiment and task-specific scripts
├── npu_parallel/     # TP, attention, MoE, conversion logic
├── tests/            # Validation and benchmark scripts
├── docs/             # Project summaries and recommendations
├── train*.py         # Main training entry points
└── *.sh / *.py       # Operational utilities
```

## Import Aliases
- No project-wide custom import alias configuration is currently documented.

## Build & Deploy
- No centralized build system; the repository is executed directly through Python and shell scripts.
- Remote execution is supported through scripts such as `sync_to_remote.sh` and `test_moe_on_remote.sh`.

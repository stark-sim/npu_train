# Product Context

## Problem Definition
Modern large-model training frameworks increasingly target newer Ascend platforms and newer CANN releases. This project addresses the gap for teams that still need to train and experiment on `Ascend 910A + lower CANN` using a `PyTorch + torch_npu` workflow.

## Target User
- Engineers running LLM training on Huawei Ascend 910A hardware
- Researchers experimenting with TP, PP, DDP, and MoE on constrained NPU environments
- Maintainers who need a practical fallback when the official newer framework stack is not deployable

## Core User Flows
1. Prepare models and datasets locally or on a remote Ascend server.
2. Run single-NPU, DDP, PP, or TP training scripts.
3. Convert supported HuggingFace models into TP form.
4. Run DeepSeek / MoE test scripts before full training.
5. Launch long-running real-data training and resume from checkpoints.
6. Benchmark throughput, memory use, and stability across TP sizes.

## UX Decisions
- The project is CLI-first and script-driven.
- Operational convenience is provided through shell scripts and Python entry points rather than UI.
- The repository favors pragmatic examples over framework abstraction.

## Key Features
- Ascend 910A training support with `torch_npu`
- Custom tensor parallelism on HCCL
- DeepSeek-V2 / DeepSeek-V2-Lite adaptation
- MoE tensor parallel support and tests
- Real-data training from offline HuggingFace datasets
- Download utilities, integrity repair, and remote execution scripts

## Success Metrics
- Models can be loaded, converted, and trained on NPU without compatibility failures.
- TP delivers usable memory reduction and throughput improvement.
- DeepSeek / MoE workflows pass conversion and training sanity checks.
- Real-data training can run for long steps with checkpoint save/resume.

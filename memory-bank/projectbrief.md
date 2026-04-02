# Project Brief

## Project Name
NPU Training Project

## Description
A Python-based training project for large language models on Huawei Ascend 910A NPUs, with support for single-device training, DDP, PP, custom TP, DeepSeek/MoE adaptation, and real-data training workflows.

## Vision
Provide a practical and reproducible large-model training stack for legacy Ascend environments, especially `Ascend 910A + lower CANN + PyTorch/torch_npu`, where newer official frameworks are not directly usable.

## Core Requirements
- Run LLM training on Ascend 910A NPUs.
- Support single NPU, DDP, PP, and custom TP training modes.
- Convert HuggingFace transformer models into TP-compatible structures.
- Support DeepSeek-family models, including DeepSeek-V2-Lite and MoE variants.
- Support real dataset training from offline HuggingFace Arrow datasets.
- Provide practical scripts for model download, verification, benchmarking, and remote execution.

## Goals
- Maintain a runnable and extensible training stack on older Ascend infrastructure.
- Improve throughput, memory efficiency, and long-run stability on NPU.
- Preserve compatibility workarounds for torch-npu and HCCL limitations.
- Keep the project useful even when newer official MindSpore capabilities are unavailable on 910A.

## Scope

### In Scope
- Core training scripts: `train.py`, `train_ddp.py`, `train_pp.py`
- Custom TP implementation in `npu_parallel/`
- DeepSeek and MoE support
- Real-data training scripts in `examples/`
- Download, repair, and benchmark utilities
- Test scripts for TP conversion, HCCL ops, NPU APIs, and DeepSeek/MoE paths

### Out of Scope
- Migration to MindSpore / MindSpore Transformers as the primary stack
- UI or end-user application workflows
- Cloud-native orchestration or large-scale cluster management beyond local/remote scripts

## Key Stakeholders
- Project maintainers working on Ascend 910A training
- AI coding agents used in the repository: Claude Code and OpenAI Codex

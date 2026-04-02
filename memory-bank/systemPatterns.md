# System Patterns

## Architecture Overview
The repository is organized around a script-first training workflow with a reusable custom TP module. Core training entry points live at the repository root and in `examples/`, while `npu_parallel/` contains NPU/HCCL-specific tensor-parallel building blocks and conversion utilities.

## Project Structure
```text
project-root/
├── train.py                    # Single-NPU baseline training
├── train_ddp.py                # Data parallel training
├── train_pp.py                 # Pipeline parallel training
├── examples/                   # TP, MoE, DeepSeek, benchmark, and real-data scripts
├── npu_parallel/               # Custom TP layers, attention, model conversion, MoE logic
├── tests/                      # Conversion, HCCL, NPU, benchmark, and model-specific tests
├── docs/project-status/        # Project summaries and recommendation documents
├── README.md                   # High-level project overview
└── CLAUDE.md / AGENTS.md       # Agent instructions and memory-bank protocol
```

## Design Patterns in Use
- Script-oriented training entry points for each execution mode
- Custom TP layer abstraction (`ColumnParallelLinear`, `RowParallelLinear`)
- Model-structure-based conversion from HuggingFace modules to TP-compatible modules
- NPU-specific operator fallbacks where unsupported ops would otherwise fail
- Test-and-benchmark scripts as validation artifacts for performance and compatibility

## Component Relationships
- `train*.py` and `examples/train_*.py` load HuggingFace models and tokenizers, then optionally call `convert_to_tp()` from `npu_parallel/convert_model.py`.
- `npu_parallel/tp_layers.py` provides the primitive TP communication-aware layers.
- `npu_parallel/tp_attention.py` builds TP-aware attention and MLP components.
- `npu_parallel/tp_moe.py` provides MoE routing, expert handling, and DeepSeek-specific compatibility logic.
- Test scripts in `tests/` validate conversion, communication, and compatibility assumptions.

## State Management
There is no application-level state manager. State is managed through:
- Python objects in training scripts
- Distributed process groups in `torch.distributed`
- Model checkpoints on disk
- Environment variables controlling HCCL and NPU behavior

## Data Flow
1. Load tokenizer and model from local HuggingFace-compatible paths.
2. Optionally convert the model to custom TP modules.
3. Prepare synthetic or real datasets and distributed samplers.
4. Execute forward/backward on NPU.
5. Synchronize gradients or TP communications through HCCL.
6. Save checkpoints and log performance metrics.

## Architecture Decisions

| Decision | Rationale | Date |
|----------|-----------|------|
| Use `PyTorch + torch_npu` instead of migrating to MindSpore | The active environment is `Ascend 910A + lower CANN`, where newer official framework paths are not directly usable | [2026-03-26] |
| Keep custom TP in `npu_parallel/` | The project needs TP that is runnable on HCCL and adaptable to HuggingFace models in the current stack | [2026-03-26] |
| Preserve NPU-specific operator workarounds in MoE code | Compatibility on 910A is more important than framework purity | [2026-03-26] |
| Prioritize script-driven workflows | The project is used by engineers running experiments and remote jobs rather than end users | [2026-03-26] |
| Add compatibility policy modes with default fallback | 910A production runs need continuity by default, but diagnostics need strict/warn modes and fallback visibility | [2026-03-26] |
| Add signature-based error classification and perf counters in compatibility layer | On older CANN stacks, triage speed and fallback observability are operationally critical | [2026-03-26] |
| Add log-analysis helper and CLI for signature mining | Real 910A failures are log-first signals; extracting candidate patterns should be scriptable and repeatable | [2026-03-26] |
| Require outcome objectives and reviewable update plans for signature mining | Keeps log-analysis outputs goal-oriented and prevents silent policy drift | [2026-03-26] |
| Generate reviewable patch templates instead of auto-applying signature updates | The workflow should accelerate triage without silently mutating compatibility policy | [2026-03-31] |
| Extend compatibility wrappers first in critical TP/MoE/attention hot paths | This captures the highest-value low-CANN operator failures without broad risky refactors | [2026-03-31] |
| Validate compatibility changes through remote staging instead of the live remote worktree | This reduces risk when the device is scarce and shared with existing experiments | [2026-03-31] |
| Use autograd-aware communication wrappers for TP forward collectives | Direct `dist.all_reduce` on forward activations can emit correctness-risk warnings under current torch/torch_npu | [2026-03-31] |
| Add smoke-run flags for skip-save and post-training compat-report export | Real 910A device time and disk are scarce, so smoke validation should avoid large checkpoints while preserving runtime evidence | [2026-03-31] |
| Filter empirically benign compiler warnings only in offline log triage | Baseline torch-npu warnings like `storage_offset/untrustworthy` can appear even on successful original-model backward runs and should not pollute failure mining | [2026-04-01] |

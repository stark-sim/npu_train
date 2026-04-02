# Project Structure

## Core Directories

```
.
├── npu_parallel/           # Core TP and NPU modules
│   ├── __init__.py
│   ├── npu_compat.py       # ⭐ NPU compatibility layer
│   ├── checkpoint_utils.py # Checkpoint utilities
│   ├── tp_layers.py        # TP linear layers
│   ├── tp_attention.py     # TP attention
│   ├── tp_moe.py           # TP MoE routing
│   ├── convert_model.py    # Model conversion
│   └── supported_models.py # Model registry
│
├── tools/                  # Diagnostic and utility tools
│   ├── tp_checkpoint.py              # Checkpoint inspect/export/reshard
│   ├── npu_compat_report.py          # Compatibility reporting
│   ├── npu_compat_benchmark.py       # Benchmark raw/safe/fallback
│   ├── npu_compat_log_analyze.py     # ⭐ Log analysis
│   └── repro_storage_offset_warning.py # ⭐ Minimal repro
│
├── tests/                  # Test suite
│   ├── test_npu_compat_layer.py
│   ├── test_npu_compat_log_analyze.py
│   ├── test_tp_attention_compat.py
│   ├── test_tp_checkpoint_*.py (4 tests)
│   └── test_*.py (original tests)
│
├── examples/               # Training examples
│   ├── train_tp_custom.py  # Custom TP training
│   ├── train_tp_moe.py     # MoE TP training
│   └── benchmark_*.py      # Benchmarks
│
├── docs/
│   └── project-status/     # Project documentation
│       ├── COMPLETION_REPORT.md
│       ├── FINAL_HANDOFF.md
│       ├── ARTIFACT_SYNC_REPORT.md
│       ├── SUMMARY_REMOTE_20260331.md
│       ├── storage_offset_diagnosis.md
│       └── stage-results*.md (short/full, zh/en)
│
├── memory-bank/            # AI agent context
│   ├── RULES.md
│   ├── activeContext.md
│   ├── progress.md
│   ├── systemPatterns.md
│   ├── techContext.md
│   └── projectbrief.md
│
├── .context/               # Remote validation artifacts
│   └── remote-npu-compat-20260331/
│       ├── INDEX.md
│       ├── SUMMARY.md
│       ├── storage_offset_diagnosis.md
│       ├── compat_*.json (6 files)
│       ├── train_tp_*.log (5 files)
│       ├── qwen_*.log (9 files)
│       └── ... (other artifacts)
│
├── train*.py               # Main training entry points
├── *.sh                    # Shell scripts
└── README.md, CHANGELOG.md, FINAL_HANDOFF.md
```

## Key Files by Purpose

### For Users
- `README.md` - Quick start and overview
- `examples/train_tp_custom.py` - TP training example
- `tools/tp_checkpoint.py` - Checkpoint management

### For Developers
- `npu_parallel/npu_compat.py` - Compatibility layer implementation
- `memory-bank/` - Project context and decisions
- `docs/project-status/COMPLETION_REPORT.md` - Full status

### For Operations
- `tools/npu_compat_log_analyze.py` - Log triage
- `tools/repro_storage_offset_warning.py` - Environment validation
- `.context/remote-npu-compat-20260331/` - Validation evidence

## Statistics

- **Total Files**: 144 tracked
- **Source Code**: ~8,000 lines
- **Documentation**: ~5,000 lines
- **Tests**: ~2,000 lines
- **Git Commits**: 26 total

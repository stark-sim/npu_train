# storage_offset Warning Diagnosis

Date: 2026-04-01
Host: `sd@bms-hit-haom-jbgs`
Stage dir: `~/npu_train_hamburg_stage`
Model: `Qwen-Qwen2.5-1.5B-Instruct`
Runtime: `torch 2.5.1` + `torch_npu 2.5.1` on Ascend 910A

## Main finding

The warning

`Warning: [Check][offset] Check input storage_offset[%ld] = 0 failed, result is untrustworthy64`

is **not TP-specific** in the current environment.

It reproduces on the **original non-TP Qwen backward path**, and it can be shrunk to the **original Qwen decoder-layer / self-attention backward** path.

It does **not** reproduce in the matched-shape custom `TPAttention` backward smoke used in this repository.

## Repro matrix

- Original Qwen eval forward only: no warning
  - log: `qwen_orig_warn.log`
- Original Qwen train forward only: no warning
  - log: `qwen_train_forward_only.log`
- Original Qwen full training backward: warning reproduced
  - log: `qwen_orig_train_warn.log`
- Original Qwen logits-mean backward: warning reproduced
  - log: `qwen_logits_backward.log`
- Original Qwen single decoder layer backward: warning reproduced
  - log: `qwen_single_layer_backward.log`
- Original Qwen `self_attn` branch backward: warning reproduced
  - log: `qwen_attn_mlp_backward.log`
- Original Qwen `mlp` branch backward: no extra warning observed in the branch split run
  - log: `qwen_attn_mlp_backward.log`
- Converted Qwen `convert_to_tp(tp_size=1)` training backward: same warning reproduced
  - log: `qwen_tp_train_warn.log`
- Custom `TPAttention` backward at matched shape: no warning reproduced
  - log: `tp_attention_backward.log`

## Practical conclusion

- Treat the warning as a **baseline torch-npu/compiler backward warning** for the current stack.
- Do **not** use this warning alone as evidence that the repository's TP changes are wrong.
- Offline log triage should ignore this warning so real failures remain visible.

## Follow-up if needed

- If a smaller external repro is needed for upstream reporting, start from the original Qwen `self_attn` backward path instead of the full model.
- If future regressions appear, compare against `tp_attention_backward.log` first to see whether the custom path remains clean.

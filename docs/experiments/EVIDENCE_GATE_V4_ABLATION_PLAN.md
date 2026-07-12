# Evidence Gate v4 Ablation Plan

## Goal

The v4 mainline is `evidence_gate_v4_residual`: base FraudGT logits plus prototype evidence, bounded high-order structural residual evidence, an uncertainty router/gate, gate-budget regularization, and early auxiliary supervision for the structural branch.

The ablation should answer one paper-facing question:

> Which parts of the v4 decoder actually explain the performance change: prototype evidence, structural residual evidence, routing/gating, auxiliary training, or residual magnitude?

## Recommended Ablation Matrix

### Core variants

| Priority | Variant | Config override | What it tests | Interpretation |
|---|---|---|---|---|
| P0 | `full_v4` | `model.edge_decoding evidence_gate_v4_residual` | Full proposed decoder | Main result reference |
| P0 | `proto_only` | `model.edge_decoding evidence_gate_proto` | Keep prototype core only; remove structural residual and v4 router | Whether prototype evidence alone helps |
| P0 | `no_gate` | `model.edge_decoding evidence_gate_v4_nogate` | Keep structural residual, force it always open | Whether routing/gating is necessary |
| P0 | `no_aux` | `model.eg_struct_aux_weight 0 model.eg_struct_aux_epochs 0` | Full v4 without auxiliary structural supervision | Whether the structural branch needs direct training signal |
| P0 | `no_budget` | `model.eg_gate_budget_weight 0` | Full v4 without target-budget regularization | Whether gate sparsity/discipline matters |
| P1 | `weak_residual` | `model.eg_struct_residual_scale 0.5` | Smaller bounded structural correction | Whether full residual magnitude is too aggressive |
| P1 | `strong_residual` | `model.eg_struct_residual_scale 1.5` | Larger bounded structural correction | Whether v4 is underusing structural evidence |

### Why these are not strictly layer-by-layer

These variants are not intended to be monotonically increasing. They are diagnostic switches around different mechanisms:

- `proto_only` removes the structural route entirely.
- `no_gate` keeps the structural route but removes sample-wise control.
- `no_aux` keeps architecture at inference but changes how the structural branch learns.
- `no_budget` keeps the gate but removes the pressure to use it selectively.
- `weak_residual` / `strong_residual` test the scale of the structural correction.

So F1 does not have to increase variant by variant. A good ablation table should show which mechanism is useful, redundant, or harmful under different dataset scales and imbalance settings.

## Execution Strategy

### Stage A: fast diagnostic pass

Run one seed first, using the same best-seed policy as the current formal evaluation where possible:

| Dataset | Seed |
|---|---:|
| AML-Small-HI | 42 |
| AML-Small-LI | 42 |
| AML-Medium-HI | 42 |
| AML-Medium-LI | 44 |
| AML-Large-HI | 43 |
| AML-Large-LI | 44 |

Stage A runs all P0 variants across six datasets. This gives the first clean module-level story.

### Stage B: confirmatory pass

For variants that matter in Stage A, add seeds `42,43,44` on the datasets where the effect is meaningful or unstable.

### Stage C: sensitivity pass

Run `weak_residual` and `strong_residual` only after Stage A shows the structural residual branch matters. These are tuning/sensitivity variants, not core paper ablations.

## Current Launch Policy

- Do not interrupt the remaining formal v4 large runs.
- Use only idle GPUs with enough free memory.
- Start with P0 variants only.
- Use `optim.max_epoch 500`, `train.early_stop False`, and `train.auto_resume True` to match the current formal raw-peak evaluation style.
- Store results under `results/evidence_gate_v4_ablation`.

## Paper Table Recommendation

Main ablation table:

1. `proto_only`
2. `full_v4 - no_gate`
3. `full_v4 - no_aux`
4. `full_v4 - no_budget`
5. `full_v4`

Optional appendix sensitivity:

1. `weak_residual`
2. `full_v4`
3. `strong_residual`

The table should report raw best Test F1, because the current evaluation policy uses raw peak test F1 rather than val-selected test F1.

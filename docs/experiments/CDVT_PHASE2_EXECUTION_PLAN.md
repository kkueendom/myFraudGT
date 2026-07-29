# CDVT Phase 2 Execution Plan

## Frozen candidate

Phase 2 evaluates the validation-selected CDVT architecture without further
model selection:

- causal transaction-event graph;
- relation-aware temporal event encoder;
- representation-level cross-attention with the FraudGT account view;
- `dual_view`;
- `lambda_cons=0`;
- history size `K=4`; and
- dynamic-random train, validation, and test sampling.

Architecture freeze commit: `9038f85`.

## Six-dataset matrix

All Phase 2 results use seed 42.

| Dataset | Source |
|---|---|
| AML Small-LI | Reuse frozen Phase 1 dual-view manifest |
| AML Small-HI | New Phase 2 run |
| AML Medium-LI | New Phase 2 run |
| AML Medium-HI | New Phase 2 run |
| AML Large-LI | Reuse frozen Phase 1 dual-view manifest |
| AML Large-HI | New Phase 2 run |

Each new run uses at most 500 epochs, 256 training iterations per epoch, batch
size 2048, evaluation every four epochs, and validation-selected checkpointing.
Early stopping begins only after epoch 80 and requires ten evaluation points
without validation improvement.

The queue uses every genuinely idle GPU, where idle means at most 512 MiB
allocated memory and at most 5% utilization. It never terminates another
process, refuses an existing task directory, and polls every 300 seconds.

## Protocol invariants

- `LinkNeighborLoader` uses `shuffle=True` for train, validation, and test.
- `val.fixed_target_panel=False`.
- No fixed target-edge panel or dedicated evaluation generator is used.
- Sampler RNG state is not reset before evaluation.
- Val-selected Test F1 and Raw-best Test F1 are stored separately.
- Every manifest records dataset, variant, seed, Git commit, materialized
  config, checkpoint, selected epoch, runtime, and
  `sampling_protocol=dynamic_random`.

## Advancement gate

The baseline hierarchy and final gate are defined by
`CDVT_BASELINE_POLICY_AMENDMENT.md`. Phase 2 advances only when:

1. CDVT exceeds PE-FraudGT Val-selected Test F1 on at least four of six
   datasets;
2. the six-dataset mean delta against PE-FraudGT is positive;
3. no dataset has an unexplained severe collapse; and
4. normal event context exceeds shuffled and off controls on at least two
   representative scales.

Multi-FraudGT and initial A2 are reported as strong secondary references.
Raw-best values cannot be used by this gate.

## Immutable execution identity

The portable Phase 2 deployment uses:

- audited source commit `df12ea6`;
- portable execution commit `2fb3333`;
- worktree `/e/yky/FraudGT_cdvt_phase2_2fb3333`; and
- result root `/e/yky/FraudGT_cdvt_results/phase2_2fb3333`.

Only complete `result_manifest.json` files are authoritative. Progress logs and
intermediate best checkpoints may be inspected operationally but cannot be
reported as final Phase 2 results.

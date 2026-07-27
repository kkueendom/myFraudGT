# CDVT Final Model Freeze

## Decision

The model advanced to the six-dataset evaluation is **CDVT dual-view without
sampling consistency**:

1. a causal transaction event graph with relation-aware temporal encoding;
2. representation-level cross-attention from the FraudGT account view to the
   event context;
3. `lambda_cons = 0`, so sampling consistency is not part of the final model.

This decision freezes the architecture before any Phase 2 result is observed.
It prevents dataset-specific selection on the six-dataset test matrix.

## Validation Evidence

The latest completed Phase 1 validation observations available at the freeze
point were:

| Variant | Small-LI validation F1 | Large-LI validation F1 | Mean |
|---|---:|---:|---:|
| Dual-view | 0.37500 | 0.39695 | 0.38597 |
| Dual-view + consistency | 0.37264 | 0.34008 | 0.35636 |

Sampling consistency reduced validation F1 at both scales and reduced the
two-dataset mean by 0.02961. The simpler dual-view model is therefore selected.
Test F1 was not used to choose between these two variants.

The observed dual-view val-selected test results were 0.43798 on Small-LI and
0.33803 on Large-LI. Against the initial A2 val-selected baselines, their
deltas are -0.02449 and +0.03695, respectively, for a positive two-scale mean
delta of +0.00623. This satisfies the preregistered rule for advancing a
cross-scale candidate, but it is not used to reverse the validation-based
variant choice.

## Frozen Phase 2 Specification

- Variant: `dual_view`
- Sampling consistency weight: `0.0`
- Seed: `42`
- History size `K`: `4`
- Event history hops: `2`
- Maximum events per target: `48`
- Event encoder: 2 layers, hidden dimension 64, 4 heads
- Training budget: at most 500 epochs, 256 iterations per epoch
- Evaluation period: 4 epochs
- Batch size: 2048
- Checkpoint selection: highest validation F1
- Sampling protocol: `dynamic_random`
- Fixed validation/test target panel: disabled

Small-LI and Large-LI reuse their existing Phase 1 dual-view seed-42 runs.
Only Small-HI, Medium-LI, Medium-HI, and Large-HI are newly trained in Phase 2.

## Decision Gate

Phase 2 advances to the representative three-seed evaluation only if the
val-selected test column wins at least four of six datasets and has a positive
mean delta against the initial A2 val-selected column. Raw-best results are
reported separately and never compared across metric columns. Absolute gains
smaller than 0.005 are labeled as possible dynamic-sampling variation.

Sampling consistency remains an ineffective optional extension for
supplementary analysis. It must not be described as a contribution of the
final CDVT model.

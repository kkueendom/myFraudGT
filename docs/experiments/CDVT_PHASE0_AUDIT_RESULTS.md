# CDVT Phase 0 Engineering Audit Results

Status: passed on 2026-07-29.

## Provenance

- Architecture freeze: `9038f85d63634a91c712302813aad221a4e907e5`
- Source commit: `3d69713f2786c8cbe5ab457faa7a35cd26494896`
- Portable execution commit:
  `9be1737ee9b1eab74e52f30f2986ba973d019b69`
- Dataset: AML Small-LI
- Variant: `dual_view`, `lambda_cons=0`, `K=4`
- Sampling protocol: `dynamic_random`
- Result:
  `results/CDVT_Phase0_GPU_Smoke_Mixed_9be1737.json`
- Result SHA-256:
  `262b235bf4c82c76f0d4859475b6b36f14419ae6c97ef6bda90d9d43c140d695`

## Static and integration checks

The remote FraudGT environment passed all 42 `test_cdvt*.py` tests before the
smoke run. The suite covers:

- causal predecessor and equal-timestamp edge-ID ordering;
- latest-`K` history, event relations, and transition features;
- target-edge ID alignment and absence of future/label leakage;
- normal, shuffled, and off intervention semantics;
- fusion placement and end-to-end backward propagation;
- empty-batch behavior and dynamic-sampling protocol invariants.

Python compilation, shell syntax, Git identity, and clean-worktree checks also
passed.

## Real GPU smoke

The final smoke selected a dynamically sampled batch containing both classes.

| Check | Result |
|---|---:|
| Target transactions | 867 |
| Fraud / normal targets | 1 / 866 |
| Account-encoder gradient norm | 0.96155 |
| Event-encoder gradient norm | 0.17916 |
| Cross-view fusion gradient norm | 0.21991 |
| Mean event count per target | 9.47982 |
| Minimum / maximum event count | 1 / 28 |
| Normal-shuffled mean absolute logit delta | 0.000977 |
| Normal-off mean absolute logit delta | 0.003887 |

All three trainable paths received nonzero gradients. Normal, shuffled, and off
conditions produced distinct logits, and every target had an event
representation.

## Fixed-batch trainability

Twenty-four AdamW steps on the same mixed-class batch reduced evaluation loss
from 0.67856 to 0.04245, a 93.74% relative reduction. This is an engineering
trainability check only. The batch is extremely imbalanced and the result is
not evidence of generalization or predictive quality.

An earlier `db54795` smoke selected an all-negative batch. It remains useful as
a GPU/gradient diagnostic but is invalid as mixed-class overfit evidence and
must not be cited as the final Phase 0 result.

## Decision

Phase 0 engineering qualification is complete. Predictive conclusions remain
conditional on complete Phase 2 and representative-scale multi-seed results.

# Fixed-Panel Prospective Pair Protocol

## Purpose

Historical AML runs recreated shuffled validation/test loaders every evaluation
while consuming at most 256 batches. Consequently, different epochs could be
ranked on different target-edge subsets. This protocol removes that checkpoint
selection confound before screening another model.

## Evaluation Contract

- Training sampling remains unchanged.
- Validation and test target panels are selected once by a label-independent,
  dataset-and-split-specific seed.
- Panel size remains the historical `batch_size * 256` target edges, or the
  entire split when it is smaller.
- Target order is fixed and the evaluation DataLoader generator is reset before
  every pass.
- The panel selector uses a local `torch.Generator`, so it does not advance the
  model or training-loader RNG.
- Validation and test labels do not affect panel membership.

Neighbor sampling remains bounded by the original `[50, 50]` configuration.
The reset generator and non-shuffled panel make repeated evaluation replay the
same target ordering and worker seed trajectory.

## First Matched Screen

The first pair reruns A2 from epoch 0:

| Dataset | Seed | Epochs |
|---|---:|---:|
| Small-LI | 42 | 500 |
| Large-LI | 44 | 500 |

Every candidate inheriting this protocol must use the same panel seed `1729`,
same pair, same scheduler, and same validation-F1 checkpoint rule. Old A2 values
`0.46247` and `0.30108` remain engineering references only; formal advancement
uses the rerun fixed-panel A2 values.

## Advancement Rule

A candidate advances only when both completed 500-epoch validation-selected
Test F1 values exceed fixed-panel A2 by at least `0.005`. Raw-best is diagnostic.
No test metric may choose a checkpoint, threshold, panel, or method variant.

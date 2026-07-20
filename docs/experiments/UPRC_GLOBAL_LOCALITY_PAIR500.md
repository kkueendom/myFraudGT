# UPRC-Global Locality Pair500

## Material Passport

- Branch: `feature/uprc-global-locality`
- Parent method commit: `49aa9c4`
- Primary metric: validation-selected Test F1
- Pair: Small-LI seed 42 and Large-LI seed 44

## Controlled Change

UPRC-Global changes only `uprc_locality_floor` from `0.10` to `1.00`. The
counterfactual pairwise objective, signed prototype direction, correction bound,
fixed panels, seeds, scheduler, and checkpoint rule are unchanged.

This removes sigmoid-entropy localization and tests the hypothesis that an
operational F1 threshold far below probability 0.5 requires rank correction
across the full score range. Prototype readiness and signed evidence still
control whether and in which direction a correction is applied.

The variant advances only if both completed 500-epoch validation-selected Test
F1 values exceed matched fixed-panel A2 by at least `0.005`.

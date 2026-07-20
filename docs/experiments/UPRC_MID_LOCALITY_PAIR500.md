# UPRC-Mid Locality Pair500

## Material Passport

- Branch: `feature/uprc-mid-locality`
- Parent method commit: `49aa9c4`
- Primary metric: validation-selected Test F1
- Pair: Small-LI seed 42 and Large-LI seed 44

## Controlled Change

UPRC-Mid changes only `uprc_locality_floor` from `0.10` to `0.50`. The
counterfactual pairwise objective, signed prototype direction, correction bound,
fixed panels, seeds, scheduler, and checkpoint rule are unchanged.

The rationale is that FraudGT tunes an F1 threshold far below probability 0.5
under extreme imbalance. A sigmoid-entropy locality prior can therefore treat
examples near the operational threshold as overconfident and suppress useful
rank corrections. The `0.50` floor retains uncertainty emphasis but allows a
moderate correction throughout the ranking range.

The variant advances only if both completed 500-epoch validation-selected Test
F1 values exceed matched fixed-panel A2 by at least `0.005`.

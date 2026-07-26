# GTF1C Phase 0 v2 Formal Plan

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Status: preregistered, not started
- Parent development result: `GTF1C_PHASE0_V2_DEV_RESULTS.md`
- Method: directional GTF1C v2, frozen
- Sampling protocol: `dynamic_random`
- Validation/test loader iterations allowed: 0

## Frozen Method

The formal run uses the exact v2 mechanism:

- independent add and remove threshold selection;
- three locked add/remove/joint candidates;
- one-sided `delta/3` candidate certification;
- intersection of time-block and graph-component paired F1 lower bounds;
- normal-selected candidate identity and thresholds frozen for controls;
- row-net, GTPRC row-harm, IID paired F1 and time-block paired F1 baselines.

No method parameter or regime parameter may change.

## Formal Execution

- seven regimes on seven GPUs;
- Small-LI and Large-LI in every regime;
- all three OOF fold rotations;
- 512 fresh replicates per regime;
- 21,504 dataset/fold trials;
- fresh seeds `74001` through `74007`;
- `block_count=32`;
- 16 thresholds per direction;
- family-level `delta=0.05`;
- minimum changes 20;
- practical paired F1 delta 0.005;
- no validation/test data.

## Formal Metrics

For every regime/dataset/method:

- qualification rate;
- false qualification rate
  `P(qualified and evaluation Delta F1 <= 0)`;
- practical failure rate
  `P(qualified and evaluation Delta F1 < 0.005)`;
- conditional violation rate among qualified trials;
- unconditional mean/median paired F1 delta;
- conditional mean/median paired F1 delta;
- unconditional oracle-coverage fraction;
- conditional clipped oracle-coverage fraction;
- per-fold and per-replicate qualification/violation counts.

Fold rotations are correlated within replicate. Trial-level rates are the
registered primary simulation summaries, while replicate-level any-violation
rates are mandatory sensitivity analyses.

## Formal Gate

All conditions must pass:

1. all seven manifests, two datasets and three rotations are complete;
2. row-net or GTPRC row-harm false qualification exceeds 0.10 in at least two
   registered negative units;
3. GTF1C false qualification is at most 0.07 in every one of the 14 normal
   regime/dataset units, not only negative regimes;
4. shuffled and harmful GTF1C qualification is at most 0.05 in every unit;
5. GTF1C practical failure is at most 0.10 in every positive unit;
6. GTF1C qualification is at least 0.25 in at least three Small-LI and two
   Large-LI positive regimes;
7. conditional median clipped oracle coverage is at least 0.30 in at least
   three Small-LI and two Large-LI positive regimes;
8. IID GTF1C qualification retains at least 80% of IID paired-F1
   qualification on both datasets.

The stricter all-unit false-qualification condition is intentionally capable
of rejecting the method given the development temporal-positive result.

## Decision

Passing authorizes a theory/assumption audit and a later real-evidence
qualification study. It does not authorize a claim of improved FraudGT F1.

Failure stops GTF1C. No v3, no block-count search, no confidence-level search
and no threshold-grid search are allowed.

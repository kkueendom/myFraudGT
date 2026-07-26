# TREFIC Phase 0 Development Plan

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Status: preregistered, not started
- Parent formal failure: `GTF1C_PHASE0_V2_FORMAL_RESULTS.md`
- Sampling protocol: `dynamic_random`
- Validation/test loader iterations allowed: 0
- FraudGT or evidence-model training allowed: no

## Fixed Input and Candidate Policies

Reuse the immutable Small-LI and Large-LI OOF payloads and the exact
directional v2 score generators. Positive and negative regime parameters are
unchanged.

Candidate construction remains:

- one add threshold selected on the selection fold;
- one remove threshold selected on the selection fold;
- three locked add, remove and joint candidates;
- no threshold re-estimation on certification or evaluation folds.

## TREFIC Certification

For each rotation:

1. estimate \(\rho_U\) from the combined base confusion counts of the
   selection and certification folds using a one-sided Wilson upper bound;
2. fix the ratio envelope to \([0,\rho_U]\);
3. for each of the three candidates, construct per-row directional utility at
   \(\rho=0\) and \(\rho=\rho_U\);
4. use one-sided `delta/6`, accounting for three candidates and two envelope
   endpoints;
5. require positive time-block and graph-component lower bounds at both
   endpoints;
6. among qualified candidates, choose the largest worst-endpoint lower bound;
7. evaluate that frozen candidate on the third fold.

The lower envelope endpoint is always zero. It may not be tuned.

## Compared Methods

- row-net;
- GTPRC row-harm;
- IID paired F1;
- GTF1C v2;
- TREFIC.

Normal-selected TREFIC candidate identity and thresholds are frozen for
shuffled and harmful controls.

## Seven GPU Development Screen

- the same seven positive/negative regimes as GTF1C formal;
- both datasets and all three fold rotations;
- 64 fresh replicates per regime;
- fresh seeds `75001` through `75007`;
- seven distinct GPU tasks;
- `block_count=32`;
- 16 thresholds per direction;
- family-level `delta=0.05`;
- minimum changes 20;
- practical paired F1 delta 0.005.

## Gate

All conditions must pass:

1. the exact F1 sign identity passes exhaustive integer confusion tests;
2. row-net or GTPRC row-harm false qualification exceeds 0.10 in at least two
   negative units;
3. TREFIC false qualification is at most 0.07 in all 14 normal units;
4. shuffled and harmful TREFIC qualification is at most 0.05 in all units;
5. TREFIC practical failure is at most 0.10 in all positive units;
6. TREFIC qualification is at least 0.25 in three Small-LI and two Large-LI
   positive regimes;
7. conditional median paired F1 delta is at least 0.005 in every positive unit
   with nonzero qualification;
8. conditional clipped oracle coverage is at least 0.30 in three Small-LI and
   two Large-LI positive regimes;
9. IID TREFIC qualification retains at least 70% of IID paired-F1
   qualification on both datasets.

## Stop Rule

Failure stops TREFIC. Do not tune:

- \(\rho_L=0\);
- Wilson confidence level;
- `delta`;
- graph/time block count;
- practical F1 threshold;
- policy quantiles;
- regime parameters.

Passing only authorizes a fresh formal preregistration. It does not authorize
validation/test or a predictive improvement claim.

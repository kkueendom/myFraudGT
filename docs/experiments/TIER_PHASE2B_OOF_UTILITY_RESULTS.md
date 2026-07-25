# TIER Phase 2b OOF Utility Qualification Results

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Verification Status: EXECUTED OOF tables audited
- Sampling Protocol: `dynamic_random`
- Teacher Commit: `c78214c1`
- Audit Commit: `061c92ed`
- Validation Loader Iterations: `0`
- Test Loader Iterations: `0`

## Fold Integrity

| Dataset | Fold | OOF edges | Positives | Pair retention | A2 loss min/last | Evidence loss min/last |
|---|---:|---:|---:|---:|---:|---:|
| Large-LI | 0 | 34796 | 29 | 0.39927 | 0.01530/0.02040 | 0.01115/0.01264 |
| Large-LI | 1 | 34975 | 15 | 0.39935 | 0.01689/0.02436 | 0.01056/0.01306 |
| Large-LI | 2 | 34589 | 28 | 0.39883 | 0.01702/0.02471 | 0.01058/0.01461 |
| Small-LI | 0 | 174832 | 92 | 0.99893 | 0.00903/0.00998 | 0.00974/0.01021 |
| Small-LI | 1 | 174450 | 84 | 0.99897 | 0.00821/0.00947 | 0.00964/0.01059 |
| Small-LI | 2 | 174577 | 63 | 0.99893 | 0.00786/0.01096 | 0.00996/0.01025 |

## Directional Utility

| Dataset | Direction | Eval candidates | Utility +/- | Prevalence | Normal AUPRC | Shuffled AUPRC | Off AUPRC | Normal-prev | Normal-shuffled | Gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Small-LI | add | 19 | 0/19 | 0.00000 | n/a | n/a | n/a | n/a | n/a | fail |
| Small-LI | remove | 5 | 3/2 | 0.60000 | 1.00000 | 0.80556 | 0.80556 | 0.40000 | 0.19444 | pass |
| Large-LI | add | 44 | 2/42 | 0.04545 | 0.30263 | 0.04177 | 0.08681 | 0.25718 | 0.26086 | pass |
| Large-LI | remove | 34 | 33/1 | 0.97059 | 0.95378 | 0.94440 | 0.96672 | -0.01681 | 0.00938 | fail |

## Locked OOF Policy

| Dataset | All candidates | Cal eligible | Eval changed | Corrected/broken | Ratio | A2 F1 | Routed F1 | Delta | Evidence gate | Policy gate | Final |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| Small-LI | 152 | 1 | 9 | 3/6 | 0.50000 | 0.19231 | 0.18182 | -0.01049 | pass | fail | fail |
| Large-LI | 453 | 0 | 2 | 2/0 | n/a | 0.04255 | 0.04444 | 0.00189 | pass | fail | fail |

## Decision

The OOF utility gate did not pass on both datasets. TIER stops: do not tune router thresholds, train CrossFusion, or open validation/test for this route. Proceed to the CET-FraudGT complementary temporal encoder mainline.

The normal/shuffled/off AUPRC comparison uses the same normal direction-candidate population. No validation or test labels were loaded by this qualification stage.

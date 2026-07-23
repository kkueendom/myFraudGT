# TIER Phase 1 Evidence Qualification Results

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Verification Status: EXECUTED manifests audited
- Sampling Protocol: `dynamic_random`
- Historical Baseline: `initial_A2`
- Fixed-panel A2: excluded

## Per-Task Results

| Dataset | Selection | Seed | Val-selected Test F1 | Delta vs initial A2 | Raw-best Test F1 | Delta vs initial A2 | Normal-shuffled | Normal-off | Corrected/broken | Correction rate | Changed | Gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Large-LI | recent | 44 | 0.17838 | -0.12270 | 0.19284 | -0.25436 | 0.17838 | 0.17838 | 0.07971 | 0.12644 | 149 | fail |
| Large-LI | role_motif | 44 | 0.16867 | -0.13241 | 0.27891 | -0.16829 | 0.16867 | 0.16867 | 0.89130 | 0.33884 | 87 | fail |
| Small-LI | recent | 42 | 0.30664 | -0.15583 | 0.35363 | -0.15304 | 0.30316 | 0.30664 | 0.33058 | 0.14760 | 161 | fail |
| Small-LI | role_motif | 42 | 0.31499 | -0.14748 | 0.35294 | -0.15373 | 0.31108 | 0.31499 | 0.44444 | 0.13043 | 117 | fail |

## Selection Summary

| Selection | Tasks | Mean Val-selected Delta | Mean Raw-best Delta | Qualified | Failed |
|---|---:|---:|---:|---:|---:|
| recent | 2 | -0.13926 | -0.20370 | 0 | 2 |
| role_motif | 2 | -0.13994 | -0.16101 | 0 | 2 |

A result with an absolute historical-A2 delta below `0.005` is within
the configured dynamic-sampling caution range. Qualification uses the
counterfactual evidence and paired A2 diagnostics in each manifest;
it is not inferred from headline F1 alone.

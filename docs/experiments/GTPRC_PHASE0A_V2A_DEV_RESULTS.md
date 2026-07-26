# GTPRC Phase 0A v2a Development Results

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Status: development screen completed; failed
- Remote commit: `a7b06f46`
- Output:
  `/e/yky/FraudGT_cet_results/gtprc_phase0a_v2_dev_a7b06f46`
- Manifests: 7/7
- Replicates: 64 per regime, 448 total
- Runtime or CUDA failures: none
- Validation/test labels opened: no

## Gate

| Requirement | Result |
|---|---|
| Row-IID violates harm in at least three dependent regimes | **fail** |
| GTPRC violation at most 0.07 in all regimes | pass |
| GTPRC retains at least 0.30 oracle coverage in five regimes | pass |
| IID coverage retention at least 0.80 | pass |
| Shuffled and harmful controls rejected | pass |
| Row-IID and GTPRC coverage differs in three regimes | pass |

Decision: `REDESIGN_STRESS_TEST`.

## Diagnostic Results

| Regime | Row-IID violation | GTPRC violation | Row-IID median coverage | GTPRC median coverage | GTPRC oracle fraction |
|---|---:|---:|---:|---:|---:|
| IID | 0.000 | 0.000 | 0.471 | 0.471 | 1.000 |
| Entity cluster | 0.000 | 0.000 | 0.630 | 0.390 | 0.618 |
| Temporal autocorrelation | 0.000 | 0.000 | 0.632 | 0.393 | 0.613 |
| Entity and temporal | 0.000 | 0.000 | 0.649 | 0.204 | 0.313 |
| Prevalence drift | 0.000 | 0.000 | 0.602 | 0.389 | 0.651 |
| Alignment drift | 0.000 | 0.000 | 0.393 | 0.000 | 0.000 |
| Rare positive and duplicate | 0.000 | 0.000 | 0.541 | 0.220 | 0.410 |

## Interpretation

v2a fixed the v1 tie: dependence-aware grouping now changes selected
coverage, and IID coverage is preserved. However, row-IID remains safe in
every replicate. The generator therefore still cannot test the claimed
benefit of dependence correction.

Two design features explain the remaining failure:

1. only 16 quantile policies create a coarse risk-coverage frontier, so the
   row-IID selector stays materially below the harm boundary;
2. the score includes a direct negative shared-cluster term, allowing it to
   avoid high-break dependency clusters before risk calibration is applied.

v2b may change only these development-test properties:

- increase policy-grid resolution;
- remove direct access to the latent dependency effect from the score;
- record selected test-risk means for diagnosis.

The harm limit, confidence level, outcomes, method comparison, and formal
advancement gates remain unchanged. v2a is not a formal method result.


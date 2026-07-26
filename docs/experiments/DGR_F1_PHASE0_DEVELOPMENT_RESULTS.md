# DGR-F1 Phase 0 Development Results

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent validate mode
- Experiment: DGR-F1 controlled development screen
- Remote commit: `e43b6b5e`
- Local implementation commit: `f88cc39`
- Remote output:
  `/e/yky/FraudGT_cet_results/dgr_f1_phase0_dev_e43b6b5e`
- Local mirror:
  `/Users/kun/FraudGT_experiment_workspace/dgr_f1_phase0_dev_e43b6b5e`
- Scenarios: 7
- Templates: Small-LI and Large-LI
- Monte Carlo experiments: 512 per scenario/template
- Independent streams: 64 per experiment
- Validation/test loader iterations: 0/0
- Status: reproducibly completed; preregistered gate failed

## Decision

`STOP_DGR_F1`.

DGR-F1 passed the positive-power, false-claim, unstable-abstention and
cross-scale gates. It failed simultaneous mean-interval coverage and
Large-LI harm-detection power. Per the stop rule, no checkpoint, confidence,
scenario or practical-margin tuning is permitted.

## Gate

| Registered condition | Result |
|---|---|
| Exact F1 identity has zero failures | PASS |
| False improvement <= 0.05 on null/harm | PASS |
| False harm <= 0.05 on null/positive | PASS |
| Positive power >= 0.80 in 3 scenarios x 2 templates | PASS, all 1.000 |
| Stable-IID median stop <= 32 streams | PASS, both 32 |
| Unstable scenarios abstain >= 0.90 | PASS, all 1.000 |
| Naive method false stable claim >= 0.15 | PASS |
| Simultaneous mean coverage >= 0.94 | **FAIL** |
| Large-IID power within 0.15 of Small-IID | PASS |
| Stable-harm power >= 0.80 on both templates | **FAIL** |
| Overall | **FAIL** |

## Primary Results

| Scenario | Template | Population mean Delta F1 | Practical improve probability | Practical harm probability | DGR improve | DGR harm | DGR insufficient |
|---|---|---:|---:|---:|---:|---:|---:|
| Stable IID improve | Small | +0.36576 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| Stable IID improve | Large | +0.23823 | 0.9999 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| Stable graph improve | Small | +0.34832 | 0.9947 | 0.0036 | 1.0000 | 0.0000 | 0.0000 |
| Stable graph improve | Large | +0.24298 | 0.9845 | 0.0011 | 1.0000 | 0.0000 | 0.0000 |
| Stable temporal improve | Small | +0.33727 | 0.9997 | 0.0001 | 1.0000 | 0.0000 | 0.0000 |
| Stable temporal improve | Large | +0.21005 | 0.9963 | 0.0002 | 1.0000 | 0.0000 | 0.0000 |
| Positive mean, low replication | Small | +0.21588 | 0.6028 | 0.3950 | 0.0000 | 0.0000 | 1.0000 |
| Positive mean, low replication | Large | +0.16943 | 0.6358 | 0.2044 | 0.0000 | 0.0000 | 1.0000 |
| Base-ratio sign reversal | Small | +0.00154 | 0.4428 | 0.4360 | 0.0000 | 0.0000 | 1.0000 |
| Base-ratio sign reversal | Large | +0.00828 | 0.4254 | 0.2704 | 0.0000 | 0.0000 | 1.0000 |
| Null | Small | 0.00000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| Null | Large | 0.00000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| Stable harm | Small | -0.24342 | 0.0002 | 0.9996 | 0.0000 | 1.0000 | 0.0000 |
| Stable harm | Large | -0.04612 | 0.0181 | 0.8780 | 0.0000 | 0.2949 | 0.7051 |

## Failure 1: Mean-Interval Coverage

The family-corrected Studentized interval achieved:

| Scenario | Small-LI | Large-LI |
|---|---:|---:|
| Null | 0.9883 | **0.8672** |
| Base-ratio reversal | 0.9785 | 0.9863 |

The Large-LI null distribution is highly discrete and concentrated near zero.
A Studentized interval can become too narrow and exclude the exact zero mean.
This invalidates the proposed joint method even though the final DGR decision
made no false improvement claim.

Replacing the interval after observing this result would be post hoc method
selection and is prohibited.

## Failure 2: Strict Replication Power

The Large-LI stable-harm population has:

- mean Delta F1: -0.04612;
- practical-harm probability: 0.8780;
- mean-only harm detection: 1.0000;
- replication-only harm detection: 0.2949;
- joint DGR-F1 harm detection: 0.2949.

The exact Clopper-Pearson lower bound and `pi0=0.75` require more evidence than
64 streams provide when the true rate is 0.878. This is not a coding failure:
it is the registered strictness/power trade-off.

## What Worked

1. All exact F1 mechanism checks passed.
2. DGR-F1 rejected every positive-mean but low-replication experiment.
3. DGR-F1 rejected every ratio-sign-reversal experiment.
4. Mean-only evaluation falsely made a stable claim in 100% of low-replication
   experiments.
5. One-stream/mean-only false stable claims reached 0.71 to 1.00 in the
   unstable Large-LI scenarios.
6. Stable positive power was 1.00 on both scales with median stopping at 32.

These are useful diagnostics, but they cannot override failed coverage and
harm-power gates.

## Research Consequence

DGR-F1 does not justify a prospective frozen-checkpoint experiment. Together,
the completed methods show:

- GTF1C: insufficient temporal reliability;
- TREFIC: safety without Large-LI positive power;
- DGR-F1: good three-way discrimination but invalid null coverage and
  insufficient strict harm power at the registered budget.

The evidence no longer supports another immediate certification variant. The
remaining defensible output is a negative reliability benchmark and an
explicit limitation result. A new methods-journal mainline would require an
independently derived inferential model for discrete, rare-positive,
dependent F1, not another interval substitution chosen on these scenarios.

## Reproducibility

- Seven distinct GPU tasks used GPUs 0 through 6.
- Seeds were `76001` through `76007`.
- All tasks used one clean remote Git commit.
- Seven manifests completed without runtime errors.
- Complete machine-readable results are stored in
  `DGR_F1_PHASE0_DEVELOPMENT_RESULTS.json`.

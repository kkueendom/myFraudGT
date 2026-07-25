# Dynamic Evidence Risk-Control Feasibility Results

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Status: completed, gate failed
- Audit commit: `699afa70`
- Sampling protocol: `dynamic_random`
- Data scope: train-only target-edge OOF
- Validation loader iterations: 0
- Test loader iterations: 0
- Remote result:
  `/e/yky/FraudGT_cet_results/risk_feasibility_699afa7`
- Local result:
  `/Users/kun/FraudGT_experiment_workspace/risk_feasibility_699afa7`

## Design

The audit reused the six completed TIER Phase 2b OOF score tables:

- Small-LI seed 42, folds 0-2;
- Large-LI seed 44, folds 0-2.

For each held-out fold, one other fold trained the add/remove utility probes and
the remaining fold selected a harm-qualified monotone policy. The policy was
then frozen before evaluation on the held-out fold. Normal, shuffled and off
evidence used the same selected policy family.

The stored OOF artifacts do not contain entity or timestamp identifiers.
Consequently, the Wilson bounds are row-level descriptive bounds, not formal
graph-dependence guarantees.

## Held-Out Results

### Small-LI Normal Evidence

| Held-out fold | Changed | Corrected/broken | Net | Paired F1 delta | Break upper 95% |
|---:|---:|---:|---:|---:|---:|
| 0 | 9 | 6/3 | +3 | -0.02973 | 0.60174 |
| 1 | 17 | 15/2 | +13 | +0.00319 | 0.30056 |
| 2 | 12 | 11/1 | +10 | +0.01255 | 0.30117 |
| **Total** | **38** | **32/6** | **+26** | **-0.01399 sum** | n/a |

The intervention corrects more rows than it breaks, but it changes only 38
predictions and decreases the sum of held-out fold F1 by 0.01399. Fold 0 also
violates the registered 0.40 harm upper bound. The corrections are dominated
by decisions that do not translate into a stable positive-class F1 gain.

Control totals:

| Condition | Changed | Corrected/broken | Net | Mean paired F1 delta |
|---|---:|---:|---:|---:|
| Normal | 38 | 32/6 | +26 | -0.00466 |
| Shuffled | 66 | 39/27 | +12 | -0.15278 |
| Off | 71 | 40/31 | +9 | -0.17997 |

Aligned evidence is less harmful than the controls, but this mechanism
difference does not create usable F1 coverage.

### Large-LI Normal Evidence

| Held-out fold | Changed | Corrected/broken | Net | Paired F1 delta |
|---:|---:|---:|---:|---:|
| 0 | 0 | 0/0 | 0 | 0.00000 |
| 1 | 0 | 0/0 | 0 | 0.00000 |
| 2 | 17 | 17/0 | +17 | 0.00000 |
| **Total** | **17** | **17/0** | **+17** | **0.00000 sum** |

Only one held-out fold admits an intervention. The policy removes false
positives while the fold has zero positive-class F1, so the 17 corrections do
not change F1. Shuffled and off controls change 19 and 21 rows respectively
and also have zero F1 delta. Normal evidence therefore does not demonstrate a
specific utility advantage.

## Registered Gate

| Dataset | Changed >= 50 | Corrected > broken | Ratio >= 1.5 | F1 delta > 0 | Normal > shuffled | Positive net in >=2 folds | Final |
|---|---|---|---|---|---|---|---|
| Small-LI | Fail | Pass | Pass | Fail | Pass | Pass | **Fail** |
| Large-LI | Fail | Pass | Pass | Fail | Fail | Fail | **Fail** |

## Decision

`STOP_PREDICTIVE_EVIDENCE_INTERVENTION`.

The same train-only OOF evidence has now failed:

1. the original locked directional utility policy;
2. the leave-one-fold-out harm-controlled policy;
3. the CET representation-fusion two-scale gate;
4. repeated dynamic CET reliability evaluation.

Do not train another router, decoder gate or CET encoder on these evidence
proposals. Do not search more thresholds on the same OOF labels.

The active research direction becomes an evaluation-method question:

> How should temporal evidence sensitivity, corrective utility and dynamic
> sampling variability be separated when evaluating fraud graph models?

The next valid work is a preregistered, multi-model reliability benchmark.
Prediction improvement is no longer the claim of this branch.

# CPSE Phase 0B Train-Only OOF Results

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Status: formal train-only OOF qualification completed; gate failed
- CPSE training commit: `a78d3396`
- Utility audit commit: `4bf1703c`
- Formal feature output:
  `/e/yky/FraudGT_cet_results/cpse_phase0b_formal_a78d3396`
- Formal audit output:
  `/e/yky/FraudGT_cet_results/cpse_phase0b_utility_4bf1703c`
- Local feature mirror:
  `/Users/kun/FraudGT_experiment_workspace/cpse_phase0b_formal_a78d3396`
- Local audit mirror:
  `/Users/kun/FraudGT_experiment_workspace/cpse_phase0b_utility_4bf1703c`
- Machine aggregate:
  `docs/experiments/CPSE_PHASE0B_OOF_RESULTS.json`
- Sampling protocol: `dynamic_random`
- Validation/test loader iterations: 0
- Fraud labels used for CPSE training: no

## Execution Audit

The six registered train-only OOF tasks completed without runtime, CUDA, or
manifest errors. GPUs 0-5 ran one distinct dataset/fold task each. GPU 6 ran
the locked grouped utility audit after all six feature tables were available.

| Dataset | Fold | Seed | OOF edges | Positives | Epochs | Best self-supervised loss | Runtime (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Small-LI | 0 | 42 | 174,832 | 92 | 20 | 0.344273 | 299.1 |
| Small-LI | 1 | 42 | 174,450 | 84 | 20 | 0.346249 | 285.0 |
| Small-LI | 2 | 42 | 174,577 | 63 | 20 | 0.344585 | 278.5 |
| Large-LI | 0 | 44 | 34,796 | 29 | 20 | 0.305398 | 1,902.8 |
| Large-LI | 1 | 44 | 34,975 | 15 | 20 | 0.309167 | 1,850.0 |
| Large-LI | 2 | 44 | 34,589 | 28 | 20 | 0.306941 | 2,015.8 |

Every exported table has 31 finite normal, shuffled, and off features per
edge. Edge IDs are unique within each fold. Held-out target edges were
excluded from CPSE loss, and all manifests record zero validation and test
iterations.

## Formal Gate

| Requirement | Small-LI | Large-LI |
|---|---|---|
| At least 50 interventions | fail: 0 | pass: 62 |
| Corrected greater than broken | fail: 0 vs 0 | pass: 59 vs 3 |
| Corrected/broken at least 1.5 | fail | pass: 19.67 |
| Positive summed paired F1 delta | fail: 0.0000 | fail: -0.1000 |
| Normal exceeds shuffled | fail | fail |
| Positive net correction in at least 2 folds | fail: 0 | pass: 2 |
| Utility AUPRC exceeds prevalence by 0.05 | pass | fail |
| Utility AUPRC exceeds shuffled by 0.02 | pass | fail |
| Nonempty GTPRC policy | fail | pass |

Neither dataset passes all registered requirements. The joint decision is
`STOP_CPSE`.

## Evidence Diagnostics

| Dataset | Normal utility AUPRC | Shuffled AUPRC | Utility prevalence | Normal changed | Corrected | Broken | Mean paired F1 delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| Small-LI | 0.42489 | 0.28589 | 0.24900 | 0 | 0 | 0 | 0.00000 |
| Large-LI | 0.49443 | 0.49169 | 0.48594 | 62 | 59 | 3 | -0.03333 |

Small-LI shows that normal CPSE features predict the utility target better
than shuffled features, but no add or remove policy survives the locked
graph-time calibration rule. Predictive evidence alone therefore does not
produce an admissible intervention.

Large-LI shows the opposite failure. The calibrated remove policy changes 62
predictions and eliminates 59 false positives while breaking 3 true
positives. Because fraud positives are extremely rare, those three broken
true positives dominate F1: one evaluation fold falls from A2 F1 0.10 to
0.00, while the other two folds have zero F1 change. Normal, shuffled, and off
conditions are almost identical, so the interventions cannot be attributed to
the learned CPSE representation.

## Interpretation

The formal audit rejects the CPSE mainline for two independent reasons:

1. representation utility is not consistently identifiable across scales;
2. a row-count harm objective can improve net classification correctness
   while reducing fraud F1 when true positives are rare.

The second finding also narrows the GTPRC claim. Its controlled simulation
validated grouped harm control for the registered row-level outcome, but that
outcome is not sufficient for F1-sensitive routing on highly imbalanced AML
data.

## Claim Boundary

This experiment supports only a negative qualification conclusion. It does
not establish that:

- CPSE improves FraudGT validation or test F1;
- the Small-LI AUPRC gap will yield a useful end-to-end model after tuning;
- positive net corrections imply positive F1;
- changing GTPRC thresholds after observing these results would be a valid
  confirmation experiment.

No validation or test loader was opened, no A2 model was retrained, and no
full CPSE-FraudGT fusion experiment is authorized by this gate.

## Decision

Stop CPSE and do not tune its decoder, router threshold, or risk budget.
Before another full model is trained, any successor must qualify a
representation-level signal on train-only OOF data and use an objective whose
held-out decision utility is aligned with F1 under extreme class imbalance.

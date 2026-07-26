# Dynamic-Sampling Evidence Reliability Results

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Status: completed
- Audit commit: `33aab60f`
- Machine aggregation commit: `a08f901`
- Model checkpoint commit: `cd72ba5b`
- Sampling protocol: `dynamic_random`
- Remote results:
  `/e/yky/FraudGT_cet_results/dynamic_reliability_33aab60`
- Local results:
  `/Users/kun/FraudGT_experiment_workspace/cet_dynamic_reliability_33aab60`
- Evaluation events: 28
- Machine-readable aggregate:
  `docs/experiments/CET_DYNAMIC_RELIABILITY_RESULTS.json`

## Design

Seven processes evaluated four fixed CET v1 checkpoints. Every process created
the original dynamic val/test loaders once, then completed four consecutive
full 256-step val/test passes without resetting RNG state. Normal, shuffled,
off and frozen-A2 scores were evaluated on the same sampled test batches.

The seven streams comprised:

- two Small-LI fusion streams;
- three Large-LI fusion streams;
- one encoder-only stream per dataset.

Reported standard deviations are descriptive sample standard deviations over
dynamic evaluation events. They are not independent-seed model uncertainty
and are not used for significance claims.

## Cross-Stream Results

Mean +/- descriptive sample standard deviation:

| Dataset | Variant | Events | Normal F1 | Delta vs initial A2 | Same-batch A2 F1 | Delta vs same-batch A2 |
|---|---|---:|---:|---:|---:|---:|
| Small-LI | encoder-only | 4 | 0.05798 +/- 0.00614 | -0.40449 +/- 0.00614 | 0.45404 +/- 0.02770 | -0.39606 +/- 0.02872 |
| Small-LI | fusion | 8 | 0.28146 +/- 0.02015 | -0.18101 +/- 0.02015 | 0.45265 +/- 0.02344 | -0.17119 +/- 0.01859 |
| Large-LI | encoder-only | 4 | 0.07572 +/- 0.01646 | -0.22536 +/- 0.01646 | 0.35440 +/- 0.03902 | -0.27868 +/- 0.03266 |
| Large-LI | fusion | 12 | 0.32108 +/- 0.04757 | +0.02000 +/- 0.04757 | 0.33150 +/- 0.03586 | -0.01042 +/- 0.02743 |

Large-LI fusion ranges from 0.24615 to 0.38323 F1. Its Delta versus the
historical initial-A2 reference ranges from -0.05493 to +0.08215. Therefore
the single Phase B gain of +0.01760 is not a stable model improvement.

The historical A2 remains the registered baseline for formal reporting. The
same-batch frozen A2 is a paired diagnostic, not a replacement baseline. It
shows that the apparent positive mean against the historical A2 is explained
by dynamic evaluation variation: on the same target batches, fusion is worse
on average.

## Mechanism Reliability

| Dataset | Variant | Normal-shuffled F1 | Normal-off F1 | Changed | Corrected | Broken | Corrected-broken |
|---|---|---:|---:|---:|---:|---:|---:|
| Small-LI | encoder-only | +0.05341 +/- 0.00732 | +0.05008 +/- 0.00589 | 999.00 +/- 276.24 | 62.75 +/- 22.63 | 936.25 +/- 266.19 | -873.50 +/- 257.73 |
| Small-LI | fusion | +0.00142 +/- 0.00485 | +0.28146 +/- 0.02015 | 248.00 +/- 29.29 | 34.38 +/- 10.78 | 213.63 +/- 26.91 | -179.25 +/- 28.68 |
| Large-LI | encoder-only | +0.06557 +/- 0.01596 | +0.07267 +/- 0.01663 | 473.00 +/- 337.89 | 27.50 +/- 5.07 | 445.50 +/- 335.37 | -418.00 +/- 332.91 |
| Large-LI | fusion | +0.29120 +/- 0.05111 | +0.01149 +/- 0.01839 | 23.42 +/- 10.17 | 10.50 +/- 6.05 | 12.92 +/- 13.17 | -2.42 +/- 17.80 |

Small-LI fusion is sensitive to evidence presence but not alignment: off
history collapses the selected-threshold F1, while shuffled history behaves
like normal history. Its corrected-minus-broken value is negative in all eight
events.

Large-LI fusion is strongly sensitive to alignment, but aligned evidence does
not produce stable utility. Corrected-minus-broken is positive in 6 events,
zero in 1 and negative in 5; it ranges from -46 to +18. The model changes only
12-50 predictions per event, below the registered minimum in all but one event.

Large-LI fusion also has severe threshold instability. Its val-derived model
threshold is 0.45095 +/- 0.25012 and ranges from 0.20028 to 0.92377.

## Support-Conditioned Audit

Counts below pool sampled rows over all repeated events. They are exposure
counts, not unique transaction counts.

### Small-LI Fusion

| Support | Sampled rows | Positives | Changed | Corrected | Broken | Net |
|---|---:|---:|---:|---:|---:|---:|
| 1-7 | 249,248 | 163 | 66 | 1 | 65 | -64 |
| 8-23 | 192,437 | 416 | 461 | 26 | 435 | -409 |
| 24-47 | 619,705 | 632 | 766 | 62 | 704 | -642 |
| 48 | 3,095,825 | 1,630 | 691 | 186 | 505 | -319 |

No nonempty support range has positive net correction.

### Large-LI Fusion

| Support | Sampled rows | Positives | Changed | Corrected | Broken | Net |
|---|---:|---:|---:|---:|---:|---:|
| 1-7 | 15,390 | 1 | 2 | 2 | 0 | +2 |
| 8-23 | 65,596 | 44 | 3 | 2 | 1 | +1 |
| 24-47 | 66,042 | 178 | 38 | 14 | 24 | -10 |
| 48 | 719,511 | 1,066 | 238 | 108 | 130 | -22 |

The low/medium-support rows have only five interventions across 80,986 sampled
rows. Their positive net count is too small to define a usable reliability
region. Higher-support rows contain nearly all interventions and have negative
net correction. Support count alone is therefore not a reliability estimate.

## Findings

1. Aligned causal history can carry signal without carrying stable corrective
   utility.
2. A normal-shuffled gap is necessary but not sufficient for useful fusion.
3. Dynamic val thresholds and sampled positive composition can reverse the
   sign of an apparent F1 gain.
4. Comparison with the historical initial A2 is required for protocol
   continuity, but a paired same-batch frozen-base diagnostic is necessary to
   identify sampling-induced apparent gains.
5. Support volume does not predict safe intervention.
6. CET v1 cannot be repaired by objective-weight ablations because its failure
   is cross-scale, decision-level and reliability-related.

The machine-validated mechanism classification is:

- sensitive-but-harmful: `3/4` dataset/variant pairs;
- used-but-unaligned: `1/4` (`Small-LI fusion`);
- useful-aligned: `0/4`;
- pairs satisfying their classification conditions in at least 75% of
  dynamic events: `3/4`;
- pairs whose single-event conclusion can reverse: `1/4`
  (`Large-LI fusion`).

Large-LI fusion is the non-repeatable pair. Its aggregate mean is classified
as sensitive-but-harmful, but the harmful-utility direction occurs in only
`8/12` events, below the preregistered `9/12` threshold. The aggregate label
must therefore not be treated as an event-consistent result.

## Decision

`CONFIRM_STOP_CET_V1`.

Do not expand, tune or rerun CET v1. The unused diagnostic-loss ablations remain
cancelled.

## New Research Mainline

The new paper question is:

> How can temporal evidence reliability be measured and controlled under
> dynamic fraud-graph sampling when evidence sensitivity does not imply
> corrective utility?

The candidate methodological contribution is a paired counterfactual
reliability framework with:

1. repeated dynamic sampling streams;
2. same-batch normal/shuffled/off/base evaluation;
3. explicit separation of evidence sensitivity from corrective utility;
4. support- and error-subset conditional reliability maps;
5. qualification gates that prevent unstable evidence from entering a
   predictive model.

This is an evaluation and reliability-method direction, not another decoder or
fusion-head variant. It requires a new literature and novelty review before
additional model training.

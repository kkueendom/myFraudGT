# Epoch-499 A2 Checkpoint Dynamic-Sampling Stability Results

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Status: completed
- Runner commit: `1ebda33e`
- Aggregation commit: `addc98f`
- Sampling protocol: `dynamic_random`
- Models: six fixed epoch-499 A2 checkpoints; no retraining
- Streams: 2 per dataset
- Events: 8 per stream, 16 per dataset, 96 total
- Val/test iterations: 256 per event
- Remote result:
  `/e/yky/FraudGT_cet_results/a2_stability_formal_1ebda33`
- Local result:
  `/Users/kun/FraudGT_experiment_workspace/a2_stability_formal_1ebda33`
- Machine-readable aggregate:
  `docs/experiments/A2_DYNAMIC_SAMPLING_STABILITY_RESULTS.json`
- Formal baseline: registered historical initial A2 table

## Protocol Audit

All 12 tasks satisfy:

- `LinkNeighborLoader(..., shuffle=True)`;
- `val.fixed_target_panel=False`;
- no fixed target edge panel;
- no dedicated evaluation generator;
- no sampler RNG restoration;
- process RNG seeded once at stream start;
- 2,048 validation and 2,048 test loader iterations per stream;
- no training or checkpoint update.

The formal queue completed 12/12 manifests and 96/96 events with no runtime,
alignment or CUDA errors.

## Critical Checkpoint Boundary

The historical Val-selected A2 scores were selected at different epochs:

| Dataset | Historical A2 F1 | Selected epoch | Available audited checkpoint |
|---|---:|---:|---:|
| Small-LI | 0.46247 | 167 | 499 |
| Small-HI | 0.77984 | 251 | 499 |
| Medium-LI | 0.51163 | 411 | 499 |
| Medium-HI | 0.77574 | 383 | 499 |
| Large-LI | 0.30108 | 445 | 499 |
| Large-HI | 0.72897 | 397 | 499 |

Therefore the epoch-499 mean minus historical A2 combines checkpoint-epoch and
dynamic-sampling effects. It is a diagnostic number only. It cannot be called
sampling bias and cannot replace the registered formal baseline.

## Aggregate Results

| Dataset | Historical A2 | Epoch-499 F1 mean +/- sd | Min-max | Diagnostic mean delta | Event band 95% | Independent comparison band |
|---|---:|---:|---:|---:|---:|---:|
| Small-LI | 0.46247 | 0.44397 +/- 0.01427 | 0.42065-0.46886 | -0.01850 | +/-0.02193 | +/-0.03954 |
| Small-HI | 0.77984 | 0.76677 +/- 0.00886 | 0.74883-0.78294 | -0.01307 | +/-0.01588 | +/-0.02455 |
| Medium-LI | 0.51163 | 0.45946 +/- 0.04176 | 0.39572-0.55497 | -0.05217 | +/-0.08477 | +/-0.11574 |
| Medium-HI | 0.77574 | 0.75259 +/- 0.01602 | 0.73113-0.78003 | -0.02315 | +/-0.02553 | +/-0.04441 |
| Large-LI | 0.30108 | 0.33138 +/- 0.05919 | 0.22222-0.46667 | +0.03030 | +/-0.12069 | +/-0.16407 |
| Large-HI | 0.72897 | 0.72042 +/- 0.02440 | 0.67474-0.76481 | -0.00855 | +/-0.04184 | +/-0.06762 |

The event band is the central 95% spread around each fixed epoch-499
checkpoint mean. The independent comparison band is
`1.96 * sqrt(2) * event_sd`; it is a descriptive approximation for two
independent dynamic evaluations, not a formal guarantee.

## Threshold and Label Variation

| Dataset | Val threshold mean +/- sd | Test positives mean +/- sd | Test prevalence mean +/- sd |
|---|---:|---:|---:|
| Small-LI | 0.66877 +/- 0.08533 | 356.19 +/- 10.75 | 0.00069 +/- 0.00002 |
| Small-HI | 0.81579 +/- 0.03955 | 961.50 +/- 19.35 | 0.00185 +/- 0.00004 |
| Medium-LI | 0.50263 +/- 0.12648 | 144.38 +/- 10.54 | 0.00060 +/- 0.00004 |
| Medium-HI | 0.73983 +/- 0.08729 | 405.25 +/- 18.25 | 0.00168 +/- 0.00008 |
| Large-LI | 0.42491 +/- 0.12890 | 110.63 +/- 9.55 | 0.00153 +/- 0.00013 |
| Large-HI | 0.74739 +/- 0.07462 | 306.94 +/- 13.56 | 0.00423 +/- 0.00018 |

Every event has a test unique-edge rate of 1.0. Thus the variation is not
caused by duplicate target edges within an event. It is consistent with
changes in the dynamically sampled target population, neighborhoods, positive
count and val-derived threshold.

## Findings

1. The uniform `0.005` warning threshold is smaller than the measured
   event-level 95% band on every dataset.
2. Sampling sensitivity is strongly scale-dependent. Medium-LI and Large-LI
   have event bands of 0.08477 and 0.12069, while the HI datasets are more
   stable but still exceed 0.005.
3. Val-derived thresholds are unstable, with standard deviation from 0.03955
   to 0.12890.
4. A one-event gain on Large-LI can easily reverse sign under another dynamic
   stream. This explains why the earlier CET +0.01760 was not reproducible.
5. Historical initial A2 remains the formal baseline, but future model claims
   require repeated streams or seeds and a same-batch diagnostic where
   possible.

## Reporting Rule

- Keep the registered historical initial A2 values in the main table.
- Mark `|Delta F1| < 0.005` as possible sampling variation as required.
- Do not treat a gain above 0.005 as stable by itself.
- For model claims, report repeated-stream or multi-seed mean and uncertainty.
- Prefer paired same-batch base/model evaluation for evidence interventions.
- Do not compare an epoch-499 re-evaluation mean directly with a historical
  Val-selected score as if the checkpoint were identical.

## Decision

`PROCEED_TO_MULTI_MODEL_RELIABILITY_BENCHMARK`.

The six-dataset A2 audit establishes that dynamic sampling variability is
large enough to change method conclusions. A journal-level evaluation method
still requires generalization beyond one checkpoint family. The next stage
must audit a preregistered set of fixed TIER/CET/COSTAR evidence models without
new tuning and test whether sensitivity-versus-utility failures recur across
evidence types.

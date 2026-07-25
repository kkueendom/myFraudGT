# CET Dynamic-Sampling Reliability Audit

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Status: preregistered, not yet executed
- Parent decision: `STOP_CET_V1`
- Sampling protocol: `dynamic_random`
- Purpose: characterize evidence reliability, not tune CET v1

## Research Question

Under the original FraudGT dynamic-random sampling protocol, are aligned
temporal-evidence effects and error corrections stable across independently
initialized sampling streams, and under which support levels do they help or
harm a frozen A2 checkpoint?

## Fixed Design

The audit loads the already trained, Val-selected CET v1 checkpoints. It does
not train, tune or select a new model.

Each process:

- initializes the process RNG once before loader creation;
- creates the original val/test dynamic loaders with `shuffle=True`;
- leaves `val.fixed_target_panel=False`;
- does not create a dedicated evaluation generator;
- does not save or restore sampler RNG state;
- evaluates normal, shuffled, off and frozen-base scores on the same test
  batches;
- repeats the complete dynamic val/test pass four times without resetting RNG.

## Registered GPU Tasks

| GPU | Dataset | Checkpoint | Audit seed | Repeats |
|---:|---|---|---:|---:|
| 0 | Small-LI | fusion | 4201 | 4 |
| 1 | Small-LI | fusion | 4202 | 4 |
| 2 | Large-LI | fusion | 4401 | 4 |
| 3 | Large-LI | fusion | 4402 | 4 |
| 4 | Small-LI | encoder-only | 4211 | 4 |
| 5 | Large-LI | encoder-only | 4411 | 4 |
| 6 | Large-LI | fusion | 4403 | 4 |

The additional fusion streams are independent sampling-reliability
replications, not attempts to find a favorable model seed. Large-LI receives
three streams because its single Phase B event was directionally ambiguous;
Small-LI receives two because its failure margin was already large.

## Required Outputs

Every stream records:

- model checkpoint, checkpoint epoch and model commit;
- audit commit, model seed and audit seed;
- loader protocol audit;
- per-repeat val thresholds and full test metrics;
- Delta versus initial A2 and same-batch frozen A2;
- normal-shuffled and normal-off F1 gaps;
- changed, corrected and broken predictions;
- unique-edge sampling rate;
- support-conditioned correction, break and score-sensitivity summaries;
- representation diagnostics and runtime;
- mean, sample standard deviation, minimum and maximum over repeats.

## Decision Rule

This audit cannot revive CET v1. It determines the next research problem:

- stable aligned-history sensitivity but unstable correction implies a
  reliability/calibration problem;
- unstable normal-shuffled effects imply a dynamic-sampling evidence problem;
- consistently negative corrected-minus-broken implies that evidence should be
  studied diagnostically rather than fused for prediction;
- only a stable, support-localized positive correction region can justify a
  future train-only reliability method.


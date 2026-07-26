# CPSE Phase 0B Train-Only OOF Qualification Plan

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Status: preregistered, not started
- Parent gate: `GTPRC_PHASE0A_V2C_FORMAL_RESULTS.md`
- Sampling protocol: `dynamic_random`
- Validation/test loader iterations allowed: 0
- A2 retraining: no
- Full FraudGT fusion training: no

## Evidence Model

CPSE learns causal next-transaction behavior without fraud labels.

Input:

- up to 32 target-admissible incident transactions;
- source/destination role flags;
- timestamp, normalized amount, currency, and payment format;
- six causal support statistics.

Self-supervised targets:

- target normalized amount;
- log time since the latest admissible incident transaction;
- target currency;
- target payment format.

The encoder pools source, destination, and global histories. Its detached
evidence vector contains:

- signed and absolute amount surprise;
- signed and absolute interarrival surprise;
- currency and payment negative log likelihood;
- categorical confidence;
- causal support;
- a compact predictive history state.

The model never optimizes fraud classification.

## OOF Contract

Reuse the six frozen A2 OOF tables from:

`/e/yky/FraudGT_tier2b_results/crossfit_c78214c1`

For each dataset and fold:

1. instantiate the original train/val/test loaders with `shuffle=True`;
2. iterate only the train loader;
3. exclude held-out fold target edges from CPSE self-supervised loss;
4. allow held-out events as unlabeled causal graph context;
5. train for at most 20 epochs with train-loss early stopping patience 4;
6. collect normal, shuffled, and off CPSE vectors on the exact frozen A2 OOF
   edge IDs;
7. store source, destination, timestamp, edge ID, A2 score, A2 threshold, and
   fold;
8. load fraud labels only after CPSE training for utility evaluation.

The runner must record `fraud_labels_used_for_cpse_training=false`,
`validation_loader_iterations=0`, and `test_loader_iterations=0`.

## Six GPU Tasks

| GPU | Dataset | Held-out fold |
|---:|---|---:|
| 0 | Small-LI | 0 |
| 1 | Small-LI | 1 |
| 2 | Small-LI | 2 |
| 3 | Large-LI | 0 |
| 4 | Large-LI | 1 |
| 5 | Large-LI | 2 |

GPU 6 remains available for aggregation or another preregistered distinct
task. It must not run a duplicate fold.

## Cross-Fold Utility Evaluation

For each evaluation fold:

- one other fold fits independent add and remove logistic utility probes;
- the remaining fold calibrates locked GTPRC policies;
- the evaluation fold reports held-out utility;
- roles rotate deterministically across all three folds.

Probe inputs are detached A2 score, A2 margin, and CPSE evidence vector.
Normal trains the probe and policy. Shuffled and off vectors are evaluated
with the same frozen probe and policy thresholds.

Graph-time groups are non-overlapping chronological blocks with within-block
entity components. The independence assumption is explicit and remains an
empirical approximation until a real-graph proof is completed.

## Gate

Both Small-LI and Large-LI must satisfy:

- at least 50 total held-out interventions;
- corrected greater than broken;
- corrected/broken at least 1.5;
- positive summed held-out paired F1 delta;
- normal exceeds shuffled by at least 0.01 paired F1 or ten net corrections;
- at least two of three held-out folds have positive net correction;
- normal utility AUPRC exceeds utility prevalence by at least 0.05;
- normal utility AUPRC exceeds shuffled by at least 0.02;
- locked GTPRC selects a nonempty policy whose registered harm bound is at
  most 0.40.

Failure on either dataset stops CPSE. Threshold tuning, decoder additions,
validation/test evaluation, and full-model training remain prohibited.


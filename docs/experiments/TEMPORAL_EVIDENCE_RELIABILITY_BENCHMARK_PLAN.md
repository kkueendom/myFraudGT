# Temporal Evidence Reliability Benchmark Plan

## Material Passport

- Origin skill: academic-research-suite / experiment-agent plan mode
- Status: preregistered, not started
- Branch: `paper/temporal-evidence-reliability-benchmark`
- Paper question: temporal evidence reliability under dynamic fraud-graph
  sampling
- Fixed model family: evidence-gate v4 full model
- Training: none
- Checkpoints: 18 existing epoch-499 checkpoints
- Sampling protocol: FraudGT `dynamic_random`

## Purpose

The preceding predictive and statistical methods failed their registered
advancement gates. This experiment does not introduce another model. It asks:

> How much of a reported fraud F1 difference is attributable to model-training
> seed, independent dynamic sampling stream, and event-to-event sampler
> variation?

The existing A2 audit used one fixed checkpoint per dataset. The v4 archive
contains three independently trained seeds for all six AML datasets, making a
balanced nested variance study possible without retraining or test-based model
selection.

## Fixed Design

| Factor | Levels |
|---|---:|
| Dataset | 6 |
| Model-training seed | 3: 42, 43, 44 |
| Independent dynamic stream | 2: 93001, 93002 |
| Sequential events per stream | 4 |
| Total tasks | 36 |
| Total dynamic events | 144 |
| Validation iterations per event | 256 |
| Test iterations per event | 256 |

All checkpoints are fixed at epoch 499. No checkpoint is selected from these
events.

The same audit seed is reused across the three model seeds within a dataset.
This is intentional blocking: if target-edge hashes align, model-seed
comparisons use the same dynamic target sequence. The two audit seeds remain
independent process streams.

## Dynamic-Random Contract

Every task must preserve:

- train/validation/test loader construction from the archived configuration;
- `LinkNeighborLoader(..., shuffle=True)`;
- `val.fixed_target_panel=False`;
- no fixed target-edge panel;
- no independent evaluation generator;
- no sampler RNG restoration before an event;
- process RNG seeded once at stream start;
- original batch size and `val.iter_per_epoch`;
- no training, optimizer step or checkpoint update.

The test F1 threshold is selected from that event's dynamic validation sample,
matching the original FraudGT evaluation flow.

## Registered Checkpoints

Root:

`/e/yky/FraudGT_evidence_gate_v4/results/evidence_gate_v4_formal`

For every dataset, the archived run directory is
`AML-{dataset}-gpu0`, with one shared `config.yaml` and checkpoints:

```text
42/ckpt/499.ckpt
43/ckpt/499.ckpt
44/ckpt/499.ckpt
```

All 18 files were verified to exist before preregistration.

## Recorded Event Data

Each event records:

- validation and test F1, precision, recall and threshold;
- TP, FP, FN, TN and positive count;
- sampled target count and unique-edge rate;
- validation and test target-edge SHA-256 hashes;
- model seed, audit seed, repeat index, commit, config and checkpoint;
- dynamic protocol audit fields;
- elapsed evaluation time.

Raw target labels and predictions are not used to tune a method.

## Primary Analysis

For each dataset, let `Y_msr` be test F1 for model seed `m`, stream `s` and
event `r`.

Use the balanced random-effects decomposition:

```text
Y_msr = mu + A_m + B_ms + epsilon_msr
```

where:

- `A_m` is model-training seed variation;
- `B_ms` is independent stream variation nested within model seed;
- `epsilon_msr` is sequential event variation within a stream.

Method-of-moments estimates:

```text
V_event = MS_event
V_stream = max((MS_stream - MS_event) / R, 0)
V_model = max((MS_model - MS_stream) / (S R), 0)
```

with `M=3`, `S=2`, and `R=4`.

Report:

- all three variance components;
- sampling share `(V_stream + V_event) / V_total`;
- model-seed share `V_model / V_total`;
- event-level min, max, mean and standard deviation;
- validation-threshold variation;
- raw-event maximum inflation above each checkpoint's event mean.

## Ranking and Sign Diagnostics

1. Rank the three model seeds by four-event mean within each independent
   stream.
2. Report pairwise ranking disagreements between streams.
3. Compare each event with the historical initial A2 using the same metric
   only as a diagnostic.
4. Count datasets where the sign of this diagnostic difference changes across
   events.
5. Do not call the historical-A2 difference sampling bias because model family
   and checkpoint differ.

## Preregistered Benchmark Readiness Gate

All protocol requirements must pass. The benchmark is considered nontrivial
only if at least one empirical condition holds:

1. sampling variance is at least model-seed variance on two or more datasets;
2. model-seed ranking changes between the two dynamic streams on at least one
   dataset;
3. the event-level diagnostic difference versus initial A2 changes sign on at
   least two datasets;
4. raw-event maximum inflation is at least 0.01 on four or more datasets.

This is a benchmark-readiness gate, not a model-success gate. Failure does not
authorize parameter tuning or new model training; it means the negative paper
needs an external dataset or a different scope.

## Integrity Gate

Before analysis:

- 36/36 manifests and 144/144 events must complete;
- every checkpoint must be epoch 499;
- every task must report `sampling_protocol=dynamic_random`;
- validation/test target-edge unique rate must be 1.0;
- for each dataset/audit seed/repeat, target-edge hashes must match across all
  three model seeds;
- no task may read another task's event outputs;
- the repository and remote worktree must be clean at launch.

## GPU Allocation

Use seven worker queues, one per available GPU. Each worker processes a
disjoint stride of task indices. At most seven tasks run concurrently, and no
checkpoint/stream pair is duplicated.


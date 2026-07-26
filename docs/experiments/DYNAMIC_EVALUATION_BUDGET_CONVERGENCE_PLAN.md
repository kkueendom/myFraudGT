# Dynamic Evaluation Budget Convergence Plan

## Material Passport

- Origin skill: academic-research-suite / experiment-agent plan mode
- Branch: `paper/temporal-evidence-reliability-benchmark`
- Status: preregistered before any formal result is inspected
- Sampling protocol: `dynamic_random`
- Training or tuning: none
- Fixed model family: evidence-gate v4 full model
- Fixed checkpoints: epoch 499, model seeds 42, 43 and 44
- Datasets: six AML scale/imbalance variants

## Research Question

How many dynamically sampled validation and test batches are required before
the reported test F1 is close to the result obtained from the original
256-batch evaluation budget?

This experiment addresses a protocol-design question. It does not claim that
the 256-batch result is the population truth, and it does not compare or tune
predictive models.

## Motivation

The nested reliability benchmark showed that dynamic evaluation variation
accounts for 71.5%-100% of the estimated F1 variance. It did not determine
whether this problem can be reduced by a specific evaluation budget. A methods
paper needs an actionable reporting recommendation rather than only a failure
diagnosis.

## Fixed Design

The factorial design contains:

- 6 AML datasets;
- 3 fixed model-training seeds: 42, 43 and 44;
- 2 independent audit streams: 94001 and 94002;
- 8 complete events per stream;
- 7 nested evaluation budgets: 4, 8, 16, 32, 64, 128 and 256 batches;
- 36 GPU tasks and 288 complete dynamic events.

Each event first evaluates all 256 validation batches and then all 256 test
batches exactly once. Metrics for smaller budgets are computed from prefixes
of the same predictions. Thus every budget comparison is paired on model,
stream, event and sampled target-edge order.

The full workload is distributed over seven GPUs by task index. No duplicate
job is launched solely to occupy a GPU.

## Dynamic-Random Invariants

Every formal task must satisfy all of the following:

- train, validation and test loaders retain dynamic random sampling;
- `LinkNeighborLoader` retains `shuffle=True`;
- `val.fixed_target_panel=False`;
- no fixed target-edge panel;
- no independent evaluation generator;
- no sampler RNG restoration;
- the archived batch size and `val.iter_per_epoch=256` are unchanged;
- all three model seeds within a dataset/audit-stream/event observe the same
  target-edge sequence at every nested budget;
- repository status is clean and the experiment commit is recorded.

## Estimands

For event \(i\), dataset \(d\), checkpoint \(m\), stream \(s\), and budget
\(b\), let \(F_{dmsi}(b)\) be test F1 using the validation-selected threshold
from the same budget. The paired full-budget deviation is:

\[
E_{dmsi}(b) = F_{dmsi}(b) - F_{dmsi}(256).
\]

The primary error is:

\[
A_{dmsi}(b) = |E_{dmsi}(b)|.
\]

For each dataset and budget, report:

- mean and median paired absolute error;
- 90th and 95th percentile paired absolute error;
- root mean squared paired error;
- signed bias relative to the 256-batch event;
- probability that the sign of the delta versus historical initial A2 agrees
  with the corresponding 256-batch event;
- probability that absolute error is at most 0.005 and at most 0.01;
- validation-threshold absolute deviation;
- sampled positives, prevalence and unique-target-edge rate.

Results are also pooled across datasets with equal dataset weighting. Raw
events remain available for cluster-aware uncertainty analysis.

## Recommendation Rule

For a dataset, a budget is called adequate only if all larger registered
budgets also satisfy:

- median absolute F1 error at most 0.005;
- 90th percentile absolute F1 error at most 0.01;
- delta-sign agreement with the 256-batch event at least 0.90;
- validation and test unique-target-edge rate equal to 1.0.

The cross-dataset minimum recommendation is the smallest budget adequate on at
least 5 of 6 datasets. If no budget below 256 passes, the recommendation is to
retain the full 256-batch evaluation and add independent repeated streams.

The adequacy rule is descriptive and protocol-specific. It is not a formal
confidence guarantee.

## Integrity Gates

The formal run is valid only if:

- exactly 36 manifests and 288 complete events are present;
- every event contains all seven budgets;
- every task uses the same clean experiment commit;
- all checkpoints resolve to epoch 499;
- all protocol flags equal the registered dynamic-random values;
- every full event consumes 256 validation and 256 test loader iterations;
- all nested prefix hashes are exact byte prefixes of the full target-edge
  sequence by construction;
- corresponding target-edge hashes match across model seeds;
- no runtime traceback or non-finite metric is present.

Failure of an integrity gate invalidates the affected task and must not be
replaced silently.

## Scope Boundary

This experiment can support:

- a quantitative evaluation-budget recommendation for FraudGT-style AML
  dynamic sampling;
- a paired analysis of finite evaluation budget and threshold instability;
- a reproducible diagnostic that other dynamic graph studies can apply.

It cannot by itself support:

- generalization to real transaction graphs;
- a universal optimal number of batches;
- a new fraud detector;
- a distribution-free confidence interval.


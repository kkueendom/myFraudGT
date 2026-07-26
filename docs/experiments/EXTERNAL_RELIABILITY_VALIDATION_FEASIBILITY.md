# External Reliability Validation Feasibility

## Material Passport

- Origin skill: academic-research-suite / experiment-agent plan mode
- Date: 2026-07-26
- Paper line: temporal evidence reliability under dynamic fraud-graph sampling
- Status: feasibility review, not an experiment preregistration
- Current external data availability on the 2080 Ti server: none

## Decision Question

Which independent real transaction graph can test whether the AML reliability
findings extend beyond one simulator family without forcing FraudGT's edge
classification interface onto an incompatible task?

The external experiment should validate the evaluation protocol, not claim a
new FraudGT predictive result. A node-classification dataset is acceptable if
the target hashes, model seeds, sampling streams and events are adapted
explicitly and the task difference is reported.

## Candidate Comparison

| Criterion | DGraph-Fin | Elliptic |
|---|---|---|
| Primary source | [Official dataset site](https://dgraph.xinye.com/dataset), [paper](https://arxiv.org/abs/2207.03579) | [Original paper](https://arxiv.org/abs/1908.02591), [Kaggle dataset page](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set/data) |
| Domain | Real-world dynamic financial graph | Bitcoin transaction graph |
| Published scale | About 3M nodes and 4M edges | About 200K transaction nodes and 234K directed payment flows |
| Label/task | Node classification | Node classification |
| Temporal information | Time-stamped interactions | 49 ordered time steps |
| Match to FraudGT edge task | Indirect | Indirect |
| Expected engineering cost | High | Moderate |
| Expected 2080 Ti memory risk | Moderate with neighbor sampling | Low to moderate |
| Main strength | Large real financial graph and rare-positive regime | Manageable size and established temporal split |
| Main limitation | Data access and preprocessing must be confirmed | Feature provenance/opacity and cryptocurrency domain shift |

Dataset sizes above are paper-level descriptions. Current download URLs,
terms of use, checksums and actual archive sizes must be verified before a
formal preregistration. No data has been downloaded in this phase.

## Recommendation

### Primary: DGraph-Fin

DGraph-Fin is the stronger journal-facing external test because it is a large,
real dynamic financial graph with severe class imbalance. It tests whether
the observed evaluation instability is specific to the AML simulator or also
appears in a different real financial network.

Use a standard neighbor-sampled node classifier rather than modifying FraudGT
into an artificial node model. Recommended first model family:

- two-layer GraphSAGE with the official train/validation/test split;
- fixed architecture and optimizer selected from the original/public
  baseline configuration, not from test performance;
- three training seeds;
- two independent dynamic evaluation streams;
- at least eight full validation-test events per stream;
- target-node hashes aligned across model seeds;
- nested evaluation budgets recorded in the same run.

An optional second family, such as a temporal GNN already supported by the
official benchmark, should be added only after the GraphSAGE protocol passes
integrity checks. Adding a second model is more valuable than adding many
GraphSAGE hyperparameter variants.

### Fallback: Elliptic

Use Elliptic if DGraph-Fin cannot be obtained without unresolved access or
licensing conditions. Elliptic is easier to run and has explicit temporal
steps, but its Bitcoin domain and opaque engineered features weaken direct
transfer to account-level transaction fraud.

The external claim would then be limited to:

> the reliability protocol transfers to a second temporal transaction graph
> and a node-classification model family.

It would not establish general financial-fraud transfer.

## External Protocol Adaptation

The AML protocol maps to node classification as follows:

| AML benchmark item | External node benchmark item |
|---|---|
| Target edge ID | Target node ID |
| Dynamic target-edge batches | Dynamic target-node batches |
| Dynamic sampled neighborhoods | Dynamic sampled neighborhoods |
| Validation-selected threshold | Validation-selected threshold |
| Test edge F1 | Test node F1 |
| Edge SHA-256 | Node-ID SHA-256 |

Required invariants:

- preserve the official chronological/data split;
- never place future edges into a target node's sampled history;
- keep validation and test neighborhood sampling dynamic;
- do not restore sampler RNG between events;
- align target-node sequences across model seeds within an event;
- report training-seed, stream and event variance separately;
- keep Val-selected test F1 primary and raw-event maxima diagnostic only;
- preregister evaluation budgets and all adequacy thresholds before results.

## Minimum External Experiment

After data access is confirmed, the smallest defensible screen is:

- one dataset: DGraph-Fin, or Elliptic as documented fallback;
- one non-FraudGT model family;
- three trained seeds;
- two independent dynamic streams;
- eight events per stream;
- seven nested budgets: 4, 8, 16, 32, 64, 128 and the full registered budget;
- 48 fixed-checkpoint dynamic events;
- no model selection from test results.

Advancement to the paper requires all integrity checks plus at least one of:

- dynamic sampling variance is at least the model-seed variance;
- a seed ranking changes between valid streams;
- a single-event comparison changes sign;
- raw-event maximum inflation is at least 0.01 F1;
- a sub-full budget fails the preregistered convergence criterion.

A negative result is still informative if protocol integrity is intact: it
would show that the AML instability does not automatically generalize to the
external graph.

## Resource Plan After Data Access

For one model family and three training seeds:

- GPU 0-2: one training seed each;
- GPU 3-5: preprocessing validation, deterministic checkpoint inference and
  independent smoke tasks after training completes;
- GPU 6: a scientifically distinct second baseline or audit task only if it
  is preregistered; otherwise it remains free;
- after checkpoints exist, all seven GPUs process unique stream/event tasks.

This is a dependency-aware plan, not permission to duplicate training jobs.

## Current Blockers

1. Neither external dataset is present locally or on the 2080 Ti host.
2. Current download procedure, access requirements, license and checksums have
   not yet been verified.
3. The exact official preprocessing and baseline code commit has not been
   frozen.
4. The full evaluation budget must be derived from the external loader rather
   than copied mechanically from AML.

Until these items are resolved, the external experiment is not registered and
must not consume GPU resources.


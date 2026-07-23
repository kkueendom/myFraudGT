# TIER Phase 1 Evidence Qualification Execution

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: design
- Verification Status: PREREGISTERED, NOT YET EXECUTED
- Branch: `feature/tier-independent-evidence-routing`
- Sampling Protocol: `dynamic_random`

## Purpose

Phase 1 asks whether raw transaction-centered context contains predictive
information that is independent of FraudGT hidden states. It does not train
CrossFusion or ErrorRouter and cannot establish the final paper claim by
itself.

The first screen compares two deterministic context-selection rules while
holding the evidence encoder fixed:

- `recent`: keep the most recent admissible incident transactions;
- `role_motif`: query a three-times larger recent pool, then prioritize cycle,
  relay, reciprocal and reverse-role evidence before recency.

The better qualified rule is fixed before evidence-family ablations. It cannot
be switched by dataset in the final method.

## Phase 1a Tasks

| Task | Dataset | Seed | Family | Selection |
|---|---|---:|---|---|
| 0 | Small-LI | 42 | all | recent |
| 1 | Small-LI | 42 | all | role_motif |
| 2 | Large-LI | 44 | all | recent |
| 3 | Large-LI | 44 | all | role_motif |

Each formal task uses at most 80 epochs, evaluates every two epochs, and may
stop after epoch 20 when validation F1 has not materially improved for 12
evaluation events. Before formal training, the same code must pass a two-epoch
smoke test with capped iterations.

## Dynamic Sampling Contract

All train, validation and test loaders are the original
`LinkNeighborLoader(..., shuffle=True)` path. The task keeps the original
dataset batch size and `val.iter_per_epoch=256`.

The run must reject:

- `val.fixed_target_panel=True`;
- fixed validation or test target panels;
- a dedicated evaluation generator;
- sampler RNG capture or restoration;
- a dirty Git worktree;
- a config batch size that differs from the registered original value.

Normal, shuffled and off evidence predictions are computed before leaving each
test batch. Shuffling permutes evidence across targets while preserving target
transactions and labels.

## Metrics

Every evaluation event records:

- evidence-only F1 and AUPRC;
- normal, shuffled and off evidence;
- sampled-instance and unique-edge metrics;
- evidence coverage and token-count distributions by class;
- motif and endpoint-role activation rates.

The epoch with the highest validation F1 supplies Val-selected Test F1. Raw-best
is the highest normal Test F1 over evaluation events. The validation-selected
threshold is applied to all test conditions at that event.

The historical initial A2 table is the only baseline for headline deltas.
Val-selected compares with Val-selected, and Raw-best compares with Raw-best.
The frozen epoch-499 A2 checkpoint is used only for same-batch complementarity
diagnostics; its freshly sampled F1 must not replace the historical baseline.

## Preregistered Qualification Gate

An evidence selection/family qualifies only when all conditions hold:

- normal minus shuffled Test F1 is at least `0.010`;
- normal minus off Test F1 is at least `0.005`;
- it corrects at least 10% of sampled frozen-A2 errors;
- `corrected / broken >= 1.5`;
- changed predictions are at least
  `max(50, 0.10 * sampled A2 error count)`;
- coverage is not confined to a tiny positive subset.

The two scale conditions are evaluated jointly: at least one evidence family
must qualify on Small-LI and at least one on Large-LI before CrossFusion and
ErrorRouter are implemented.

## Required Manifest

Every task records dataset, model, variant, seed, Git commit, config,
checkpoint, Val-selected Test F1, Raw-best Test F1, both same-metric deltas,
`sampling_protocol=dynamic_random`, loader audit, selected trajectory event,
same-batch A2 correction diagnostics and the qualification decision.

An absolute historical-A2 delta below `0.005` is labelled as potentially
within dynamic-sampling variation. This label does not override the
counterfactual qualification gate.

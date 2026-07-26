# CDVT Phase 0 and Phase 1 Preregistration

## Objective

Test whether an explicit causal transaction-event graph provides information
that is complementary to FraudGT's account-graph representation. This is a
method screen, not a hyperparameter search.

## Frozen protocol

- Sampling protocol: `dynamic_random` for train, validation, and test.
- `LinkNeighborLoader` keeps `shuffle=True`.
- `val.fixed_target_panel=False`.
- No fixed validation/test targets, evaluation generator, or sampler RNG reset.
- Primary result: test F1 at the epoch selected by validation F1.
- Secondary result: raw-best test F1, reported separately.
- Screening seeds: Small-LI seed 42 and Large-LI seed 44.
- Every run records dataset, variant, seed, commit, config, checkpoint,
  selected epoch, both F1 metrics, runtime, and `sampling_protocol`.

## Model contract

### Account view

The account view is the original end-to-end FraudGT encoder:

```text
h_account = FraudGTEncoder(G_account)
```

### Causal event view

Each transaction is an event node. A directed transition `e_i -> e_j` exists
only when both events share an account and:

1. `timestamp_i < timestamp_j`; or
2. timestamps are equal and `global_edge_id_i < global_edge_id_j`.

For each account-role stream, only the latest `K` admissible predecessors are
connected. Transition features include log time gap, log amount ratio, amount
change, shared-source/shared-destination indicators, in-to-out/out-to-in role
changes, currency change, payment-format change, and account-role change.

A relation-aware temporal Transformer updates event representations on this
explicit graph. The target event can receive messages only from admissible
predecessors. Labels are never event features.

### Representation fusion

```text
h_fused = LayerNorm(
    h_account + CrossAttention(h_account, H_event_context)
)
logit = MLP(h_fused)
```

The model is trained end to end. It has no scalar logit residual, decoder gate,
prototype, support coefficient, OOF router, or rule score.

### Sampling consistency

Two independently sampled account neighborhoods for the same target are used:

```text
L_cons = JS(p(S1), p(S2))
L = L_fraud + lambda_cons * L_cons
```

This is the only auxiliary objective.

## Phase 0 qualification

Before predictive experiments, all of the following must pass:

- causal ordering and equal-time edge-ID tie tests;
- latest-`K` predecessor tests per shared account;
- relation and transition-feature tests;
- target edge-ID alignment tests;
- future-event and label-leakage tests;
- normal/shuffled/off intervention tests;
- dynamic-random protocol audit;
- tiny-batch overfit and one-GPU smoke tests.

## Phase 1 matrix

| Dataset | Seed | Variant |
|---|---:|---|
| AML Small-LI | 42 | FraudGT account-only |
| AML Small-LI | 42 | event-only |
| AML Small-LI | 42 | dual-view, no consistency |
| AML Small-LI | 42 | full CDVT |
| AML Large-LI | 44 | FraudGT account-only |
| AML Large-LI | 44 | event-only |
| AML Large-LI | 44 | dual-view, no consistency |
| AML Large-LI | 44 | full CDVT |

The full model is additionally evaluated with normal, shuffled, and off event
graphs. Distinct scientific variants may run concurrently; duplicate runs may
not be launched merely to occupy GPUs.

The original dynamic-random A2 reference is 0.46247 on Small-LI and 0.30108 on
Large-LI for val-selected test F1. A matched account-only run is retained as an
implementation control, while advancement is judged against the preregistered
A2 reference.

## Advancement and stop rules

Advance only when:

- full CDVT beats A2 on both datasets;
- at least one gain is at least 0.01 F1;
- normal event context clearly beats shuffled and off context;
- full CDVT beats account-only;
- the gain is not caused by one anomalous epoch;
- event coverage and fusion-gain diagnostics confirm that the event view is
  actually used.

Stop this architecture if normal versus shuffled differs by less than 0.01,
fusion contribution is near zero, a key component works on only one scale, or
all gains are below 0.005 without cross-seed evidence. Do not respond by adding
decoder gates, prototypes, routers, or extra auxiliary losses.

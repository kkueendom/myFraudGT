# Multi-CDVT Backbone Screen

## Purpose

This screen tests whether the frozen causal event graph and representation-level
cross-attention improve the strongest public FraudGT configuration. It is a
separate branch from the PE-FraudGT experiments and does not replace the
existing CDVT evidence package.

## Component Audit

| Variant | Account view | Reverse message passing | Ports | Ego ID | Event view | Fusion |
|---|---|---:|---:|---:|---|---|
| PE-FraudGT | SparseNodeGT | No | Yes | Yes | None | Original FraudGT head |
| Multi-FraudGT | SparseNodeGT | Yes | Yes | Yes | None | Original FraudGT head |
| PE-CDVT | PE-FraudGT account view | No | Yes | Yes | Causal relation-aware event graph | Cross-attention |
| Multi-CDVT | Multi-FraudGT account view | Yes | Yes | Yes | Same frozen CDVT event graph | Same frozen cross-attention |

The public Multi configuration differs from the PE configuration by
`dataset.reverse_mp=True`; the published Multi configuration retains Ports and
Ego ID. The screen materializer rejects a Multi configuration if any of these
three components is disabled.

## Fixed Settings

- `history_k=4`
- Same event hops, maximum event count, hidden size, layers, heads, dropout,
  loss, optimizer, epoch budget and early-stopping settings as the frozen CDVT
- `lambda_cons=0`
- `dynamic_random` sampling for train, validation and test
- `LinkNeighborLoader` with `shuffle=True`
- `val.fixed_target_panel=False`
- No independent evaluation generator and no sampler RNG restoration
- Primary metric: Val-selected Test F1
- Raw-best Test F1: supplementary only

## Three-Scale Screen

| Dataset | Seed | Matched control | Candidate |
|---|---:|---|---|
| Small-LI | 42 | Multi-FraudGT | Multi-CDVT |
| Medium-LI | 42 | Multi-FraudGT | Multi-CDVT |
| Large-LI | 42 | Multi-FraudGT | Multi-CDVT |

The paired difference is:

```text
delta_multi = F1(Multi-CDVT) - F1(Multi-FraudGT)
```

The screen advances only when all conditions hold:

1. At least two of three paired deltas are positive.
2. Mean Val-selected delta is greater than `+0.005`.
3. At least one delta is at least `+0.010`.
4. No delta is less than `-0.020`.
5. All manifests pass the configuration, reverse-relation and dynamic-random
   audits.

An absolute delta below `0.005` is labeled as possible sampling variation and
is not described as a stable gain.

## Execution Control

The post-followup queue waits for the existing follow-up queue's successful
`queue_complete.json`, runs strict Phase 3, ablation and runtime audits, and
then owns one allocator for three additive controls plus six Multi screen
tasks. A failed scientific Gate is a valid result and does not trigger another
round of architecture search.

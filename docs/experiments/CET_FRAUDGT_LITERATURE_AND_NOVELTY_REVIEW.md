# CET-FraudGT Literature and Novelty Boundary

## Material Passport

- Origin Skill: academic-research-suite / deep-research
- Mode: focused literature and innovation-boundary review
- Review date: 2026-07-25
- Research question: can causal transaction history learn a representation
  complementary to a frozen FraudGT representation?
- Sampling protocol: `dynamic_random`
- Evidence status: scoped review, not a systematic-review claim

## Prior Evidence That Changes the Research Question

TIER Phase 2b used train-only, target-edge cross-fitting on Small-LI and
Large-LI. Its directional utility policy failed both scales:

| Dataset | Locked OOF changed | Corrected/broken | Paired F1 delta |
|---|---:|---:|---:|
| Small-LI | 9 | 3/6 | -0.01049 |
| Large-LI | 2 | 2/0 | +0.00189 |

The result rejects the current formulation of raw evidence as a standalone
classifier followed by a prediction-level override. It does not establish that
causal history is useless. The remaining question is whether a different
history representation can alter the FraudGT representation before
classification and remain counterfactually necessary.

## Scoped Search

The preceding TIER review title/year-checked 15 publication DOIs with Crossref
and five preprints with the official arXiv API. The current update searched
OpenAlex for temporal graph learning, transaction-subgraph fraud detection,
error-aware expert routing and OOF prediction, and counterfactual multi-view
graph fusion. The search found adjacent methods, but no exact match to the
complete CET training and evaluation relationship. This is not evidence of
absolute priority.

## Literature Matrix

| Area | Representative work | What already exists | Boundary for CET |
|---|---|---|---|
| Fraud graph transformer | [FraudGT, 2024](https://doi.org/10.1145/3677052.3698648) | Directed multigraph encoding, edge-aware local attention, ports and ego IDs | Keep as the base encoder; do not claim graph-transformer novelty |
| Directed multigraph expressivity | [Provably Powerful Directed Multigraph GNN, 2024](https://doi.org/10.1609/aaai.v38i10.29069) | Port numbering, reverse message passing and ego identifiers | These are inherited controls, not CET contributions |
| Transaction subgraphs | [Graph Feature Preprocessor, 2024](https://doi.org/10.1145/3677052.3698674); [The Shape of Money Laundering, 2024](https://doi.org/10.48550/arxiv.2404.19109) | Explicit fan-in/out, cycles, scatter-gather and subgraph-level AML analysis | CET must learn event-level history representations rather than append motif scalars |
| Subgraph representation | [Bitcoin Subgraph Contrastive Learning, 2024](https://doi.org/10.3390/e26030211) | Supervised contrastive subgraph representations | Contrastive or counterfactual training alone is not novel |
| Temporal transaction modeling | [BERT4ETH, 2023](https://doi.org/10.1145/3543507.3583345); [STA-GT, 2024](https://doi.org/10.1109/TII.2024.3423447) | Transaction sequences and temporal graph transformers | CET must show complementarity to an existing base representation, not merely temporal encoding |
| Multi-view fraud graphs | [Elliptic++, 2023](https://doi.org/10.1145/3580305.3599803); [LineMVGNN, 2025](https://doi.org/10.3390/ai6040069) | Multiple graph views and role-asymmetric transaction propagation | Incoming/outgoing streams are an implementation choice, not a standalone contribution |
| Graph counterfactual learning | [CaT-GNN, 2024](https://doi.org/10.48550/arxiv.2402.14708) | Intervention and invariant/environment feature separation | CET uses normal/shuffled/off history as a mechanism constraint, not a causal-effect identification claim |
| Graph mixture of experts | [Graph-MoE, 2025](https://doi.org/10.1609/aaai.v39i16.33921) | Learned expert routing over graph/time representations | CET must not claim novelty for gates, experts or routing |
| Failure prediction and deferral | [ConfidNet, 2019](https://doi.org/10.48550/arxiv.1910.04851); [SelectiveNet, 2019](https://doi.org/10.48550/arxiv.1901.09192); [Consistent Estimators for Learning to Defer, 2020](https://doi.org/10.48550/arxiv.2006.01862) | Confidence estimation, selective prediction and deferral | OOF base-error emphasis is useful supervision, but error prediction itself is established |

## Innovation Boundary

No individual CET component is new. The defensible contribution is the joint
methodological relationship:

1. build an independent representation from target-admissible raw transaction
   history, separated into source and destination event streams;
2. use train-only OOF FraudGT errors to emphasize complementary cases without
   using validation or test labels;
3. fuse history and FraudGT edge representations before the classifier;
4. train and evaluate normal, shuffled and off history paths so that the
   claimed gain must depend on aligned history;
5. report representation norms, changed/corrected/broken predictions and
   dynamic-sampling uncertainty.

The paper may claim this combination as the proposed method. It must not claim
the first temporal GNN, first fraud subgraph encoder, first cross-attention
fusion, first MoE, first OOF learner or first counterfactual graph model.

## Strongest Counter-Argument

A reviewer can reasonably describe CET as a combination of a temporal set
encoder, cross-attention and auxiliary losses. That criticism is fatal unless
the experiments show all of the following on both Small-LI and Large-LI:

- the full model beats the initial A2 by more than likely sampling noise;
- normal history beats shuffled history by at least 0.01 F1;
- fusion changes at least 50 predictions and corrects more than it breaks;
- removing OOF complementarity or counterfactual constraints removes a
  repeatable part of the gain;
- the representation-level fusion gain is nonzero and input-dependent.

Therefore novelty remains conditional on mechanism evidence and cross-scale
results. Architecture naming alone cannot support the paper.

## Review Verdict

`PROCEED_WITH_BOUNDED_PHASE_B`.

CET is sufficiently differentiated for a two-scale feasibility experiment,
but not yet sufficient for a journal claim. A failure on either scale, a
normal-shuffled gap below 0.01, or near-zero fusion contribution stops this
specific architecture before six-dataset expansion.

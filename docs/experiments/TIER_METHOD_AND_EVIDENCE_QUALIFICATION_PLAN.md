# TIER-FraudGT Method and Evidence Qualification Plan

## Material Passport

- Stage: research design approved for bounded implementation
- Branch: `feature/tier-independent-evidence-routing`
- Base: dynamic-random A2 protocol commit `f6965a1`
- Baseline: initial A2 in `run/dynamic_random_a2_baseline.json`
- Sampling protocol: `dynamic_random`
- Primary metric: Val-selected Test F1
- Scope of the next implementation: evidence data path and evidence-only tests
- Explicitly out of scope: full CrossFusion and ErrorRouter training

## Research question

Can transaction-centered higher-order evidence, learned directly from raw graph
events rather than from FraudGT's hidden representation, identify and
selectively correct A2 errors through error-aware representation fusion?

The paper must establish the following mechanism chain:

1. Raw evidence has independent predictive signal.
2. It corrects a nontrivial subset of A2 errors.
3. Shuffling the evidence destroys part of that signal.
4. A router can identify samples where A2 is wrong and evidence helps.
5. Final gains correspond to actual, beneficial interventions.

## Why the previous decoder line stops

COSTAR's formal Val-selected results were:

| Dataset | COSTAR | Initial A2 | Delta |
|---|---:|---:|---:|
| Small-LI | 0.45833 | 0.46247 | -0.00414 |
| Large-LI | 0.34965 | 0.30108 | +0.04857 |

Mechanism diagnostics do not attribute the Large-LI gain to COSTAR:

- Small-LI changed 4 of 519,704 sampled test predictions.
- Large-LI changed none of 72,472 sampled test predictions.
- Large-LI normal and shuffled evidence produced the same F1.
- Adapter gradients existed, but the correction was orders of magnitude below
  the A2 margin.
- The prototype residual was close to a sample-independent offset.

Therefore, TIER must not add another logit gate, residual coefficient,
prototype coefficient, or auxiliary loss after `z_base`.

## Literature and novelty boundary

The literature review covers FraudGT, directed multigraph GNNs, AML subgraph
features, transaction sequence models, temporal graph fraud detection,
multi-view graph learning, graph mixture-of-experts, selective prediction,
failure prediction, and learning to defer.

Fifteen publication DOIs were title/year checked with Crossref. Five arXiv
records were title/author/date checked with the official arXiv API. Targeted
OpenAlex searches did not find a method that directly combines independent
transaction evidence, base-error prediction, evidence utility, representation
fusion, and paired counterfactual diagnostics. This is a scoped search result,
not a claim of absolute priority.

The novelty is the unified problem and validation methodology, not any one
module:

> qualify base-error-correctable graph evidence before learning a
> utility-supervised intervention policy.

The following would be only incremental combinations:

- FraudGT plus motif scalars;
- FraudGT plus cross-attention plus a generic gate;
- temporal experts plus a standard MoE router;
- prototype plus support plus confidence residuals;
- ConfidNet or learning-to-defer applied without independent evidence
  qualification.

## Method

### Time-admissible event set

For target transaction \(e_i=(u_i,v_i,t_i)\), define:

\[
\mathcal S_i =
\left\{
e_j \in \mathcal N_2(u_i,v_i):
t_j \le t_i,\; t_i-t_j \le W,\; j \ne i
\right\}.
\]

The target edge still comes from the public FraudGT dynamic
`LinkNeighborLoader`. Evidence retrieval uses its global edge ID to query a
read-only historical event index. This is not a second target-edge sampler and
does not change the evaluation target distribution.

### Raw evidence token

\[
x_{ij} =
\left[
\operatorname{RawAttr}(e_j)
\Vert \phi(t_i-t_j)
\Vert \operatorname{Emb}(\rho_{ij})
\Vert m_{ij}
\right].
\]

`RawAttr` contains the fields actually retained by the AML pipeline:

- timestamp;
- Amount Received;
- Received Currency;
- Payment Format.

\(\rho_{ij}\) is the event's role relative to the target:

- source incoming;
- source outgoing;
- target incoming;
- target outgoing;
- reverse transfer;
- two-hop relay.

\(m_{ij}\) contains label-free relation indicators such as reciprocity,
fan-in/fan-out membership, relay membership, and short cycle completion.

Tokens must not contain labels, `h_base`, base logits, prototype-bank state, or
validation/test statistics.

### Independent evidence representation

\[
H_i^{evi} =
\operatorname{EvidenceEncoder}\left(\{x_{ij}:e_j\in\mathcal S_i\}\right),
\qquad
h_i^{evi} = \operatorname{AttnPool}(H_i^{evi},x_i).
\]

The first version is a small relation-aware set encoder with one or two layers,
32 or 64 hidden dimensions, and a fixed token cap. It is trained first with an
evidence-only classifier:

\[
z_i^{evi}=C_{evi}(h_i^{evi}).
\]

### Future full model

The full model is implemented only after evidence qualification:

\[
c_i =
\operatorname{MHA}
\left(W_qh_i^{base},W_kH_i^{evi},W_vH_i^{evi}\right),
\]

\[
h_i^{fused} =
\operatorname{LN}\left(W_bh_i^{base}+W_oc_i\right),
\qquad
z_i^{fused}=C_{fused}(h_i^{fused}\Vert h_i^{evi}),
\]

\[
q_i =
\sigma R\left[
\operatorname{sg}(h_i^{base})
\Vert \operatorname{sg}(h_i^{evi})
\Vert |z_i^{base}-z_i^{fused}|
\Vert u_i^{base}
\Vert s_i
\right],
\]

\[
z_i^{final}
=
(1-q_i)z_i^{base}+q_i z_i^{fused}.
\]

This is representation fusion followed by interpolation between two complete
predictions. It is not an additive low-magnitude decoder residual.

### Cross-fitted router target

The router cannot generate its target from its own online prediction. After
evidence qualification, training is staged:

1. Freeze A2 and train EvidenceEncoder plus CrossFusion without a router.
2. Cross-fit A2 and the fused expert inside the train split.
3. Store detached OOF base and fused logits by global edge ID.
4. Construct fixed error/utility targets.
5. Train the router against those fixed targets.
6. Never update router state from validation or test labels.

Base error and evidence advantage are:

\[
r_i^{OOF}
=
\mathbf 1[\hat y_i^{base,OOF}\ne y_i],
\]

\[
a_i
=
\sigma\left(
\frac{
\ell(z_i^{base,OOF},y_i)
-
\ell(z_i^{fused,OOF},y_i)
-
\epsilon
}{\tau}
\right),
\qquad
t_i=r_i^{OOF}a_i.
\]

## Raw evidence data contract

The current FraudGT edge encoder overwrites `edge_attr` with a hidden
representation. TIER therefore needs a parallel immutable attribute path:

- `raw_edge_attr` stores timestamp, amount, and categorical IDs.
- Existing `edge_attr` remains byte-for-byte compatible with A2 behavior.
- Amount normalization is fitted on the train interval only.
- Validation and test reuse the train normalization parameters.
- `LinkNeighborLoader` carries `raw_edge_attr` and global `e_id`.
- The backbone encoder must not write to `raw_edge_attr`.
- Historical event indices contain no labels or model outputs.
- Context edges must satisfy `context_time <= target_time`.
- The target edge is excluded from its own support.

Required unit tests:

1. future events are excluded;
2. equal-time historical events follow a deterministic tie rule;
3. target self-evidence is excluded;
4. empty support produces finite outputs and `support=0`;
5. duplicate directed edges retain distinct global edge IDs;
6. shuffling changes evidence assignment but not targets or labels;
7. off mode zeros evidence without changing the sampled target batch;
8. A2 outputs are unchanged when the evidence path is disabled.

## Evidence families

All families use the same tokenization and differ only by input masks:

| Family | Information |
|---|---|
| `structure` | reciprocity, fan-in/out, relay and cycle relation |
| `temporal` | relative time, burst, completion time, short/long windows |
| `flow_role` | endpoint role, amount ratio, flow balance and relay role |
| `all` | all qualified information |

The final evidence set must be fixed once after the pair screen. It cannot use
a dataset-specific hard switch.

## Paired counterfactual evaluation

Normal, shuffled, and off predictions must be produced in the same forward
pass over the same dynamically sampled batch:

- do not rebuild the loader;
- do not restore sampler RNG;
- do not create an evaluation generator;
- shuffle evidence across target samples while preserving targets and labels;
- compute normal/off/shuffled losses and predictions before leaving the batch.

Report both:

- sampled-instance counts, which retain repeated dynamic samples;
- unique-edge counts, which aggregate repeated predictions by global edge ID.

Required evidence diagnostics:

1. evidence-only F1 and AUPRC;
2. A2 wrong/evidence right;
3. A2 right/evidence wrong;
4. corrected/broken ratio;
5. normal minus shuffled;
6. normal minus off;
7. evidence/base error complementarity;
8. evidence coverage and token-count distributions by class;
9. motif and role activation frequency.

## Evidence qualification gate

A family advances only if all applicable conditions hold:

- same-batch normal minus shuffled Test F1 is at least `0.010`;
- normal minus off/raw-edge-only Test F1 is at least `0.005`;
- at least 10% of sampled A2 errors are corrected;
- `corrected / broken >= 1.5`;
- `corrected - broken > 0`;
- changed predictions are at least
  `max(50, 0.10 * A2_error_count)`;
- coverage is not confined to a tiny set of positives;
- at least one family qualifies on Small-LI and at least one on Large-LI.

These thresholds are preregistered before training. They must not be relaxed
after observing results.

## Screening order

### Phase 0: static and coverage audit

- Small-LI seed 42;
- Large-LI seed 44;
- no model training;
- validate data fields, temporal filtering, role coverage, token counts,
  class-conditional support, memory use, and loader invariants.

### Raw-evidence cache and batch contract

TIER is opt-in through `dataset.tier_evidence=True`. With the default `False`,
the historical A2 data and sampler path does not load a sidecar or attach TIER
target metadata.

Legacy AML `data.pt` files are not rewritten. When TIER is enabled, immutable
raw attributes are loaded from the versioned
`processed/tier_raw_edge_attr_v2.pt` sidecar. Amount is represented by a
train-prefix population z-score; timestamp and categorical transaction IDs are
kept separately.

If the formatted transaction CSV is available, v2 is built directly from it.
For historical caches where that CSV was deleted, the migration uses the fact
that each cached split column is an affine transform of the same pre-encoder
transaction field. It fits and validates the full-prefix-to-train-prefix affine
map only on their shared train edges, reconstructs train-only normalized amount,
and recovers evenly spaced categorical IDs. The sidecar records either
`formatted_csv_v2` or `legacy_cache_affine_v2` as its source. Recovery is
rejected if overlap RMSE exceeds `1e-5`, maximum error exceeds `5e-4`, or
categories are not evenly spaced. It never reads an encoder-produced
`edge_attr`.

`LinkNeighborLoader` retains sampled message-edge `raw_edge_attr` through its
global `e_id`. The TIER-only batch transform additionally maps loader
`input_id` to an explicit global `target_edge_id`. This mapping is required for
time-admissible evidence lookup, unique-edge diagnostics, and same-batch
normal/shuffled/off comparisons. `HeteroRawEdgeEncoder` may replace
`edge_attr`, but it must leave `raw_edge_attr` unchanged.

### Phase 1: evidence-only qualification

- Phase 0 passed on Small-LI and Large-LI at commit `1d37669c`; see
  `TIER_PHASE0_COVERAGE_RESULTS.md`.
- The vectorized query passed exact-equivalence and temporal-invariant tests at
  commit `0a005d87`. Large-LI query throughput improved from about 87 to
  66,493 targets/s, and end-to-end audit time improved by 10.90x; see
  `TIER_VECTORIZED_QUERY_BENCHMARK_RESULTS.md`.
- The reference Python query remains a correctness oracle.
- train only EvidenceEncoder and `C_evi`;
- 80 to 120 epochs initially;
- structure, temporal, flow/role and all masks;
- normal/shuffled/off in the same sampled batch;
- stop failed evidence families before full-model implementation.

### Phase 2: TIER pair screen

Only after Phase 1 passes:

- frozen A2;
- EvidenceEncoder, CrossFusion and fixed-target ErrorRouter;
- Small-LI seed 42 and Large-LI seed 44;
- trajectory screen before any 500-epoch confirmation.

The full candidate advances only if:

- Small-LI Val-selected Test F1 is at least `0.46747`;
- Large-LI Val-selected Test F1 is at least `0.30608`;
- pair mean delta versus initial A2 is positive;
- normal evidence is materially better than shuffled evidence;
- changed predictions meet the preregistered minimum;
- corrected predictions clearly outnumber broken predictions;
- median `abs(z_fused-z_base)/(abs(z_base)+epsilon)` is at least `0.10`;
- router error AUROC is preferably at least `0.65`;
- router calibration beats a constant-error-rate predictor.

Failure triggers mechanism diagnosis, not coefficient search.

### Phase 3: expansion

After the pair gate:

1. Medium-HI;
2. Large-HI;
3. all six datasets with three seeds;
4. formal ablation, efficiency and statistical analysis.

## Formal experiment matrix

The final study includes:

- six-dataset, three-seed mean and standard deviation;
- Val-selected Test F1 as primary and Raw-best as supplementary;
- same-metric deltas against initial A2;
- module, evidence-type, and fusion-location ablations;
- evidence-only, shuffled, off, and error-subset tests;
- parameter count, epoch time, inference latency and peak memory;
- router calibration, risk-coverage and failure cases;
- LI/HI and graph-scale stability;
- paired bootstrap or permutation tests for same-batch counterfactuals and
  same-seed ablations.

Because initial A2 is a specified historical point with no rerun distribution,
candidate-versus-A2 deltas are descriptive. Statistical significance is not
claimed against that single point. Inference is reserved for paired
counterfactual and ablation comparisons.

## Fixed experiment protocol

Every baseline, full model and ablation follows
`docs/experiments/DYNAMIC_RANDOM_A2_PROTOCOL.md`:

- dynamic train, validation and test sampling;
- `LinkNeighborLoader(..., shuffle=True)`;
- `val.fixed_target_panel=False`;
- no fixed target edge panel;
- no evaluation generator;
- no sampler RNG restoration;
- original batch size and `val.iter_per_epoch`;
- `sampling_protocol=dynamic_random`;
- no Fixed-panel A2 and no A2 rerun.

Every run records dataset, model/variant, seed, Git commit, config, checkpoint,
both F1 metrics, same-metric deltas, and the sampling protocol. Absolute deltas
below `0.005` are marked as potentially within dynamic-sampling variation.

## Paper claim and contributions

Provisional claim:

> TIER-FraudGT tests whether transaction-centered higher-order evidence,
> learned independently from the FraudGT backbone, can identify and
> selectively correct base-model errors through utility-supervised
> representation routing.

The statement remains a hypothesis until the complete mechanism and
performance gates pass.

Provisional contributions:

1. A base-error-correctable graph evidence problem formulation and
   counterfactual diagnostic protocol.
2. Independent transaction evidence encoding, representation fusion, and an
   OOF utility-supervised error router.
3. Multi-scale AML evaluation that connects F1 changes to evidence coverage,
   routing decisions, corrected/broken samples, calibration, and cost.

Provisional title:

> TIER-FraudGT: Error-Aware Representation Fusion of Independent Transaction
> Evidence for Financial Fraud Detection

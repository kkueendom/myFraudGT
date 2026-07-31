# CDVT Code-to-Method Alignment Audit

## Scope

- Audited branch: `feature/cdvt-phase3-experiments`
- Architecture freeze: `9038f85`
- Follow-up execution source: `34456ab`
- Additive-control source: `dac3c7b`
- Manuscript audited: `docs/paper/CDVT_Manuscript_Draft.md`
- Status: method claims align with the frozen implementation; final numerical
  claims remain conditional on completed manifests.

## Claim-to-Code Matrix

| Manuscript claim | Authoritative implementation evidence | Assessment |
|---|---|---|
| Each target receives only transactions ordered before it | `fraudGT/cdvt/event_graph.py`: `_latest_for_account`, `_collect_nodes`, `_build_one` | Supported. Equal timestamps are ordered by global edge ID. |
| Retrieval uses recent history per encountered account | `CausalEventGraphIndex._latest_for_account` and `_collect_nodes` | Supported. The frozen settings are `K=4`, two expansion hops, and at most 48 events. |
| Event edges are directed from earlier to later transactions sharing an account | `CausalEventGraphIndex._build_one` | Supported. Selected nodes are chronological and the streaming builder only connects stored predecessors to the current event. |
| Four account-role transition types are represented | `OUT_OUT`, `IN_IN`, `OUT_IN`, `IN_OUT` and `_transition_features` | Supported through both relation embeddings and explicit transition indicators. |
| Event inputs contain timestamp, train-standardized amount, currency, and payment format | `fraudGT/evidence/tier.py`: `build_raw_edge_attributes`; `fraudGT/datasets/aml_dataset.py`: `_build_tier_sidecar_from_csv` | Supported. Amount normalization uses training-prefix statistics. |
| Event attention is relation- and transition-aware | `fraudGT/cdvt/temporal_transformer.py`: `RelationAwareTemporalLayer` | Supported. Relation and continuous transition features enter key and value pathways. |
| FraudGT and event states are fused before classification | `fraudGT/cdvt/fusion.py`: `DualViewFusionClassifier.forward` | Supported. The account representation is the query; all local event states are keys and values; the fused state is classified afterward. |
| Both encoders train jointly | `fraudGT/network/cdvt_model.py` and `run/cdvt_phase1_screen.py` | Supported. The optimizer receives all model parameters and gradients were verified by the mixed-class GPU smoke. |
| Normal, shuffled, and off conditions preserve target alignment for paired diagnostics | `CausalEventGraphBatch.shuffled`, `.off`, and `CDVTModel.forward_counterfactuals` | Supported. The target IDs and account representation are retained while event context is permuted or removed. |
| The evaluation follows dynamic-random FraudGT sampling | CDVT YAML configs, `run/cdvt_phase1_screen.py:audit_loaders`, and result manifests | Supported. Fixed panels, dedicated generators, and sampler-state restoration are absent or explicitly rejected. |
| Event retrieval is cached without changing graph semantics | `CausalEventGraphIndex._cached_build_one` | Supported. The LRU key includes target ID, K, hops, event cap, and time window. |

## Boundary Conditions

1. **Meaning of causal.** CDVT guarantees temporal precedence and excludes
   post-target events. It does not identify statistical causal effects. The
   manuscript states this explicitly.
2. **Amount feature.** Transition ratios use the absolute train-standardized
   amount plus one, not the ratio of original currency amounts. The equations
   in Section 3.4 match this implementation.
3. **Target-conditioned retrieval.** Every expansion frontier retrieves events
   preceding the target, rather than preceding the frontier event. All selected
   events remain target-admissible, and transition edges are subsequently
   constructed in chronological order.
4. **Categorical cardinality.** Embedding table sizes are determined from the
   immutable full-prefix category IDs. No labels or future transaction values
   are passed through the event graph, but this metadata dependency should be
   retained in reproducibility documentation.
5. **Intervention interpretation.** Normal-versus-shuffled/off differences show
   reliance on aligned event context. They are mechanism diagnostics, not
   estimates of intervention effects in the causal-inference sense.
6. **Cross-attention attribution.** The implementation contains an
   `additive_view` control. Its three representative-dataset queue is staged at
   commit `dac3c7b`; no contribution claim will be made until those manifests
   are complete.

## Verification Evidence

- Event-graph unit tests cover future exclusion, equal-time ordering, latest-K
  retrieval, explicit relation features, off-context semantics, label
  exclusion, and bounded caching.
- Model tests cover pre-classification fusion, gradient propagation,
  event-only independence, additive fusion, complete removal of relation
  pathways, and distinct normal/shuffled/off outputs.
- The corrected remote follow-up bundle passed 43 tests before training.
- Local additive-control summary and queue tests pass; PyYAML-dependent config
  tests are skipped by the local system Python and must be rerun in the remote
  FraudGT environment before deployment.

## Open Evidence Items

- representative three-seed paired CDVT versus account-only results;
- completed account-only/event-only core ablation on Medium-LI;
- no-relation and K=2 manifests on all three representative scales;
- normal-only runtime benchmarks;
- additive-fusion manifests and cross-attention-minus-additive deltas;
- final claim-to-reference and numerical consistency audit.

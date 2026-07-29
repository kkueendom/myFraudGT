# Experiment Results Directory

## Current protocol (effective 2026-07-29)

The active paper line is **Causal Dual-View Transaction Transformer (CDVT)**:
an explicit causal transaction-event graph is encoded alongside FraudGT's
account graph and fused before the classifier.

Current scientific question:

> Does a causal transaction-event view complement FraudGT's account-graph
> representation under the original dynamic-random sampling protocol?

The model is frozen as causal event graph plus representation-level dual-view
fusion with `lambda_cons=0`. Decoder gates, scalar logit residuals, prototypes,
support coefficients, routers, and rule scores are not part of this line.

Current read order:

1. `CDVT_PHASE0_AUDIT_RESULTS.md`
   Records the 42-test remote audit, real GPU gradients, event coverage,
   intervention differences, and mixed-class fixed-batch trainability check.

2. `CDVT_BASELINE_POLICY_AMENDMENT.md`
   Defines PE-FraudGT as the primary published baseline, Multi-FraudGT as the
   strong published reference, A2 as an internal comparator, and account-only
   as the matched multi-seed control.

3. `CDVT_FINAL_MODEL_FREEZE.md`
   Records the validation-only selection of dual-view without consistency and
   the frozen Phase 2 architecture. Its historical A2 gate is superseded by
   the baseline amendment.

4. `CDVT_PHASE0_PHASE1_PREREGISTRATION.md`
   Defines causal event construction, fusion, sampling protocol, Phase 0
   checks, Phase 1 variants, and the pre-execution seed amendment.

5. `CDVT_PHASE2_EXECUTION_PLAN.md`
   Defines the six-dataset seed-42 evaluation and immutable result manifests.

6. `CDVT_PHASE3_ABLATION_PLAN.md`
   Defines paired CDVT/account-only seeds, module ablations, relation and
   history-size controls, and runtime benchmarks.

Current reporting rules:

- Primary: Val-selected Test F1 against architecture-matched PE-FraudGT.
- Strong references: Multi-FraudGT and the unpublished initial A2 result.
- Representative multi-seed inference: same-seed paired CDVT minus account-only.
- Supplementary Raw-best may only be compared with A2 Raw-best.
- Never compare Val-selected and Raw-best across columns.
- Mark `abs(delta_f1) < 0.005` as potentially within dynamic-sampling noise.
- Record `sampling_protocol=dynamic_random` and full provenance in every
  experiment manifest.

Earlier decoder and dynamic-reliability experiments remain historical evidence.
They are not the current paper line and must not be mixed into CDVT main tables.

The files dated `20260626` below are a historical evidence bundle from
`snapshot/fraudgt-results-20260626`. They remain useful for diagnosis, but their
old reporting convention is not the baseline or formal protocol for new runs.

Historical 2026-06-26 read order:

1. `fraudgt_current_results_for_model_review_20260626.md`  
   Main narrative summary: method, protocol, results, ablation interpretation, and recommended next direction.

2. `original_raw_baseline_audit_20260626.tsv`  
   Raw audit of the re-run original FraudGT-style baseline over 6 datasets x 3 seeds.

3. `mainline_matched3_rawbest_audit_20260626.tsv`  
   Raw audit of the current scale-adaptive mainline over 6 datasets x 3 seeds.

4. `best_seed_clean_ablation_audit_20260626.tsv`  
   Raw audit of the best-seed clean ablation package. This is incomplete in the snapshot: 26/30 jobs are complete.

Historical metric note:

```text
raw_best_test_f1 = max_epoch test_f1(epoch)
```

This was the metric used for the main comparison in the archived summary. It
must not replace the current primary Val-selected comparison.

Important caveats:

- The current formal mainline wins 5/6 datasets, not 6/6.
- Large-LI is below the original baseline in 3-seed mean.
- The ablation variants are evidence routes, not a perfect one-module-at-a-time monotonic chain.
- The strongest next direction appears to be safer routing of class/prototype, class-split, and subgraph evidence rather than simply adding more decoder modules.

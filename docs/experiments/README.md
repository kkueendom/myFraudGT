# Experiment Results Directory

## Current protocol (effective 2026-07-23)

All new screening, full-model, and ablation experiments must follow
`DYNAMIC_RANDOM_A2_PROTOCOL.md`. The only baseline for new experiments is the
initial A2 table in `run/dynamic_random_a2_baseline.json`.

Current TIER research line:

1. `TIER_METHOD_AND_EVIDENCE_QUALIFICATION_PLAN.md`
2. `TIER_PHASE0_COVERAGE_AUDIT_PLAN.md`
3. `TIER_PHASE0_COVERAGE_RESULTS.md`
4. `tier_phase0_1d37669c/` for the raw Phase 0 JSONL and manifest

- Primary: candidate Val-selected Test F1 versus initial A2 Val-selected Test F1.
- Supplementary: candidate Raw-best Test F1 versus initial A2 Raw-best Test F1.
- Never compare across the two metric columns.
- Never include Fixed-panel A2 in a new main result table.
- Mark `abs(delta_f1) < 0.005` as potentially within dynamic-sampling noise.
- Record `sampling_protocol=dynamic_random` in every experiment manifest.

The files dated `20260626` below are a historical evidence bundle from
`snapshot/fraudgt-results-20260626`. They remain useful for diagnosis, but their
old reporting convention is not the baseline or formal protocol for new runs.

Read order for another model/reviewer:

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

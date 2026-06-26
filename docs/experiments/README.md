# Experiment Results Directory

This directory is the main evidence bundle for the `snapshot/fraudgt-results-20260626` branch.

Read order for another model/reviewer:

1. `fraudgt_current_results_for_model_review_20260626.md`  
   Main narrative summary: method, protocol, results, ablation interpretation, and recommended next direction.

2. `original_raw_baseline_audit_20260626.tsv`  
   Raw audit of the re-run original FraudGT-style baseline over 6 datasets x 3 seeds.

3. `mainline_matched3_rawbest_audit_20260626.tsv`  
   Raw audit of the current scale-adaptive mainline over 6 datasets x 3 seeds.

4. `best_seed_clean_ablation_audit_20260626.tsv`  
   Raw audit of the best-seed clean ablation package. This is incomplete in the snapshot: 26/30 jobs are complete.

Metric note:

```text
raw_best_test_f1 = max_epoch test_f1(epoch)
```

This is the metric used for the main comparison in the summary. Do not confuse it with `test_at_val`, which is also printed by some audit scripts.

Important caveats:

- The current formal mainline wins 5/6 datasets, not 6/6.
- Large-LI is below the original baseline in 3-seed mean.
- The ablation variants are evidence routes, not a perfect one-module-at-a-time monotonic chain.
- The strongest next direction appears to be safer routing of class/prototype, class-split, and subgraph evidence rather than simply adding more decoder modules.

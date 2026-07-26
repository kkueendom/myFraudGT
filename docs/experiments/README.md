# Experiment Results Directory

## Current protocol (effective 2026-07-26)

All new screening, full-model, and ablation experiments must follow
`DYNAMIC_RANDOM_A2_PROTOCOL.md`. The only baseline for new experiments is the
initial A2 table in `run/dynamic_random_a2_baseline.json`.

Current paper line:

> Temporal evidence reliability under dynamic fraud-graph sampling.

The predictive decoder/encoder search and four registered reliability methods
have stopped after failed advancement gates. The active deliverable is a
negative reliability benchmark and reproducible evaluation protocol.

Current active experiment:

- `DYNAMIC_EVALUATION_BUDGET_CONVERGENCE_PLAN.md`
  preregisters a paired 6-dataset x 3-model-seed x 2-stream evaluation-budget
  audit. Seven nested budgets are extracted from each full dynamic trajectory,
  so the experiment can recommend how many sampled batches are required
  without confounding budget and target-edge draws.
- `EXTERNAL_RELIABILITY_VALIDATION_FEASIBILITY.md`
  compares DGraph-Fin and Elliptic for the mandatory external-validation
  stage. It is a feasibility review, not permission to start an external run.

Current read order for another model or reviewer:

1. `TEMPORAL_EVIDENCE_RELIABILITY_BENCHMARK_RESULTS.md`
   Six datasets, three model seeds, two dynamic streams and four events:
   nested variance, target-edge hash alignment, ranking reversals and raw-max
   inflation.

2. `METHODS_JOURNAL_MAINLINE_COMPLETION_AUDIT.md`
   Requirement-by-requirement audit of TIER, CET, CPSE, GTF1C, TREFIC,
   DGR-F1 and GT-psF1.

3. `MULTI_MODEL_DYNAMIC_RELIABILITY_RESULTS.md`
   Cross-family normal/shuffled/off sensitivity-versus-utility benchmark.

4. `A2_DYNAMIC_SAMPLING_STABILITY_RESULTS.md`
   Fixed-checkpoint dynamic evaluation over all six datasets.

5. `GT_PSF1_PHASE0_DEVELOPMENT_RESULTS.md`
   Final independent statistical-method attempt and its preregistered stop
   decision.

6. `TEMPORAL_EVIDENCE_RELIABILITY_BENCHMARK_PLAN.md` and
   `DYNAMIC_EVIDENCE_RELIABILITY_LITERATURE_AND_NOVELTY_REVIEW.md`
   Protocol, research question and conservative novelty boundary.

7. `DYNAMIC_EVALUATION_BUDGET_CONVERGENCE_PLAN.md`
   Preregistered actionability experiment for the current benchmark paper.

8. `EXTERNAL_RELIABILITY_VALIDATION_FEASIBILITY.md`
   External dataset/task/resource comparison and remaining access blockers.

Publication figures are generated deterministically by
`run/plot_temporal_reliability_results.py`. The script reads the authoritative
nested-result JSON plus all 36 manifests and records SHA-256 hashes for every
source artifact used in the plots.

- Primary: candidate Val-selected Test F1 versus initial A2 Val-selected Test F1.
- Supplementary: candidate Raw-best Test F1 versus initial A2 Raw-best Test F1.
- Never compare across the two metric columns.
- Never include Fixed-panel A2 in a new main result table.
- Mark `abs(delta_f1) < 0.005` as potentially within dynamic-sampling noise.
- Record `sampling_protocol=dynamic_random` in every experiment manifest.

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

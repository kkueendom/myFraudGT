# FraudGT Current Experiment Snapshot for Model Review

Updated: 2026-06-26 12:54 CST  
Repository branch intended for sharing: `snapshot/fraudgt-results-20260626`  
Remote experiment host: `yyk@10.168.1.102`  
Remote working repo: `/e/yyk/FraudGT_multi6_wt_rawpeak_supportmixconsis_remaining`  
Result root: `/e/yyk/FraudGT_multi6`

This document summarizes the current FraudGT extension work so another model or reviewer can quickly understand the method, code state, experiment protocol, results, limitations, and likely next direction.

---

## 1. Executive Summary

The current work extends FraudGT mainly at the decoder stage. The encoder/backbone remains FraudGT-style; the decoder is augmented with controlled evidence branches, including support-conditioned evidence, class prototype evidence, class-split evidence, high-order subgraph routing, uncertainty gates, and bounded residual fusion.

The formal current mainline is **scale-adaptive**:

| Dataset scale | Decoder route used in formal 3-seed mainline |
|---|---|
| Small / Medium | Full dual-uncertainty high-order subgraph gate route |
| Large | ClassMix / prototype / bounded-residual fallback route |

Important: the formal 6-dataset mainline is **not** a single identical decoder string across all datasets. It is a deterministic scale-adaptive rule.

Formal raw-best test F1 comparison over 3 seeds is complete for baseline and mainline:

| Item | Result |
|---|---:|
| Original baseline complete | 18/18 |
| Mainline complete | 18/18 |
| Mainline wins | 5/6 datasets |
| Baseline average raw-best F1 | 0.61910 |
| Mainline average raw-best F1 | 0.63640 |
| Average gain | +0.01730 |
| Negative case | Large-LI, -0.00851 |

Current clean best-seed ablation is still running:

| Item | Status |
|---|---:|
| Total ablation jobs | 30 |
| Complete | 26/30 |
| Running | Large-LI / supportmix_consis_proto / seed44 |
| Missing | Large-LI classmix_proto_bound, class_split, full_dual_uncert_gate |

The ablation evidence suggests that the current full route is useful but not uniformly best. A stronger next mainline should probably be a cleaner **ClassMix/Core Evidence + gated ClassSplit + Safe Subgraph Router** rather than the current full decoder being used as a monolithic final architecture.

---

## 2. Metric and Protocol

Main reporting metric in these experiments:

```text
raw_best_test_f1 = max_epoch test_f1(epoch)
```

This is different from best-validation-selected test F1. Many audit scripts also report `test_at_val`, but the current user-requested comparison standard is raw-best test F1.

Datasets:

| Dataset |
|---|
| Small-HI |
| Small-LI |
| Medium-HI |
| Medium-LI |
| Large-HI |
| Large-LI |

Formal seeds:

```text
42, 43, 44
```

Best-seed ablation uses the best mainline seed per dataset:

| Dataset | Best seed | Mainline raw-best F1 |
|---|---:|---:|
| Small-HI | 42 | 0.80227 |
| Small-LI | 42 | 0.52252 |
| Medium-HI | 42 | 0.80639 |
| Medium-LI | 44 | 0.55281 |
| Large-HI | 43 | 0.78072 |
| Large-LI | 44 | 0.42640 |

Raw audit files committed with this snapshot:

| File | Meaning |
|---|---|
| `docs/experiments/original_raw_baseline_audit_20260626.tsv` | Original baseline 3-seed raw-best audit |
| `docs/experiments/mainline_matched3_rawbest_audit_20260626.tsv` | Current mainline 3-seed raw-best audit |
| `docs/experiments/best_seed_clean_ablation_audit_20260626.tsv` | Current best-seed clean ablation audit |

---

## 3. Method Summary

### 3.1 Original FraudGT Baseline

Original baseline used here:

```text
SparseNodeGT + ports + Ego + dot edge decoder
```

The original baseline was re-run locally instead of directly copying numbers from the paper. This keeps implementation environment, data paths, and metric protocol consistent.

### 3.2 Current Mainline: Scale-Adaptive Evidence-Gated Decoder

The current mainline keeps the FraudGT encoder and changes the final edge decoder. It uses auxiliary evidence to adjust the base edge prediction.

Main evidence/control ideas:

| Module | Role |
|---|---|
| Support-conditioned evidence | Estimate whether local evidence is reliable |
| Class prototype evidence | Compare the target transaction to normal/fraud prototypes |
| ClassMix / bounded residual | Add prototype/class-aware correction without replacing base prediction |
| Class-split evidence | Separate positive-class and negative-class evidence |
| High-order subgraph route | Use local higher-order transaction structure |
| Uncertainty gate | Let high-order evidence intervene more when the base decision is uncertain |
| Bounded residual | Keep auxiliary evidence as controlled correction rather than full replacement |
| Scale-adaptive rule | Use full route on Small/Medium, fallback on Large |

Formal mainline config families:

| Dataset scale | Structure name in audit | Example configs |
|---|---|---|
| Small/Medium | `full_dual_uncert_subgraph_gate` | `configs/AML-Small-HI/AML-Small-HI-UnifiedClassSplitSubgraphDualUncertGate180.yaml` |
| Large | `scale_fallback_classmix_proto_bound_resid` | `configs/AML-Large-HI/AML-Large-HI-ClassMixProtoBoundResid240Seed42.yaml` |

The large fallback exists because earlier diagnosis showed the full high-order subgraph route can be unstable on large graphs.

---

## 4. Formal Baseline Results

Original baseline was completed for 6 datasets x 3 seeds = 18 runs.

| Dataset | Baseline raw-best F1 mean | Std |
|---|---:|---:|
| Small-HI | 0.78572 | 0.00188 |
| Small-LI | 0.49746 | 0.00716 |
| Medium-HI | 0.76386 | 0.00508 |
| Medium-LI | 0.52291 | 0.01542 |
| Large-HI | 0.72126 | 0.01151 |
| Large-LI | 0.42336 | 0.00889 |

Seed-level baseline results:

| Dataset | Seed 42 | Seed 43 | Seed 44 |
|---|---:|---:|---:|
| Small-HI | 0.78599 | 0.78372 | 0.78744 |
| Small-LI | 0.50474 | 0.49042 | 0.49722 |
| Medium-HI | 0.76768 | 0.76580 | 0.75809 |
| Medium-LI | 0.51016 | 0.54005 | 0.51852 |
| Large-HI | 0.70978 | 0.72120 | 0.73279 |
| Large-LI | 0.43137 | 0.42493 | 0.41379 |

---

## 5. Formal Mainline Results

Mainline was completed for 6 datasets x 3 seeds = 18 runs.

| Dataset | Baseline mean +/- std | Mainline mean +/- std | Delta |
|---|---:|---:|---:|
| Small-HI | 0.78572 +/- 0.00188 | 0.80011 +/- 0.00345 | +0.01439 |
| Small-LI | 0.49746 +/- 0.00716 | 0.50219 +/- 0.02021 | +0.00473 |
| Medium-HI | 0.76386 +/- 0.00508 | 0.79855 +/- 0.00736 | +0.03469 |
| Medium-LI | 0.52291 +/- 0.01542 | 0.53470 +/- 0.01571 | +0.01179 |
| Large-HI | 0.72126 +/- 0.01151 | 0.76799 +/- 0.01355 | +0.04673 |
| Large-LI | 0.42336 +/- 0.00889 | 0.41485 +/- 0.01688 | -0.00851 |

Overall:

| Model | Average raw-best F1 |
|---|---:|
| Original baseline | 0.61910 |
| Mainline | 0.63640 |
| Delta | +0.01730 |

Seed-level mainline results:

| Dataset | Seed 42 | Seed 43 | Seed 44 |
|---|---:|---:|---:|
| Small-HI | 0.80227 | 0.79613 | 0.80193 |
| Small-LI | 0.52252 | 0.48211 | 0.50195 |
| Medium-HI | 0.80639 | 0.79178 | 0.79749 |
| Medium-LI | 0.52473 | 0.52657 | 0.55281 |
| Large-HI | 0.75375 | 0.78072 | 0.76949 |
| Large-LI | 0.39548 | 0.42268 | 0.42640 |

Claim that is supported:

```text
The proposed scale-adaptive decoder improves raw-best test F1 on 5/6 AML datasets, with an average +0.01730 gain over the re-run FraudGT-style baseline.
```

Claim that is NOT supported:

```text
The 3-seed mean improves over baseline on all six datasets.
```

Large-LI remains the formal negative case.

---

## 6. Best-Seed Diagnostic Analysis

Best-seed analysis is not the formal 3-seed mean result. It is useful only as a potential/diagnostic view.

| Dataset | Baseline 3-seed mean | Mainline best seed | Best seed | Delta vs baseline mean |
|---|---:|---:|---:|---:|
| Small-HI | 0.78572 | 0.80227 | 42 | +0.01655 |
| Small-LI | 0.49746 | 0.52252 | 42 | +0.02506 |
| Medium-HI | 0.76386 | 0.80639 | 42 | +0.04253 |
| Medium-LI | 0.52291 | 0.55281 | 44 | +0.02990 |
| Large-HI | 0.72126 | 0.78072 | 43 | +0.05946 |
| Large-LI | 0.42336 | 0.42640 | 44 | +0.00304 |

Best-seed average:

| Item | Value |
|---|---:|
| Baseline 3-seed mean average | 0.61910 |
| Mainline best-seed average | 0.64852 |
| Delta | +0.02942 |
| Win count | 6/6 |

Do not report this as formal mean performance.

---

## 7. Clean Best-Seed Ablation Results

Current ablation variants:

| Variant | Actual route meaning |
|---|---|
| `base_dot` | Original FraudGT-style dot decoder |
| `supportmix_consis_proto` | Support-conditioned evidence plus class prototype context |
| `classmix_proto_bound` | Support + classmix/prototype + bounded residual |
| `class_split` | `classmix_proto_bound` plus class-split evidence |
| `full_dual_uncert_gate` | Class-split plus high-order subgraph route and dual uncertainty gate |

These variants are partially incremental but not a perfectly clean one-module-at-a-time ablation chain. The early transition from `supportmix_consis_proto` to `classmix_proto_bound` changes the route structure, and the full route adds multiple mechanisms at once.

Current status as of 2026-06-26 12:54 CST:

| Status | Count |
|---|---:|
| Complete | 26/30 |
| Running | 1/30 |
| Missing | 3/30 |

Incomplete Large-LI entries:

| Variant | Status |
|---|---|
| `supportmix_consis_proto` | Running |
| `classmix_proto_bound` | Missing |
| `class_split` | Missing |
| `full_dual_uncert_gate` | Missing |

Completed raw-best ablation table:

| Dataset | base_dot | supportmix | classmix | class_split | full_dual |
|---|---:|---:|---:|---:|---:|
| Small-HI | 0.77465 | 0.79360 | 0.80331 | **0.80535** | 0.79289 |
| Small-LI | 0.48519 | 0.46154 | 0.48454 | 0.49474 | **0.49821** |
| Medium-HI | 0.74028 | 0.79617 | 0.79306 | 0.79124 | **0.80053** |
| Medium-LI | 0.48458 | 0.48387 | **0.55510** | 0.53140 | 0.52675 |
| Large-HI | 0.71264 | **0.76875** | 0.74128 | 0.75949 | 0.74054 |
| Large-LI | 0.35644 | running | missing | missing | missing |

Small/Medium average:

| Variant | Average raw-best F1 |
|---|---:|
| base_dot | 0.62118 |
| supportmix_consis_proto | 0.63380 |
| classmix_proto_bound | **0.65900** |
| class_split | 0.65568 |
| full_dual_uncert_gate | 0.65460 |

Key ablation interpretation:

1. `classmix_proto_bound` is the strongest Small/Medium average variant.
2. `full_dual_uncert_gate` is best on Small-LI and Medium-HI, but not uniformly best.
3. `class_split` is best on Small-HI and strong on Large-HI.
4. `supportmix_consis_proto` is best on Large-HI and probably important for large-scale stability.
5. The most complex full route can introduce noise; it should not be treated as universally superior.

---

## 8. Current Methodological Problem

The present mainline has a real positive signal but the story is not yet as clean as a strong paper would like.

Problems:

| Problem | Evidence |
|---|---|
| Mainline is not 6/6 on formal 3-seed mean | Large-LI is -0.00851 below baseline |
| Full route is not uniformly strongest | Small/Medium average best is `classmix_proto_bound`, not `full_dual_uncert_gate` |
| Complex modules have mixed contribution | Full route loses to simpler routes on Small-HI, Medium-LI, and Large-HI |
| Ablation is not perfectly monotonic | Variants are routes, not one-module increments |
| Large graphs dislike unconditional subgraph route | Large-HI best ablation is `supportmix_consis_proto`; full route is lower |

This means the current paper story should not be: "we add many decoder modules and every one improves performance." The better story is: "evidence quality and scale matter; class/prototype evidence is the stable core, and high-order subgraph evidence should be routed safely."

---

## 9. Recommended Next Mainline

Recommended direction:

```text
ClassMix/Core Evidence + Gated ClassSplit + Safe Subgraph Router
```

Proposed v2 structure:

```text
z_base
  -> classmix_proto_bound core evidence
  -> gated class_split correction
  -> optional safe subgraph correction only for uncertain/conflicting cases
  -> z_final
```

Suggested candidates:

| Candidate | Purpose |
|---|---|
| `v2_classmix_core` | Confirm that classmix/prototype bounded residual is a strong simple core |
| `v2_classmix_classsplit_gate` | Gate class-split evidence instead of always adding it |
| `v2_safe_subgraph_router` | Use subgraph evidence only when prediction is uncertain or evidence branches disagree |

Suggested acceptance criteria:

| Criterion | Target |
|---|---|
| Small/Medium average | > 0.65460 current full, ideally > 0.65900 classmix |
| 6-dataset win count | at least 5/6, ideally 6/6 |
| Large-LI | not below baseline mean 0.42336 |
| Complexity | simpler than current full route |
| Interpretability | each module has a clear reason and ablation support |

---

## 10. z_base Exploration Status

A separate z_base controlled-calibration line was explored. It should be treated as exploratory and not part of the current formal mainline.

Summary:

| Line | Best result | Interpretation |
|---|---:|---|
| z_base controlled calibration attempts | Small-HI best 0.80363 | Small gain, not enough to justify formal inclusion |
| Next5 branch conflict attempts | Small-HI best 0.80519 | Best single Small-HI signal came from support disagreement gate |
| Dualdisagree multi6 audit | pass_target 1/6 | Not strong enough as a current mainline |

Interpretation:

`z_base` calibration can slightly improve a single dataset but did not become a robust six-dataset mainline. This line should remain a supplementary exploration unless revalidated under the same multi-dataset 3-seed protocol.

---

## 11. Files and Scripts to Inspect

Core implementation:

| File | Purpose |
|---|---|
| `fraudGT/head/hetero_edge.py` | Decoder implementation and evidence routes |
| `fraudGT/graphgym/checkpoint.py` | Uses `map_location=cfg.device` for cross-GPU checkpoint resume |

Main audit / queue scripts:

| File | Purpose |
|---|---|
| `run/original_raw_baseline_audit.py` | Audit original baseline raw-best F1 |
| `run/original_raw_baseline_queue.py` | Queue original baseline runs |
| `run/mainline_matched3_rawbest_audit.py` | Audit current scale-adaptive mainline 3-seed raw-best F1 |
| `run/mainline_matched3_rawbest_queue.py` | Queue current mainline 3-seed runs |
| `run/paper_ablation_clean_audit.py` | Audit clean best-seed ablations |
| `run/paper_ablation_clean_queue.py` | Queue clean best-seed ablations |

Representative configs:

| Dataset scale | Example config |
|---|---|
| Small full route | `configs/AML-Small-HI/AML-Small-HI-UnifiedClassSplitSubgraphDualUncertGate180.yaml` |
| Medium full route | `configs/AML-Medium-HI/AML-Medium-HI-UnifiedClassSplitSubgraphDualUncertGate180.yaml` |
| Large fallback | `configs/AML-Large-HI/AML-Large-HI-ClassMixProtoBoundResid240Seed42.yaml` |

---

## 12. How Another Model Should Read This Snapshot

Best concise understanding:

1. This is a FraudGT decoder-side extension, not a new encoder.
2. The formal current method is scale-adaptive: full route for Small/Medium, fallback for Large.
3. Formal 3-seed results are positive on 5/6 datasets, with +0.01730 average raw-best F1 gain.
4. Large-LI remains the weak formal case.
5. Clean ablation suggests classmix/prototype bounded residual is the most stable core; full subgraph route is useful but noisy.
6. Next work should simplify the mainline into a safe evidence router rather than adding more modules.


# FraudGT Evidence-Decoder Experiment Snapshot

This branch is a review snapshot for the current FraudGT extension work.
It contains the original FraudGT codebase plus the current decoder-side experimental branch, reproducibility configs, audit scripts, and summarized experiment results.

Recommended branch for review:

```text
snapshot/fraudgt-results-20260626
```

Primary review document:

```text
docs/experiments/fraudgt_current_results_for_model_review_20260626.md
```

If you are another model or reviewer, start there first. It explains the method, which results are formal, which are diagnostic, and what still needs improvement.

---

## How To Read This Snapshot

### 1. Start With The Summary

Read this file first:

```text
docs/experiments/fraudgt_current_results_for_model_review_20260626.md
```

It contains:

- the current paper/mainline idea;
- the exact metric protocol;
- baseline vs mainline 3-seed results;
- best-seed diagnostic results;
- clean ablation progress;
- current problems and recommended next model direction.

Important high-level conclusion:

```text
The current method is a scale-adaptive decoder-side FraudGT extension.
It improves 3-seed raw-best F1 on 5/6 AML datasets, but Large-LI is still below baseline.
The clean ablation suggests class/prototype evidence is the stable core, while high-order subgraph evidence should be gated more safely.
```

### 2. Check The Raw Result Tables

The raw audit snapshots are under:

```text
docs/experiments/
```

| File | What it contains |
|---|---|
| `original_raw_baseline_audit_20260626.tsv` | Re-run original FraudGT-style baseline, 6 datasets x 3 seeds |
| `mainline_matched3_rawbest_audit_20260626.tsv` | Current scale-adaptive mainline, 6 datasets x 3 seeds |
| `best_seed_clean_ablation_audit_20260626.tsv` | Best-seed clean ablation status and results |

The main metric used in these summaries is:

```text
raw_best_test_f1 = max_epoch test_f1(epoch)
```

This is different from best-validation-selected test F1. Some audit files also include `test_at_val`, but the current user-requested comparison uses raw-best test F1.

### 3. Understand The Current Mainline

The current formal mainline is not one identical decoder string for all datasets. It uses a deterministic scale rule:

| Dataset scale | Decoder route |
|---|---|
| Small / Medium | full dual-uncertainty high-order subgraph gate route |
| Large | classmix/prototype bounded-residual fallback route |

Representative configs:

| Purpose | Example config |
|---|---|
| Small full route | `configs/AML-Small-HI/AML-Small-HI-UnifiedClassSplitSubgraphDualUncertGate180.yaml` |
| Small-LI full route seed package | `configs/AML-Small-LI/AML-Small-LI-UnifiedClassSplitSubgraphDualUncertGate240Seed42.yaml` |
| Medium full route | `configs/AML-Medium-HI/AML-Medium-HI-UnifiedClassSplitSubgraphDualUncertGate180.yaml` |
| Large fallback | `configs/AML-Large-HI/AML-Large-HI-ClassMixProtoBoundResid240Seed42.yaml` |

Formal mainline audit script:

```text
run/mainline_matched3_rawbest_audit.py
```

Formal mainline queue script:

```text
run/mainline_matched3_rawbest_queue.py
```

### 4. Compare Against The Original Baseline

Baseline audit script:

```text
run/original_raw_baseline_audit.py
```

Baseline queue script:

```text
run/original_raw_baseline_queue.py
```

Baseline model setting:

```text
SparseNodeGT + ports + Ego + dot decoder
```

### 5. Inspect Clean Ablations

Clean ablation scripts:

```text
run/paper_ablation_clean_audit.py
run/paper_ablation_clean_queue.py
```

Ablation variants:

| Variant | Meaning |
|---|---|
| `base_dot` | original FraudGT-style dot decoder |
| `supportmix_consis_proto` | support-conditioned evidence plus prototype context |
| `classmix_proto_bound` | class/prototype bounded residual route |
| `class_split` | classmix route plus positive/negative evidence split |
| `full_dual_uncert_gate` | class-split plus subgraph route and dual uncertainty gate |

These variants are partially incremental, but not a perfectly one-module-at-a-time chain. Do not assume F1 should monotonically increase from left to right.

Current ablation status in this snapshot:

```text
complete: 26/30
running at snapshot time: Large-LI / supportmix_consis_proto / seed44
missing: Large-LI classmix_proto_bound, class_split, full_dual_uncert_gate
```

### 6. Inspect The Decoder Implementation

Main decoder implementation:

```text
fraudGT/head/hetero_edge.py
```

Relevant route strings to search for:

```text
supportmixconsisproto
supportmixconsisclassmixprotoboundresid
supportmixconsisclassmixprotoboundclasssplitresid
supportmixconsisclassmixprotoboundclasssplitsubgraphdualuncertgateboundresid
```

Cross-GPU checkpoint resume compatibility:

```text
fraudGT/graphgym/checkpoint.py
```

This snapshot includes `torch.load(path, map_location=cfg.device)` to avoid checkpoint deserialization onto the wrong GPU during resume.

---

## Current Formal Result Summary

| Dataset | Baseline mean | Mainline mean | Delta |
|---|---:|---:|---:|
| Small-HI | 0.78572 | 0.80011 | +0.01439 |
| Small-LI | 0.49746 | 0.50219 | +0.00473 |
| Medium-HI | 0.76386 | 0.79855 | +0.03469 |
| Medium-LI | 0.52291 | 0.53470 | +0.01179 |
| Large-HI | 0.72126 | 0.76799 | +0.04673 |
| Large-LI | 0.42336 | 0.41485 | -0.00851 |

Average:

```text
baseline mean: 0.61910
mainline mean: 0.63640
delta: +0.01730
win count: 5/6
```

Do not claim that the current 3-seed mean wins on all six datasets. It does not.

---

## Recommended Next Direction

Based on the current ablation, the recommended next mainline is:

```text
ClassMix/Core Evidence + Gated ClassSplit + Safe Subgraph Router
```

Suggested design:

```text
z_base
  -> classmix_proto_bound core evidence
  -> gated class_split correction
  -> optional subgraph correction only for uncertain/conflicting cases
  -> z_final
```

Why:

- `classmix_proto_bound` is the strongest Small/Medium average ablation route.
- `class_split` is useful on some datasets but should not always be forced.
- `full_dual_uncert_gate` is helpful on Small-LI and Medium-HI but not uniformly best.
- Large graph results suggest unconditional high-order subgraph routing can add noise.

---

## Original FraudGT README

The remainder below is the original project README content.

---
# FraudGT: A Simple, Effective, and Efficient Graph Transformer for Financial Fraud Detection
![framework](imgs/framework.png)
This repository holds the code for FraudGT framework.

## Environment Setup
You can create a conda environment to easily run the code. For example, we can create a virtual environment named `fraudGT`:
```
conda create -n fraudGT python=3.9 -y
conda activate fraudGT
```
Install the required packages using the following commands:
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install pyg -c pyg
pip install -r requirements.txt
```

## Run the Code
You will need to firstly specify the dataset path (`./data` in this example) and log location (`./results` in this example) by editing the config file provided under `./configs/{dataset_name}/`. An example configuration is
```
......
out_dir: ./results
dataset:
  dir: ./data
......
```
Download and unzip the [Anti-Money Laundering dataset](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml) into your specified dataset path (for example, put the unzipped `HI-small.csv` into `./data`).
Dataset will be automatically processed at the first run.

![experiments](imgs/experiment.png)

For convenience, a script file is created to run the experiment with specified configuration. For instance, you can edit and run the `interactive_run.sh` to start the experiment.
```
cd FraudGT
chmox +x ./run/interactive_run.sh
./run/interactive_run.sh
```


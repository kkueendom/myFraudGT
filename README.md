# FraudGT — Evidence-Gated Edge Decoder (v2)

Branch: `feature/evidence-gate-decoder` (forked from `snapshot/fraudgt-results-20260626`).

This branch contains the original FraudGT codebase plus a **new, scale-agnostic
decoder** that replaces the previous "scale-adaptive" mainline. The encoder is
unchanged; the contribution is entirely at the edge-decoding stage.

> **Status:** the decoder and its configs are implemented and committed, but
> **not yet trained/evaluated on this branch**. There are no new result numbers
> here yet — run the smoke test and the evaluation plan below first. The result
> tables further down are from the *previous* snapshot and are kept only for
> context.

If you are another model or reviewer, **start with the handoff document**:

```text
docs/experiments/NEXT_STEPS_evidence_gate.md
```

It explains why the previous mainline was not yet publishable and the exact
experiment plan to validate this one.

---

## 1. What this branch adds

A single edge-decoding route, `evidence_gate`, selectable purely via config
(`model.edge_decoding: evidence_gate`). It is **one architecture used identically
on all six AML datasets — no dataset-size route switching.** It fuses three
sources of evidence:

```text
z_base   = MLP([h_u ‖ h_v ‖ e_uv])                         # original FraudGT decoder
z_core   = z_base + sigmoid(a_p) · ready · z_proto         # bounded class-prototype residual (stable core)
z_struct = MLP([ctx_out(u) ‖ ctx_in(v) ‖ h_uv])           # 1-hop higher-order structural evidence
g        = sigmoid(MLP[H(z_core), |proto margin|, ready, base margin])   # per-sample uncertainty gate
z_final  = z_core + g · sigmoid(a_s) · z_struct            # structural evidence routed by uncertainty
y_hat    = sigmoid(z_final)
```

Key idea: the gate `g` decides **per sample** how much higher-order structural
evidence to inject, driven by the *uncertainty* of the prototype core — not by
dataset size. On confident samples `g → 0`, so structural evidence is suppressed
automatically. This replaces the old hard `if scale == Large: fallback` rule
with a learned, input-conditioned mechanism.

Architecture figure: `docs/figures/evidence_gate_arch.svg` (and `.tex` for LaTeX).

## 2. How it differs from the previous (snapshot) mainline

| | Previous mainline (`snapshot/...`) | This branch (`evidence_gate`) |
|---|---|---|
| Architecture selection | **hard switch by dataset size** (full route on Small/Medium, fallback on Large) | **single architecture**, identical on all 6 datasets |
| Structural evidence | always-on (in the full route) | gated per-sample by core uncertainty |
| Decoder complexity | 60+ route strings, ~4.3k-line `_pair_chain_head` | one ~120-line method, 3 evidence sources |
| Prototype core | one of many stacked modules | kept as the stable backbone (reused verbatim) |

## 3. File map (this branch)

| File | Purpose |
|---|---|
| `fraudGT/head/hetero_edge.py` | `evidence_gate` route: `_build_evidence_gate`, `_evidence_gate_head`, `forward` dispatch (does not touch the old `_pair_chain_head`) |
| `configs/evidence_gate/AML-*.yaml` | six identical-architecture configs (only `name`/`batch_size`/`out_dir` differ) |
| `docs/experiments/NEXT_STEPS_evidence_gate.md` | handoff: why + the full experiment plan |
| `docs/figures/evidence_gate_arch.svg` / `.tex` | paper-ready architecture figure |

## 4. How to run

Smoke test first (on the experiment host `yyk@10.168.1.102`; the dev machine has
no working Python):

```bash
python run/main.py --cfg configs/evidence_gate/AML-Small-HI.yaml \
  optim.max_epoch 4 train.iter_per_epoch 16
```

Confirm: it runs, loss decreases, val F1 is computed, no shape/NaN errors, and
the prototype banks fill (`support_class_proto_ready > 0` after epoch 1).

Full evaluation (see the handoff doc for the rationale):

1. **Fix the metric first.** Report **validation-selected test F1** (`test_at_val`),
   not `raw_best_test_f1 = max_epoch test_f1`. The latter selects the epoch on the
   test set and is leaky; re-audit the baseline and previous mainline under the
   honest metric before comparing.
2. Run `configs/evidence_gate/AML-*.yaml` for **5 seeds** (42–46); report
   val-selected test F1 mean ± std **and PR-AUC**, with a paired significance test.
3. Clean **additive, single-architecture** ablation: `base → +prototype core →
   +structural (gate=1) → +uncertainty-gated structural`.
4. Frame the **low-illicit (LI) regime** as the case to fix (sweep the imbalance
   loss weight / focal loss), not as a wart.

Acceptance for `evidence_gate` to become the mainline: val-selected test F1 beats
the re-run baseline on ≥5/6 (ideally 6/6), **Large-LI ≥ baseline 0.42336**, and a
reported significance test.

---

## Previous snapshot results (context only — NOT this branch's results)

These are the previous **scale-adaptive** mainline's 3-seed numbers under the
**`raw_best_test_f1`** metric, kept for reference. See
`docs/experiments/fraudgt_current_results_for_model_review_20260626.md` and the
TSV audits in `docs/experiments/` for the full breakdown.

| Dataset | Baseline mean | Prev. mainline mean | Delta |
|---|---:|---:|---:|
| Small-HI | 0.78572 | 0.80011 | +0.01439 |
| Small-LI | 0.49746 | 0.50219 | +0.00473 |
| Medium-HI | 0.76386 | 0.79855 | +0.03469 |
| Medium-LI | 0.52291 | 0.53470 | +0.01179 |
| Large-HI | 0.72126 | 0.76799 | +0.04673 |
| Large-LI | 0.42336 | 0.41485 | **−0.00851** |

```text
baseline mean: 0.61910   prev. mainline mean: 0.63640   delta: +0.01730   wins: 5/6
```

Caveats (the reasons this branch exists): the metric above is leaky
(test-selected epoch); the previous mainline switched architecture by dataset
size; Large-LI regressed; and the clean ablation showed the simplest prototype
route (`classmix_proto_bound`) was the strongest Small/Medium average — i.e. the
most complex route was not uniformly best.

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

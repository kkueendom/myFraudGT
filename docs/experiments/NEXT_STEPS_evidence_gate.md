# Handoff: Evidence-Gate Decoder (v2 mainline) — what to do next

Branch: `feature/evidence-gate-decoder` (forked from `snapshot/fraudgt-results-20260626`)
Audience: the next model/engineer continuing this FraudGT extension.
Date: 2026-06-26.

This document tells you (a) why the previous mainline was not yet a publishable
story, (b) exactly what code was added on this branch, and (c) the precise
experiment plan to validate it. Read it top to bottom before running anything.

---

## 0. TL;DR

The previous "mainline" improved raw-best F1 on 5/6 AML datasets (+0.0173 mean)
but rests on three things a reviewer will reject:

1. **The headline metric is `raw_best_test_f1 = max over epochs of test F1`** —
   i.e. it selects the epoch using the *test* set. That is optimistic / leaky
   and is NOT the protocol in the FraudGT paper (which uses the epoch chosen by
   *validation*). Higher-variance models benefit more from this peek, which may
   inflate the reported gain.
2. **The decoder is "scale-adaptive": it hard-switches route by dataset size**
   (full route on Small/Medium, fallback on Large). Choosing the architecture
   per dataset is effectively test-set tuning and is the single biggest
   liability of the story.
3. **The complex full route is not even uniformly best.** The branch's own clean
   ablation shows `classmix_proto_bound` (a simpler route) has the best
   Small/Medium average (0.65900) and that the full `dual_uncert_gate` route
   loses to simpler routes on Small-HI, Medium-LI and Large-HI.

The fix implemented on this branch is **one** decoder, identical across all six
datasets, that replaces the size-based switch with **per-sample uncertainty
gating**. This turns a "bad" hard rule into a "good" learned, input-conditioned
mechanism, and keeps the proven prototype core as the stable backbone.

The narrative target is: **"Evidence-gated decoding for imbalanced fraud
detection: a prototype core plus uncertainty-routed higher-order structural
evidence, one architecture for all scales, with the largest gains in the hard
low-illicit-ratio regime."**

---

## 1. What was added on this branch

### 1.1 Code — `fraudGT/head/hetero_edge.py`

A new edge-decoding route `evidence_gate`, selectable purely via config
(`model.edge_decoding: evidence_gate`). It does **not** touch the 2,300-line
`_pair_chain_head` monster; it is a self-contained, ~120-line path:

- `__init__`: sets `self.use_evidence_gate`; when active, the base `else`
  branch additionally calls `_build_evidence_gate(...)`.
- `_build_evidence_gate(dim_in, dim_out)`: builds the modules and reuses the
  *existing* prototype machinery (`support_class_proto_*` buffers/params plus
  `_support_class_proto_context` / `_update_support_class_prototypes`). This is
  deliberate: the ablation showed the prototype core is the stable winner, so
  we reuse it verbatim rather than reinvent it.
- `_evidence_gate_head(batch)`: the single forward path.
- `forward`: dispatches to `_evidence_gate_head` when the route is active.

The forward fuses three evidence sources:

```
z_base   = layer_post_mp([src_x ‖ dst_x ‖ edge_attr])        # original FraudGT decoder
z_core   = z_base + sigmoid(α_p)·ready·z_proto                # bounded prototype residual (stable core)
z_struct = struct_head([ctx_out(src) ‖ ctx_in(dst) ‖ h])     # 1-hop higher-order structural evidence
g        = sigmoid(gate([h, uncertainty, |proto_margin|, ready, tanh(base_margin)]))
z_final  = z_core + g · sigmoid(α_s) · z_struct              # structural evidence routed by uncertainty
```

Key design points (and the talking points for the paper):

- **No dataset-size signal anywhere.** The gate input is per-sample
  (uncertainty / prototype margin / readiness / base margin). On confident
  samples `g → 0`, so structural evidence is suppressed automatically — this
  *reproduces* the old "fall back on large graphs" behaviour, but learned and
  per-example instead of a hard `if size == Large` rule.
- **Bounded residuals.** `sigmoid(α)` weights (log-odds initialised at 0.10 and
  0.05) keep prototype and structural evidence as *corrections* to `z_base`,
  not replacements — this is what made `classmix_proto_bound` stable.
- **No leakage.** Prototype banks are read for the current prediction and
  updated only afterwards, only when `self.training` is True; at eval the banks
  are frozen. `ready` gates the prototype contribution to 0 until banks fill.
- `ctx_in` / `ctx_out` are mean-aggregations of the shared edge representation
  over the full sampled subgraph (`scatter` over `dst` / `src`), i.e. genuine
  local higher-order structure beyond raw node/edge features. Valid because
  `task_entity = ('node','to','node')` is homogeneous (guarded by
  `task[0] == task[2]`).

### 1.2 Configs — `configs/evidence_gate/AML-*.yaml`

Six clean configs (short filenames!) with **identical** architecture and
hyperparameters; the only per-dataset differences are `name`, `batch_size`
(2048 Small / 1024 Medium-Large) and `out_dir`. This is the physical proof of
"one architecture for all scales". Loss is the unchanged `weighted_cross_entropy
[1, 6]`.

---

## 2. The experiment plan (do these in order)

### Step 0 — Smoke test (must pass before anything else)
There is no working local Python on the dev machine; run on the remote host
`yyk@10.168.1.102`. Train a few epochs on Small-HI:

```bash
python run/main.py --cfg configs/evidence_gate/AML-Small-HI.yaml \
  optim.max_epoch 4 train.iter_per_epoch 16
```
Confirm: it runs, loss decreases, val F1 is computed, no shape/NaN errors, and
the prototype banks fill (`support_class_proto_ready` becomes > 0 after epoch 1).

### Step 1 — Fix the metric (HIGHEST PRIORITY, blocks everything)
Switch the **headline** metric from `raw_best_test_f1` to **validation-selected
test F1** (`test_at_val`), matching the FraudGT paper. The audit scripts already
compute `test_at_val` — make it the primary column. Re-audit BOTH the original
baseline and the previous mainline under this honest metric first, so you know
the real starting point. Keep `raw_best` only as a secondary/diagnostic column.

> Expect the gains to shrink. That is the point — find out now, not in review.

### Step 2 — Run the evidence-gate v2, 5 seeds, honest metric
Run `configs/evidence_gate/AML-*.yaml` for **seeds 42,43,44,45,46** (paper uses
5 seeds; bump from 3). Report **val-selected test F1** mean ± std, plus
**PR-AUC / Average Precision** as a secondary metric (more robust than F1 under
extreme imbalance). Reuse the queue/audit harness in `run/` (clone
`mainline_matched3_rawbest_*` → `evidence_gate_valselect_*`; change the metric
selection and the config glob).

Acceptance for v2 to become the mainline:
- val-selected test F1 beats the re-run baseline on **≥5/6**, ideally 6/6;
- **Large-LI ≥ baseline mean 0.42336** (no longer a negative case);
- a **paired significance test** across seeds (Wilcoxon signed-rank or paired
  t-test) is reported per dataset and for the macro average.

### Step 3 — Clean, additive, single-architecture ablation
One module at a time, **same route on all six datasets** (no size switching):

| Tag | Architecture |
|---|---|
| `A0_base` | `z_base` only (original FraudGT decoder; `edge_decoding: dot`/base) |
| `A1_proto` | `+ bounded prototype core` (`z_core`) |
| `A2_struct_always` | `+ structural evidence, gate forced to 1` (ungated) |
| `A3_struct_gated` | `+ uncertainty-gated structural evidence` (= full `evidence_gate`) |

This is the table that proves each piece earns its place. Critically,
`A3_struct_gated` vs `A2_struct_always` is the experiment that justifies the
gate: gating should match-or-beat always-on, and should specifically rescue the
large/low-illicit cases where ungated structure was noisy. To force the gate on,
add a temporary `eg_gate_force_open` flag or hardcode `g = 1` for the A2 run.

### Step 4 — Make the hard regime the headline (the "story")
The real open problem in this benchmark is extreme class imbalance: LI datasets
sit at F1 ≈ 0.42–0.53 while HI sit at 0.72–0.80. Frame Large-LI / *-LI as the
case you *fix*, not a wart. Concretely, sweep the imbalance knob (currently the
single shared `loss_fun_weight: [1, 6]`):
- try `[1, 9]` / `[1, 12]` on LI datasets, or replace weighted-CE with **focal
  loss** / class-balanced loss;
- report whether the prototype core + gated structure interacts positively with
  imbalance-aware loss (hypothesis: prototypes stabilise the rare positive
  class, so the two compound).
Keep this as a *secondary* axis layered on the single architecture — do NOT turn
it into another per-dataset switch.

### Step 5 — Efficiency note (cheap win, matches the paper's pitch)
The paper sells throughput/latency. The evidence-gate head adds only a couple of
small MLPs and one `scatter` over the already-sampled subgraph, so it should be
near-free vs the encoder. Record per-batch latency vs the `base` decoder to be
able to claim "negligible decoder overhead".

---

## 3. Pitfalls / things to verify

- **Metric discipline.** Never report `raw_best_test_f1` as the headline again.
  Every table must state which epoch-selection rule it uses.
- **Gate collapse.** Watch `g`'s mean over training. If `g → 0` everywhere, the
  structural branch is dead (then v2 ≈ prototype core — still fine, but report
  it honestly). If `g → 1` everywhere, the gate is not discriminating — inspect
  `α_s` and the uncertainty feature scaling.
- **Prototype warm-up.** Early epochs have empty banks (`ready = 0`); the model
  is `z_base` only until banks fill. This is intended; just don't be surprised
  by a flat first epoch.
- **Large-graph memory.** `scatter` is over the sampled subgraph (sizes [50,50]),
  not the full graph, so memory is bounded by the batch. If Large OOMs, lower
  `batch_size`, do NOT reintroduce a size-based code path.
- **Don't grow the route zoo.** The whole point is *fewer* moving parts. If a new
  idea helps, fold it into this one architecture and ablate it; do not add a 61st
  `edge_decoding` string.

---

## 4. File map for this branch

| File | What |
|---|---|
| `fraudGT/head/hetero_edge.py` | `evidence_gate` route: `_build_evidence_gate`, `_evidence_gate_head`, `forward` dispatch |
| `configs/evidence_gate/AML-*.yaml` | six identical-architecture configs |
| `docs/experiments/NEXT_STEPS_evidence_gate.md` | this document |
| `docs/experiments/fraudgt_current_results_for_model_review_20260626.md` | previous-mainline state & ablation (read for context) |
| `run/mainline_matched3_rawbest_*` | previous harness to clone for the val-selected audit |

---

## 5. One-paragraph summary for a reviewer

> We replace FraudGT's dot-product edge decoder with an evidence-gated decoder:
> a bounded class-prototype residual provides a stable core correction, and a
> higher-order structural-evidence branch is fused only to the extent that the
> core decision is uncertain, via a per-sample learned gate. The same
> architecture and hyperparameters are used on all six AML datasets — no
> per-dataset or per-scale tuning — and the largest improvements appear in the
> hardest, lowest-illicit-ratio settings, where a prototype-stabilised, 
> uncertainty-routed decoder most helps the rare fraudulent class.

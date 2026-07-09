# Experiment-host next steps (after `git pull`)

Branch: `feature/evidence-gate-decoder`. Latest commit should be `c0c2886`
("Log evidence-gate g/uncertainty stats during eval") or newer.

Context: v2 (`evidence_gate_v2_fixed`) does **not** beat v1 under the
paper-consistent **val-selected test F1** metric (≈ −0.013 mean; it loses on the
LI datasets). Before spending more compute, we must (P0) check whether v2's
core mechanism — the per-sample uncertainty gate — is even active, then (P1)
finish and honestly recompute results. Do the steps in order.

---

## ✅ P0 RESULT (done) — the gate collapsed to always-open

The diagnostic on Large-HI and Medium-LI showed, for both val and test:
`g mean=1.0000 std=0.0000 frac>.95=1.000` (uncertainty itself was fine:
`uncert std ≈ 0.22–0.28`). So the per-sample gate **learned to be a constant 1
(fully open everywhere)**. The uncertainty routing is inactive; **v2 is really
"prototype core + an always-on structural branch"**, and that does not beat v1.

## 🔜 NEXT ACTION — run M1 vs v2 (decides the paper's main line)

This is now the top priority. It answers: is that always-on structural branch
helping, or is the **prototype core alone** the real contribution?

New code adds an M1 route `evidence_gate_proto` = base decoder + bounded
prototype residual ONLY (no structural evidence, no gate). Its configs are
`configs/evidence_gate_proto/AML-*.yaml` — **identical to the v2 configs except
that one branch** (same 500-epoch + early-stop budget), so M1 vs v2 is a clean,
budget-matched comparison.

**Steps:**
```bash
git pull                 # get the evidence_gate_proto route + M1 queue/configs
# NOTE: the queue hardcodes REPO=/e/yyk/FraudGT_evidence_gate_decoder.
# If your worktree differs, edit REPO in run/evidence_gate_m1_queue.py first
# (or export EVIDENCE_GATE_M1_OUT_DIR to an absolute path).

python run/evidence_gate_m1_queue.py          # runs M1: 6 datasets x seeds 42,43,44
                                              # results -> results/evidence_gate_m1_proto

# audit M1 under val-F1 (test_at_val):
python run/evidence_gate_v2_audit.py --roots <REPO>/results/evidence_gate_m1_proto
# v2 for comparison:
python run/evidence_gate_v2_audit.py --roots <REPO>/results/evidence_gate_v2_fixed
```

**Compare M1 vs v2 per dataset on `Test@Val F1` (3-seed mean), and decide:**
- **M1 ≥ v2 on most datasets** → the always-on structural branch is dead weight
  (or harmful). Main line = **prototype core (M1)**; drop the structural branch
  and gate. Write the gate collapse + this result as the paper's "extra evidence
  doesn't help / simplicity wins" finding. **Stop trying to save the gate.**
- **M1 < v2 on most datasets** → structural evidence helps even when always-on;
  only then is it worth fixing the gate (a `gate_v3` attempt). Report back first.

If time-constrained, prioritize the datasets that decide it: **Medium-LI,
Large-HI, Large-LI, Small-LI** (the ones where v2 diverged from expectation).

Report back the M1-vs-v2 val-F1 table before doing P2/P3.

## 🆕 v3 — the redesigned (fixed) gate

Root cause of the P0 collapse: `g` and `σ(α_s)` both multiplied `z_struct` (two
learned knobs on the same branch → non-identifiable → `g` drifts to constant 1).
The new route `evidence_gate_v3` fixes this:
- **removes** the redundant `σ(α_s)`;
- **convex fusion** `z_final = (1−g)·z_core + g·z_struct` (opening the gate costs
  you the reliable core, so `g` can't trivially saturate);
- gate fed **routing signals only** `[uncertainty, |proto_margin|, ready,
  struct_support, |base_margin|]` (no `h`), started **shut** (negative bias),
  with an **L1 penalty** `λ·mean(g)` (`cfg.model.eg_gate_l1`, default 1e-3) and a
  **warm-up** (`cfg.model.eg_gate_warmup_epochs`, default 20).

Base + prototype core are identical to M1, so **v3 vs M1** isolates whether a
*properly-gated* structural branch beats the prototype core alone.

**Steps (after M1 vs v2):**
```bash
git pull
# smoke test first (banks warm, no NaN, loss falls, gate penalty finite):
python -m fraudGT.main --cfg configs/evidence_gate_v3/AML-Small-HI.yaml \
  --repeat 1 --gpu 0 optim.max_epoch 4 train.iter_per_epoch 16 seed 42
# then the queue (seeds 42-44 -> results/evidence_gate_v3):
python run/evidence_gate_v3_queue.py
python run/evidence_gate_v2_audit.py --roots <REPO>/results/evidence_gate_v3
```

**Verify the gate is now ALIVE (primary):** grep `evidence_gate/` in a v3 eval
log — `g.std` must be clearly > 0 and `frac<.05` substantial (NOT `g=1.000
std=0`). `g` should track `uncertainty`. If it still collapses, report back
before running all seeds.

**Decide:** compare **M1 vs v2 vs v3** on `Test@Val F1` (3-seed mean, all 6
datasets). v3 is only worth keeping if it **beats M1**; otherwise the honest
conclusion is the prototype core (M1) is the main line and the gate is a
documented negative result.

---

## P0 — Gate diagnostic (reference; already DONE, result above)

Commit `c0c2886` adds an **eval-only, throttled** log line in
`fraudGT/head/hetero_edge.py::_evidence_gate_head`. It prints, during
validation/test, the distribution of the gate `g` and the `uncertainty` signal.

**Do this:**
1. Pick two datasets that behaved differently: **Large-HI** (v2 degrades vs v1)
   and **Medium-LI** (v2's val-select collapses the hardest).
2. Trigger a fresh **eval pass with the updated code** on an existing v2
   checkpoint (load checkpoint → run val/test only if the harness supports it;
   otherwise resume the run so at least one eval pass executes). The already
   finished runs do NOT contain these logs — you must run eval again with the
   new code.
3. `grep "evidence_gate/" <log>` and report the lines for both `val` and `test`.

**Interpret (this is the whole point):**
- `g.std ≈ 0` and (`frac<.05 ≈ 1.0` or `frac>.95 ≈ 1.0`) → **gate collapsed to a
  constant**. The uncertainty routing is inactive; v2 ≈ prototype core only.
  Conclusion: the v2 "innovation" isn't doing anything → stop propping up v2,
  pivot the paper to the prototype core (see P2/P3).
- `g.std` clearly > 0, p10/p50/p90 spread out → gate is genuinely per-sample.
  Then v2's mechanism works but simply doesn't help → still a valid (negative)
  ablation result.
- `uncert.std ≈ 0` → the uncertainty input itself is degenerate (single-logit
  binary). The gate has nothing to route on; note it and consider feeding the
  gate a stronger signal instead of adding more modules.

**Report back:** paste the `evidence_gate/` lines and state which of the three
cases holds for each dataset.

---

## P1 — Finish and recompute everything honestly under val-F1

1. **Finish `Large-LI seed44`** for v2 (the only missing v2 cell).
2. Using `run/evidence_gate_v2_audit.py`, recompute v2 for all 6 datasets in
   **all four views**: {raw-best, val-select} × {3-seed mean, best-seed}. The
   **headline is val-selected test F1 (`test_at_val`), mean ± std** — this is
   what matches the FraudGT paper. `raw_best` is diagnostic only (it selects the
   epoch on the test set; do not headline it).
3. Emit a clean **three-way table (baseline / v1 / v2) under val-F1**. For
   reference, these were computed from the committed audit TSVs
   (`docs/experiments/*.tsv`), val-select 3-seed mean:

   | Dataset | Baseline | v1 | v2 |
   |---|---:|---:|---:|
   | Small-HI | 0.7728 | 0.7787 | 0.7749 |
   | Small-LI | 0.4794 | 0.4221 | 0.4427 |
   | Medium-HI | 0.7288 | 0.7859 | 0.7556 |
   | Medium-LI | 0.4408 | 0.4787 | 0.4022 |
   | Large-HI | 0.6939 | 0.7144 | 0.7000 |
   | Large-LI | 0.3403 | 0.2983 | (finish seed44) |
   | **Mean** | **0.5760** | **0.5797** | — |

   Confirm these on the host (baseline uses column `test_at_best_val`; v1/v2 use
   `test_at_val`). v1 only beats baseline by +0.0036 and loses on Small-LI /
   Large-LI — so v1 is not a strong story either.
4. Add **PR-AUC / Average Precision** as a supplementary column (robust under
   imbalance; does not replace F1).

---

## P2 — Fix the epoch-budget confound before any fair claim

Budgets are currently mismatched: **baseline = 500 ep, v1 = 180/240 ep, v2 = 500
ep + early stop** (`early_stop_metric: f1`, `min_epoch 300`, `patience 80`). The
v1-vs-v2 comparison in the summary is therefore v1@240 vs v2@500.

- Equalize the budget (pick one protocol, e.g. 500 ep + the same early-stopping
  for all methods), or at minimum document the mismatch in every table.
- Also check `early_stop.json` in each run dir: on the LI datasets, early
  stopping on **val-F1** may have fired early at a bad point (val-F1 is very
  noisy under extreme imbalance), which would itself depress val-select. If so,
  switch the early-stop metric to val loss or val AP for the LI datasets.

---

## P3 — The critical-path experiment for the paper (controlled ablation)

The current ablation (`run/paper_ablation_clean_queue.py`) is **not publishable**:
its variants come from two different config lineages, mix epoch budgets (180 vs
240), are single-seed, and jump ~11 modules at once (base → supportmix). Also,
the "winning" `classmix_proto_bound` is **not** a simple prototype core — its
route carries a full PairChain/sequence stack underneath, so the prototype
contribution has never been isolated.

Design a **controlled, one-factor-at-a-time** ablation on a single code path
(toggle sub-modules by flag, identical epochs/seeds/datasets incl. Large-LI):

| Tag | Config |
|---|---|
| M0 | base dot decoder (`z_base`) |
| **M1** | **M0 + bounded class-prototype residual ONLY** ← isolates the real core |
| M2 | M1 + sequence/support stack (PairChain etc.) |
| M3 | M1 + class-split |
| M4 | M1 + high-order subgraph |
| M5 | M1 + uncertainty gate |
| Mfull | M1 + all |

Run each × 6 datasets × ≥3 seeds under val-F1 with mean ± std + a paired
significance test. **Run M0 and M1 first** — they decide the paper: if M1 beats
M0, the prototype core is the contribution and M2–M5 become "extra complexity
doesn't help" controls; if M1 does not beat M0, the story must change.

---

## Priority summary

1. **P0 gate diagnostic** — cheap, tells us if v2's mechanism is alive. Do first.
2. **P1** — finish Large-LI seed44, recompute val-F1 three-way + PR-AUC.
3. **P2** — equalize epoch budget / audit early-stop on LI.
4. **P3** — rebuild the ablation as a controlled M0–Mfull chain; run M0/M1 first.

Report back after P0 with the `evidence_gate/` log lines before launching P3.

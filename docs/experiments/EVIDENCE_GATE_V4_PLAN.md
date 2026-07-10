# Evidence Gate v4: Residual Router

## Problem

v3 learns a nearly always-closed gate because its fusion rule is:

```text
z_final = (1 - g) * z_core + g * z_struct
```

Opening `g` replaces a reliable core prediction with an initially weak
structural prediction. During the zero-gate warm-up, `z_struct` also receives no
classification gradient. A closed gate is therefore the easiest stable solution.

## v4 Architecture

v4 changes the structural branch from a replacement expert into a bounded
correction expert:

```text
delta_struct = residual_scale * tanh(struct_head(struct_feat))
z_candidate  = z_core + delta_struct
z_final      = z_core + g * delta_struct
```

The core prediction is never removed. The router only decides how much of the
proposed correction to apply.

The seven router inputs are:

1. Core uncertainty.
2. Absolute prototype margin.
3. Prototype readiness.
4. Local structural support.
5. Absolute base-logit margin.
6. Structural correction magnitude.
7. Core/correction alignment.

Router inputs are detached from the experts. The router learns how to route
evidence without indirectly changing the core merely to make routing easier.

## Training

- Epochs 0-19: use a fixed `g = 0.10`, not `g = 0`.
- Epochs 0-59: add an auxiliary loss on
  `stopgrad(z_core) + delta_struct`.
- After warm-up: use the learned per-sample router.
- Regularize only the mean gate budget:

```text
L_gate = lambda_budget * (mean(g) - target_gate)^2
```

- Full objective during the auxiliary phase:

```text
L = L_cls(z_final, y)
  + lambda_aux * L_cls(stopgrad(z_core) + delta_struct, y)
  + L_gate
```

The no-gate diagnostic uses the same bounded expert:

```text
z_final = z_core + delta_struct
```

It answers whether the structural correction itself is useful before blaming
the router.

## Non-Duplicative Screening

Existing M1 and v3 runs are reused and truncated to their first 80 epochs.
They are not retrained.

1. Smoke: v4 router, Small-HI, seed 42, 4 epochs x 16 iterations.
2. v4 no-gate, Small-HI, seed 42, 80 epochs.
3. v4 no-gate, Small-LI, seed 42, 80 epochs.
4. v4 router, Small-HI, seed 42, 80 epochs.
5. v4 router, Small-LI, seed 42, 80 epochs.

The comparison metric is raw-best test F1 within epochs 0-79, matching the
current reporting rule.

## Decision Rules

- **No-gate <= M1 on both datasets:** the structural expert is weak. Stop
  tuning the router and redesign structural evidence.
- **No-gate > M1, router <= no-gate:** the correction is useful but routing is
  still the bottleneck. Tune the router/budget only.
- **Router > M1 on both datasets:** proceed to three seeds and then all six
  datasets.
- **Only one dataset improves:** inspect support distribution and correction
  magnitude before scaling up.

Healthy v4 router diagnostics should show:

- `delta_abs` is nonzero and changes during training.
- Gate standard deviation is nontrivial after warm-up.
- Most samples are not below `g < 0.05`.
- Gate behavior differs between high-support and low-support samples.

## Commands

Start the low-frequency queue:

```bash
nohup python run/evidence_gate_v4_wait_queue.py \
  > .evidence_gate_v4_queue.stdout 2>&1 &
```

Audit the 80-epoch screen:

```bash
python run/evidence_gate_v4_screen_audit.py
```

Useful environment overrides:

```bash
export EVIDENCE_GATE_V4_POLL_SECONDS=1800
export EVIDENCE_GATE_V4_MIN_FREE_MIB=9000
export FRAUDGT_PYTHON=/d/miniconda3/envs/fraudGT/bin/python3.9
```

## Final 80-Epoch Screen Result

The completed raw-best test-F1 screen (seed 42, epochs 0-79) is:

| Variant | Small-HI | Small-LI | Mean | Delta vs M1 |
|---|---:|---:|---:|---:|
| M1 prototype core | 0.76600 | 0.44646 | 0.60623 | +0.00000 |
| v3 convex gate | 0.77943 | 0.45773 | 0.61858 | +0.01235 |
| v4 no-gate | 0.77707 | 0.46365 | 0.62036 | +0.01413 |
| **v4 residual router** | **0.77311** | **0.47925** | **0.62618** | **+0.01995** |

The v4 residual router beats M1 on both screening datasets:

- Small-HI: +0.00711 F1.
- Small-LI: +0.03279 F1.
- Mean: +0.01995 F1.
- Mean versus v3: +0.00760 F1.

The router is learned rather than constant, although its sample-level spread is
small (`g.std` about 0.003-0.005 on Small-HI and 0.002 on Small-LI). The Small-LI
structural correction often reaches its bound, so the formal experiment must
retain the no-gate ablation as a control.

**Decision:** promote `evidence_gate_v4` as the formal mainline and run all six
datasets with seeds 42, 43, and 44.

## Formal Mainline Run

The formal queue is `run/evidence_gate_v4_formal_queue.py`.

- Matrix: 6 datasets x 3 seeds = 18 runs.
- Budget: 500 complete epochs per run.
- Early stopping: disabled, matching the raw-best test-F1 reporting rule.
- Output: `results/evidence_gate_v4_formal`.
- GPU selection: configurable through `EVIDENCE_GATE_V4_FORMAL_GPU_IDS`.
- Queue persistence: run inside a detached tmux session on the experiment host.

Example:

```bash
env \
  FRAUDGT_V4_FORMAL_REPO=/e/yky/FraudGT_evidence_gate_v4 \
  EVIDENCE_GATE_V4_FORMAL_OUT_DIR=/e/yky/FraudGT_evidence_gate_v4/results/evidence_gate_v4_formal \
  EVIDENCE_GATE_V4_FORMAL_GPU_IDS=0,2,3 \
  EVIDENCE_GATE_V4_FORMAL_POLL_SECONDS=1800 \
  EVIDENCE_GATE_V4_FORMAL_MIN_FREE_MIB=14000 \
  EVIDENCE_GATE_V4_FORMAL_MAX_UTIL=10 \
  FRAUDGT_PYTHON=/d/miniconda3/envs/fraudGT/bin/python3.9 \
  /d/miniconda3/envs/fraudGT/bin/python3.9 \
  run/evidence_gate_v4_formal_queue.py
```

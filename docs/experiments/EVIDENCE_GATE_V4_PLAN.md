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
export EVIDENCE_GATE_V4_POLL_SECONDS=900
export EVIDENCE_GATE_V4_MIN_FREE_MIB=9000
```

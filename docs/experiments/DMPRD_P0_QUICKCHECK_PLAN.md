# DMPRD P0 Reliability Quick-Check Plan

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan + run
- Plan Date: 2026-07-17
- Verification Status: PRE-REGISTERED / RUNNING NOT YET STARTED
- Branch: `feature/support-calibrated-prototype-residual`
- Parent Commit: `65706ed`

## Research Question

Can support- and uncertainty-calibrated prototype reliability make the A2
prototype residual more useful at validation-selected checkpoints, without
losing its raw-best F1 advantage?

## Method Change

A2 applies the same learned global residual scale whenever both class prototype
banks are ready. P0 additionally records, for every prototype slot:

- accumulated assigned sample count `n`;
- running within-cluster squared distance `v`.

Its slot reliability is

```text
q_slot = n / (n + tau_n) * exp(-v / tau_v).
```

For each query, positive and negative class reliabilities are averaged using
that query's prototype-assignment weights. Their geometric mean is the local
reliability `q_local`. The full sample signal is

```text
signal = q_local * base_uncertainty * tanh(abs(proto_margin) / tau_margin)
gate   = gate_floor + (1 - gate_floor) * signal
z      = z_base + beta * ready * gate * delta_proto.
```

The gate is detached from backpropagation. It controls residual magnitude but
cannot train the base classifier to manufacture high uncertainty or margin.
The floor keeps a bounded part of A2 active when reliability is low.

## Controlled Variants

| Variant | Support/variance | Sample gate | Purpose |
|---|---:|---:|---|
| `a2_control` | no | no | exact matched A2 control |
| `p0_supportvar` | yes | no | isolate local prototype reliability |
| `p0_full` | yes | yes | complete P0 method |

All variants use four prototypes per class, no A3 distribution statistics,
the same seed, the same configuration, and exactly 120 epochs.

## Tasks and GPU Policy

Seven tasks are queued:

| Dataset | Seed | Variants |
|---|---:|---|
| Small-LI | 42 | A2, support/variance only, full P0 |
| Medium-HI | 42 | A2, full P0 |
| Large-LI | 44 | A2, full P0 |

Large-LI and Medium-HI pairs launch first to minimize wall-clock time. The
queue uses every GPU with at least 14 GiB free and no compute PID. GPU 0/1 are
automatically excluded while other users' processes remain. Polling is every
600 seconds and unchanged wait states are not repeatedly logged.

## Metrics and Decision Rule

Primary metric: test F1 at the checkpoint selected by validation F1.

Secondary diagnostic: raw-best test F1. It is useful for screening but is not
valid as the paper's official model-selection rule.

P0 passes the quick screen only if all conditions hold across the three paired
datasets:

1. mean validation-selected test-F1 gain is greater than `+0.005`;
2. validation-selected F1 improves on at least `2/3` datasets;
3. mean raw-best gain is non-negative;
4. at most one raw-best regression is below `-0.010`.

If P0 passes, run all six datasets at 500 epochs, then three seeds with
validation-selected reporting and matched A2 controls. If it fails, stop this
direction rather than tuning on test F1. A borderline result may justify one
predefined gate-floor sensitivity check, but not an unrestricted parameter
search.

## Reproduction

```bash
nohup python3 run/dmprd_p0_quick_queue.py \
  > .dmprd_p0_quick120_queue.nohup.log 2>&1 &

python3 run/dmprd_p0_quick_audit.py
```

Expected results directory: `results/dmprd_p0_quick120/`.

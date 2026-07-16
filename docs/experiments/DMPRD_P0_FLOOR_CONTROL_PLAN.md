# DMPRD P0 Fixed-Floor Control Plan

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: post-screen diagnostic plan
- Plan Date: 2026-07-17
- Verification Status: PRE-REGISTERED / NOT YET STARTED
- Branch: `feature/support-calibrated-prototype-residual`
- Trigger: P0 final mean gates were close to the configured `0.25` floor

## Question

Does P0 improve because its gate varies with reliability, or because it reduces
the prototype residual to an almost fixed quarter of A2's magnitude?

## Fixed-Floor Control

The control keeps the same P0 code, support/variance buffers, optimizer, seed,
and 120-epoch budget. It sets `dmprd_margin_tau=1e9`, making
`tanh(abs(proto_margin) / tau)` numerically zero. Therefore:

```text
gate = 0.25 + 0.75 * q_local * base_uncertainty * 0 = 0.25.
```

This changes no model parameters and isolates a fixed residual shrinkage from
sample-dependent reliability.

## Tasks

| Dataset | Seed | Task |
|---|---:|---|
| Small-LI | 42 | fixed-0.25 control |
| Medium-HI | 42 | fixed-0.25 control |
| Large-LI | 44 | fixed-0.25 control |
| Small-LI | 43 | matched A2 |
| Small-LI | 43 | matched full P0 |

The five tasks launch concurrently on GPU 2-6. GPU 0/1 remain excluded while
other users have compute PIDs.

## Decision Rule

Sample-dependent reliability contributes only if full P0 versus fixed-0.25:

1. improves mean validation-selected test F1 by more than `+0.005`;
2. wins validation-selected F1 on at least `2/3` datasets;
3. has mean raw-best delta of at least `-0.005`.

If this check fails, the current evidence supports controlled residual
shrinkage, not the claimed dynamic reliability mechanism. The architecture must
then be simplified or the gate redesigned before formal 500-epoch experiments.

The seed43 pair is an additional robustness signal only and is not included in
the dynamic-gate go/no-go rule.

## Reproduction

```bash
nohup python3 run/dmprd_p0_floor_control_queue.py \
  > .dmprd_p0_floor_control120_queue.nohup.log 2>&1 &

python3 run/dmprd_p0_floor_control_audit.py
```

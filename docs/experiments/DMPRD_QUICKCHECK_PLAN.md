# DMPRD Go/No-Go and Formal Screen

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan + run
- Origin Date: 2026-07-15
- Verification Status: QUICK SCREEN VERIFIED / FORMAL SCREEN PENDING
- Version Label: dmprd_quickcheck_v2

## Objective

Test whether a bounded multi-prototype residual is a better mainline than
EvidenceGate-v4 before spending compute on formal three-seed runs.

## Execution Status

- Branch: `feature/dmprd-prototype-residual`
- Worktree: `/e/yky/FraudGT_dmprd_quickcheck`
- Four-epoch, 16-iteration smoke test: passed on 2026-07-15.
- Smoke diagnostics: no non-finite values; prototype readiness `1.0000`;
  learned beta about `0.10`; maximum absolute bounded residual about `0.203`
  under a configured upper bound of `1.0`.
- Main 120-epoch screen: completed without failures on 2026-07-15.
- A3 versus epoch-matched `proto_only`: mean val-select delta `+0.03766`,
  mean raw-best delta `+0.02198`, and zero raw-best losses below `-0.005`.
- Six-dataset quick decision: `PASS`.
- Small-LI ablation selected A2: four prototype slots are useful, while the
  added distribution statistics in A3 do not improve either selection rule.
- Formal 500-epoch A2 screen: pending at the time of this document update.

## Quick Results

| Small-LI variant | Val-selected test F1 | Raw-best test F1 |
|---|---:|---:|
| A1 single prototype | 0.45081 | 0.48178 |
| A2 four prototypes | **0.46575** | **0.48880** |
| A3 four prototypes + distribution statistics | 0.46024 | 0.48790 |

The multi-slot change contributes `+0.01494` val-select and `+0.00702`
raw-best over A1. Distribution statistics reduce val-select by `0.00551` and
raw-best by `0.00090` relative to A2, so they are removed from the formal route.

## Reused Controls

- Baseline: existing matched FraudGT seed-level results; no retraining.
- Unbounded prototype reference: completed `proto_only` runs under
  `/e/yky/FraudGT_evidence_gate_v4/results/evidence_gate_v4_best_seed_ablation`.
- The quick audit truncates the reference to epoch 119 for a fair trajectory
  comparison with the new 120-epoch runs.

## Variants

| Variant | Slots per class | Distribution statistics | Residual |
|---|---:|---:|---|
| A1 `a1_single` | 1 | off | bounded `tanh`, learned global beta |
| A2 `a2_multi` | 4 | off | bounded `tanh`, learned global beta |
| A3 `a3_full` | 4 | peak + concentration | bounded `tanh`, learned global beta |

All variants use the same `dmprd` code path. Only the listed knobs differ.

## Task Order

1. Small-LI: A1, A2, A3 in parallel to test the additive ablation chain.
2. Medium-HI: A3 to ensure the new bound preserves the strongest v4 regime.
3. A3 on Large-LI, Small-HI, Medium-LI, Large-HI as GPUs become available.

The first four jobs are intentionally ordered to occupy the four GPUs that were
idle when the plan was created. The queue skips every GPU with an existing
compute PID, so the two running `no_prototype` jobs and external users are not
interrupted.

## Commands

Smoke test:

```bash
CUDA_VISIBLE_DEVICES=<idle_gpu> /d/miniconda3/envs/fraudGT/bin/python3.9 \
  -m fraudGT.main \
  --cfg configs/evidence_gate_v4/AML-Small-HI.yaml \
  --repeat 1 --gpu 0 \
  out_dir results/dmprd_smoke \
  name_tag DMPRDSmoke \
  seed 42 optim.max_epoch 4 train.iter_per_epoch 16 \
  train.early_stop False train.tqdm False val.tqdm False \
  model.edge_decoding dmprd \
  model.dmprd_num_slots 4 \
  model.dmprd_use_distribution_stats True
```

Queue:

```bash
nohup env DMPRD_MAX_EPOCH=120 DMPRD_POLL_SECONDS=600 \
  /d/miniconda3/envs/fraudGT/bin/python3.9 \
  run/dmprd_quick_queue.py \
  >> .dmprd_quick120_queue.nohup.log 2>&1 &
```

Audit:

```bash
/d/miniconda3/envs/fraudGT/bin/python3.9 \
  run/dmprd_quick_audit.py
```

## Go/No-Go Rule

The quick screen compares A3 with `proto_only` using only epochs 0-119.

Promote the selected variant to a 500-epoch best-seed screen only if, across
six datasets:

- mean raw-best delta versus truncated `proto_only` is at least 0;
- mean val-select delta versus truncated `proto_only` is greater than 0;
- no more than two datasets have raw-best delta below -0.005;
- Small-LI A2/A3 shows whether multi-slot or distribution statistics earns a
  place. Any non-contributing layer is removed before the 500-epoch run.

The quick screen passed and the ablation selected A2. The formal screen runs
A2 on all six datasets for 500 epochs without early stopping. Its primary
paper-facing criterion is raw-best test F1: all six datasets must beat the
historical baseline and the mean gain must be at least `0.005`. The audit also
reports val-select and full-budget `proto_only` deltas.

If the formal screen fails but the existing `proto_only` remains above baseline, the fallback
mainline is the simpler prototype residual. Do not add another gate.

Formal queue:

```bash
nohup env DMPRD_FORMAL_MAX_EPOCH=500 \
  DMPRD_FORMAL_POLL_SECONDS=600 \
  /d/miniconda3/envs/fraudGT/bin/python3.9 \
  run/dmprd_formal_queue.py \
  >> .dmprd_formal500_queue.nohup.log 2>&1 &
```

Formal audit:

```bash
/d/miniconda3/envs/fraudGT/bin/python3.9 \
  run/dmprd_formal_audit.py
```

## Outputs

- Results: `results/dmprd_quick120`
- Queue events: `.dmprd_quick120_queue.events`
- Active markers: `.dmprd_quick120_active/`
- Failed markers: `.dmprd_quick120_failed/`
- Audit: stdout from `run/dmprd_quick_audit.py`
- Formal results: `results/dmprd_formal500`
- Formal queue events: `.dmprd_formal500_queue.events`
- Formal audit: stdout from `run/dmprd_formal_audit.py`

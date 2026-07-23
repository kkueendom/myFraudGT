# TIER Phase 0 Coverage Audit Plan

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-07-23
- Verification Status: UNVERIFIED
- Version Label: tier_phase0_plan_v1

## Experiment Overview

- **Objective**: Verify that independent, time-admissible raw transaction
  evidence is present at usable coverage on Small-LI and Large-LI before
  training an evidence classifier.
- **Hypothesis**: Both datasets contain non-trivial context for normal and
  illicit target transactions, with active role or motif features.
- **Type**: Deterministic data validation plus stochastic dynamic-sampling
  coverage audit; no model training.

## Registered Setup

- Small-LI, seed 42, batch size 2048.
- Large-LI, seed 44, batch size 1024.
- Train, validation, and test use `LinkNeighborLoader(..., shuffle=True)`.
- `val.fixed_target_panel=False`.
- No fixed target panel, independent generator, or RNG restoration.
- Original `val.iter_per_epoch=256` remains configured.
- Audit stops after at least four batches and 32 positives, or at the registered
  cap of 32 batches. This cap is diagnostic and does not define a performance
  epoch.
- Maximum 32 evidence tokens per target.

## Entry Command

```bash
PYTHONDONTWRITEBYTECODE=1 /d/miniconda3/envs/fraudGT/bin/python3.9 \
  run/tier_phase0_coverage_audit.py \
  --output-dir /e/yky/FraudGT_tier_phase0_results/phase0_<commit>
```

## Outputs

| Output | Format | Success criterion |
|---|---|---|
| `coverage_rows.jsonl` | JSONL | One complete row per dataset and split |
| `experiment_manifest.jsonl` | JSONL | Commit/config/seed/protocol and no-training fields recorded |
| `phase0_summary.md` | Markdown | Coverage and resource tables generated |
| `tier_raw_edge_attr_v1.pt` | PyTorch sidecar | Schema and source-boundary validation passes |

## Mandatory Invariants

- Sampled `raw_edge_attr` equals the full graph value indexed by global `e_id`.
- Batch `target_edge_id` equals the registered split target indexed by
  `input_id`.
- A target edge never appears in its own evidence.
- Context is earlier in time, or tied in time with a smaller global edge ID.
- No normalized encoder `edge_attr` is used as a raw-evidence fallback.

## Continue / Stop Gate

Each split passes static coverage when:

- overall non-empty evidence coverage is at least 0.25;
- illicit-target coverage is at least 0.10;
- at least 10 illicit targets were sampled.

Fewer than 10 illicit samples is `inconclusive`, not a failure. Any invariant
violation is a hard stop. A coverage pass only authorizes Phase 1
evidence-only qualification; it is not evidence of predictive improvement.

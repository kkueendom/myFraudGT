# Multi-CDVT Formal 256-Batch Runtime Plan

## Material Passport

- Experiment ID: multi_runtime_formal_256_six_dataset
- Type: matched inference runtime benchmark
- Status: queued
- Branch: experiment/multi-cdvt-runtime
- Datasets: AML Small/Medium/Large, LI and HI
- Models: Multi-FraudGT account backbone and Multi-CDVT
- Hardware: one NVIDIA GeForce RTX 2080 Ti
- Sampling protocol: dynamic_random

## Protocol

Each model and dataset receives four warm-up batches followed by eight
independent 32-batch measurement windows, for 256 measured batches. Both
models run serially on the same physical GPU. Timing is normal-only end-to-end
inference and includes dynamic sampling, CPU data preparation, and GPU forward
computation.

The public loader has a finite physical pass even when its configured evaluation
budget is larger. If a 32-batch measurement window reaches that boundary, the
formal benchmark creates the next dynamic-random loader iterator and continues.
It neither fixes target edges nor restores sampler RNG. The number of such
natural loader-pass restarts is retained in every benchmark artifact.

The queue waits until cdvt_phase1_screen.py training processes have exited and
the selected GPU is idle. This prevents the active Small-LI multi-seed training
from contaminating the paper-facing latency measurements.

## Required Artifacts

Each of the 12 model-dataset tasks must produce benchmark.json with:

- phase=CDVT_multi_runtime_formal_256;
- evidence_tier=formal_256;
- total_measured_batches=256;
- loader restart policy and restart count;
- parameter count, peak GPU memory, latency, throughput, environment metadata;
- complete dynamic_random loader audit.

The queue must finish with runtime_summary.json, runtime_summary.md, and a
successful queue_complete.json.

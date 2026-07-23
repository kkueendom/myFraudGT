# TIER Vectorized Evidence Query Benchmark

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Execution Date: 2026-07-23
- Verification Status: EXECUTED
- Code Commit: `0a005d87`
- Sampling Protocol: `dynamic_random`
- Training: none
- Remote artifacts:
  `/e/yky/FraudGT_tier_phase0_results/vectorized_0a005d87/`

## Purpose

Phase 0 showed adequate evidence coverage, but the Python reference query was
too slow for repeated evidence-only training. The preregistered continuation
gate required the vectorized query to:

1. match the reference implementation exactly on context IDs, masks, tokens,
   support statistics, and motif flags;
2. preserve all temporal and target-edge invariants;
3. improve Large-LI query throughput by at least 10x, or otherwise make the
   registered evidence-only experiment feasible.

## Correctness

The vectorized implementation was compared against the retained reference
query:

- deterministic hand-built graphs;
- 12 randomized directed multigraphs;
- multiple token caps and time windows;
- repeated target IDs, tied timestamps, empty histories, and self-loops.

All equality tests passed. Phase 0 was then repeated under the original
dynamic-random loader. Coverage, class-conditional coverage, token counts,
support statistics, and role/motif activation matched the reference run.

## Performance

| Dataset | Reference query time | Vectorized query time | Targets | Reference throughput | Vectorized throughput | Query speedup |
|---|---:|---:|---:|---:|---:|---:|
| Small-LI | 210.47 s | 0.9499 s | 182,272 | 866/s | 191,890/s | 221.6x |
| Large-LI | 1123.74 s | 1.4784 s | 98,304 | 87/s | 66,493/s | 760.1x |

The query times are summed across train, validation, and test audit splits.

| Dataset | Reference total time | Vectorized total time | End-to-end speedup |
|---|---:|---:|---:|
| Small-LI | 231.67 s | 21.02 s | 11.02x |
| Large-LI | 1268.04 s | 116.31 s | 10.90x |

End-to-end time includes dataset loading, sidecar loading, incident-index
construction, dynamic neighbor sampling, invariant checks, and result writing.

## Resource Check

| Dataset | Index build | Peak RSS | Sidecar |
|---|---:|---:|---:|
| Small-LI | 0.62 s | 2.63 GiB | 110,786,373 bytes |
| Large-LI | 14.06 s | 43.40 GiB | 2,817,066,501 bytes |

Large-LI remains memory intensive on the host because the complete immutable
event index is resident in CPU memory. This does not consume GPU memory, but
Phase 1 must avoid constructing one copy per data-loader worker or per GPU
process.

## Decision

The vectorized query **passes** the preregistered continuation gate:

- exact-equivalence tests pass;
- dynamic-sampling invariants pass;
- Large-LI query speedup is about 760x;
- Large-LI end-to-end audit speedup is 10.90x.

This removes the engineering blocker and authorizes Phase 1 evidence-only
qualification. It does not establish predictive signal, complementarity with
A2, or a valid TIER full model. CrossFusion and ErrorRouter remain out of scope
until normal evidence beats shuffled/off evidence and passes the registered
error-correction gates.

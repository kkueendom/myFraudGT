# TIER Phase 0 Coverage Results

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Execution Date: 2026-07-23
- Verification Status: EXECUTED
- Code Commit: `1d37669c`
- Sampling Protocol: `dynamic_random`
- Training: none
- Raw artifacts: `docs/experiments/tier_phase0_1d37669c/`

## Decision

Phase 0 **passes** on Small-LI seed 42 and Large-LI seed 44. Independent
time-admissible transaction context has enough coverage on both scales to
justify Phase 1 evidence-only qualification.

This result does **not** show that the evidence predicts fraud or corrects A2.
It only establishes that the required evidence exists, reaches illicit targets,
activates non-trivial structural patterns, and can be extracted without
violating the registered temporal rules.

## Main Findings

| Dataset | Split | Overall coverage | Illicit coverage | Mean tokens | Relay token rate | Cycle token rate |
|---|---|---:|---:|---:|---:|---:|
| Small-LI | Train | 0.9007 | 1.0000 | 18.64 | 0.1558 | 0.0072 |
| Small-LI | Validation | 1.0000 | 1.0000 | 30.86 | 0.0892 | 0.0059 |
| Small-LI | Test | 1.0000 | 1.0000 | 29.42 | 0.0620 | 0.0055 |
| Large-LI | Train | 0.9882 | 1.0000 | 29.69 | 0.0675 | 0.0068 |
| Large-LI | Validation | 0.9999 | 1.0000 | 31.29 | 0.0975 | 0.0093 |
| Large-LI | Test | 0.9999 | 1.0000 | 31.48 | 0.0915 | 0.0111 |

The positive samples are sparse, as expected for LI data: each audited split
contains 16 to 32 illicit targets. The registered minimum of 10 was met, but
class-conditional estimates remain descriptive and should not be treated as
precise population rates.

## Mechanism Evidence

- All six split audits preserved global message-edge `e_id` alignment.
- All six preserved global target-edge mapping through `input_id`.
- No target edge appeared in its own context.
- No future edge or invalid equal-time edge entered the evidence.
- Relay evidence is active on 6.2% to 15.6% of valid context tokens.
- Reciprocal and cycle evidence is rarer but non-zero on both datasets.
- No duplicate target edge was observed within an audited split sample.

These findings rule out the simplest failure modes: empty evidence, evidence
confined to normal targets, dead motif indicators, and temporal leakage.

## Saturation Finding

Validation and test samples are close to the registered 32-token cap:

- Small-LI validation mean: 30.86;
- Large-LI validation mean: 31.29;
- Large-LI test mean: 31.48;
- every split has token-count p90 equal to 32.

Therefore evidence selection is a Phase 1 design variable. A model that simply
consumes the last 32 events may hide informative rare events behind frequent
routine activity. Phase 1 must compare at least recency-only selection with a
role/motif-aware selection using the same evidence encoder.

## Efficiency Finding

| Dataset | Query time across splits | Targets | Approx. throughput | Peak RSS |
|---|---:|---:|---:|---:|
| Small-LI | 210.47 s | 182,272 | 866 targets/s | 2.55 GiB |
| Large-LI | 1123.74 s | 98,304 | 87 targets/s | 42.0 GiB |

The current reference query is too slow for repeated 80-120 epoch training
under the original 256-iteration evaluation schedule. Phase 1 is blocked on an
implementation requirement, not an evidence failure:

1. vectorize recent incident-edge retrieval;
2. prove exact equivalence to the reference query on randomized graphs;
3. benchmark Small-LI and Large-LI throughput;
4. retain the reference implementation as an audit oracle.

## Vectorization Follow-up

The vectorized query passed the continuation gate at commit `0a005d87`.
Large-LI query throughput increased from about 87 to 66,493 targets/s
(approximately 760x), while end-to-end audit time decreased from 1268.04 to
116.31 seconds (10.90x). Exact-equivalence and dynamic-sampling invariant tests
passed.

See `TIER_VECTORIZED_QUERY_BENCHMARK_RESULTS.md` for the complete audit.
Phase 1 evidence-only qualification is now authorized. CrossFusion and
ErrorRouter remain out of scope until evidence-only passes the preregistered
predictive and correction gates.

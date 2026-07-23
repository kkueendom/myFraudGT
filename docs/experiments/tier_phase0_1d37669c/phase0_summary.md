# TIER Phase 0 Evidence Coverage Audit

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Verification Status: EXECUTED
- Git Commit: `1d37669c`
- Sampling Protocol: `dynamic_random`
- Model Training: none

## Coverage

| Dataset | Split | Samples | Positives | Coverage | Positive coverage | Tokens mean | Tokens p90 | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Small-LI | train | 65536 | 29 | 0.9007 | 1.0000 | 18.6417 | 32.0000 | pass |
| Small-LI | val | 65536 | 31 | 1.0000 | 1.0000 | 30.8606 | 32.0000 | pass |
| Small-LI | test | 51200 | 32 | 1.0000 | 1.0000 | 29.4238 | 32.0000 | pass |
| Large-LI | train | 32768 | 16 | 0.9882 | 1.0000 | 29.6927 | 32.0000 | pass |
| Large-LI | val | 32768 | 23 | 0.9999 | 1.0000 | 31.2885 | 32.0000 | pass |
| Large-LI | test | 32768 | 25 | 0.9999 | 1.0000 | 31.4759 | 32.0000 | pass |

## Evidence Activation

| Dataset | Split | Reverse role | Reciprocal | Relay | Cycle |
|---|---|---:|---:|---:|---:|
| Small-LI | train | 0.0071 | 0.0071 | 0.1558 | 0.0072 |
| Small-LI | val | 0.0058 | 0.0058 | 0.0892 | 0.0059 |
| Small-LI | test | 0.0054 | 0.0054 | 0.0620 | 0.0055 |
| Large-LI | train | 0.0065 | 0.0065 | 0.0675 | 0.0068 |
| Large-LI | val | 0.0079 | 0.0079 | 0.0975 | 0.0093 |
| Large-LI | test | 0.0089 | 0.0089 | 0.0915 | 0.0111 |

## Resource Summary

| Dataset | Load (s) | Index (s) | Total (s) | Peak RSS (MB) | Sidecar (bytes) | Decision |
|---|---:|---:|---:|---:|---:|---|
| Small-LI | 2.19 | 0.59 | 231.67 | 2545.3 | 110786373 | pass |
| Large-LI | 47.25 | 14.29 | 1268.04 | 43007.9 | 2817066501 | pass |

A Phase 0 pass only authorizes evidence-only qualification. It is not
evidence of predictive improvement and does not contain an F1 result.

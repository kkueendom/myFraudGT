# Temporal Evidence Reliability Benchmark Results

## Material Passport

- Origin skill: academic-research-suite / experiment-agent validate mode
- Branch: `paper/temporal-evidence-reliability-benchmark`
- Experiment commit: `ed13b729`
- Fixed model: evidence-gate v4 full model, epoch 499
- Datasets: six AML scale/imbalance variants
- Model-training seeds: 42, 43 and 44
- Independent dynamic streams: 93001 and 93002
- Events per stream: 4
- Manifests: 36/36
- Dynamic events: 144/144
- Validation/test loader iterations: 256/256 per event
- Sampling protocol: `dynamic_random`
- Training or tuning: none
- Remote output:
  `/e/yky/FraudGT_cet_results/nested_reliability_formal_ed13b72`
- Local mirror:
  `/Users/kun/FraudGT_experiment_workspace/nested_reliability_formal_ed13b72`

## Decision

`PROCEED_NEGATIVE_BENCHMARK_PAPER`.

All integrity requirements and all four preregistered empirical readiness
conditions passed. The result supports a reliability-limit benchmark, not a
new predictive model or a valid universal confidence procedure.

## Integrity Audit

| Requirement | Result |
|---|---|
| 36 manifests | PASS |
| 144 events | PASS |
| One clean experiment commit | PASS |
| All checkpoints at epoch 499 | PASS |
| All tasks use `dynamic_random` | PASS |
| No fixed panel/evaluation generator/RNG restoration | PASS |
| Full 256 validation and 256 test iterations per event | PASS |
| Validation/test unique-edge rate 1.0 | PASS |
| 48 dataset/stream/event cross-seed hash groups | PASS |
| Target-edge hash mismatch across model seeds | 0 |
| Runtime traceback | 0 |

The target-edge hash audit is important: for each dataset, audit seed and
repeat, all three model-training seeds were evaluated on exactly the same
validation and test target-edge sequence. Model-seed comparisons are therefore
blocked on the dynamic target sample rather than confounded by different
targets.

## Main Result

`V_model`, `V_stream` and `V_event` are method-of-moments variance components.
Sampling share is `(V_stream + V_event) / V_total`.

| Dataset | Test F1 mean +/- sd | Min-max | V_model | V_stream | V_event | Sampling share |
|---|---:|---:|---:|---:|---:|---:|
| Small-LI | 0.44990 +/- 0.02023 | 0.40502-0.47882 | 0.000044 | 0.000000 | 0.000386 | 89.8% |
| Small-HI | 0.75970 +/- 0.01270 | 0.72801-0.78305 | 0.000052 | 0.000051 | 0.000081 | 71.5% |
| Medium-LI | 0.45282 +/- 0.04071 | 0.37864-0.52874 | 0.000105 | 0.000000 | 0.001975 | 94.9% |
| Medium-HI | 0.76427 +/- 0.01550 | 0.74528-0.80101 | 0.000000 | 0.000029 | 0.000237 | 100.0% |
| Large-LI | 0.34409 +/- 0.05147 | 0.22500-0.43312 | 0.000000 | 0.000000 | 0.003197 | 100.0% |
| Large-HI | 0.72076 +/- 0.02115 | 0.67838-0.76236 | 0.000019 | 0.000027 | 0.000411 | 95.8% |

The zero components are nonnegative method-of-moments truncations: the
corresponding mean square did not exceed the lower-level mean square. They do
not prove that the population component is exactly zero.

## Readiness Gate

| Preregistered empirical condition | Result |
|---|---|
| Sampling variance >= model-seed variance on at least 2 datasets | PASS, 6/6 |
| Model-seed ranking changes between streams on at least 1 dataset | PASS, Medium-LI |
| Event delta versus initial A2 changes sign on at least 2 datasets | PASS, 6/6 |
| Median raw-event max inflation >= 0.01 on at least 4 datasets | PASS, 6/6 |

All four conditions passed, although only one was required for benchmark
readiness.

## Model-Seed Ranking

Rankings are based on the four-event mean within each independent stream.

| Dataset | Stream 93001 | Stream 93002 | Pair reversal |
|---|---|---|---|
| Small-LI | 44 > 43 > 42 | 44 > 43 > 42 | No |
| Small-HI | 44 > 42 > 43 | 44 > 42 > 43 | No |
| Medium-LI | 43 > 44 > 42 | 43 > 42 > 44 | **42 versus 44** |
| Medium-HI | 42 > 44 > 43 | 42 > 44 > 43 | No |
| Large-LI | 44 > 43 > 42 | 44 > 43 > 42 | No |
| Large-HI | 43 > 44 > 42 | 43 > 44 > 42 | No |

The result does not imply that seed ranking is generally random. It shows a
concrete case where a ranking conclusion depends on which valid dynamic stream
is observed even when each stream contains four complete events.

## Single-Event Sign Reversal

The historical initial A2 values are used only as fixed diagnostic references.
The v4 model family and epoch-499 checkpoints differ from historical A2, so
these differences are not called sampling bias.

| Dataset | Minimum event Delta | Maximum event Delta | Sign reversal |
|---|---:|---:|---|
| Small-LI | -0.05745 | +0.01635 | Yes |
| Small-HI | -0.05183 | +0.00321 | Yes |
| Medium-LI | -0.13299 | +0.01711 | Yes |
| Medium-HI | -0.03046 | +0.02527 | Yes |
| Large-LI | -0.07608 | +0.13204 | Yes |
| Large-HI | -0.05059 | +0.03339 | Yes |

Thus a single valid dynamic event can support opposite qualitative conclusions
for every dataset in this fixed family. This is stronger than the earlier
`|Delta F1| < 0.005` warning rule: the observed reversals reach several F1
points and, on Large-LI, more than 0.13.

## Raw-Max Inflation

For each fixed checkpoint, raw-event max inflation is:

```text
maximum event F1 - eight-event checkpoint mean F1.
```

The table reports the median over the three model seeds.

| Dataset | Median raw-max inflation |
|---|---:|
| Small-LI | 0.02026 |
| Small-HI | 0.01333 |
| Medium-LI | 0.06300 |
| Medium-HI | 0.03438 |
| Large-LI | 0.08383 |
| Large-HI | 0.03684 |

Reporting the best dynamic test event would therefore add 1.3 to 8.4 F1
points above a checkpoint's repeated-event mean in the median training seed.
This quantifies why raw-best test F1 is not an appropriate formal selection
criterion under this loader.

## Threshold Instability

Validation-derived threshold standard deviation across the 24 events per
dataset was:

| Dataset | Threshold sd |
|---|---:|
| Small-LI | 0.12191 |
| Small-HI | 0.05008 |
| Medium-LI | 0.12448 |
| Medium-HI | 0.07975 |
| Large-LI | 0.18600 |
| Large-HI | 0.10391 |

Threshold instability and target/neighborhood sampling act together. The
current design measures their combined evaluation contribution; it does not
identify a causal share for thresholding alone.

## Combined Evidence

This nested experiment adds a six-dataset, three-training-seed layer to the
earlier reliability evidence:

1. A2 fixed-checkpoint audit: 96 events across six datasets showed event bands
   much larger than 0.005.
2. Multi-model evidence audit: 31 manifests and 220 events showed 9/10
   sensitivity/utility mismatches and 0/10 useful-aligned units.
3. TIER/CET/CPSE: aligned temporal evidence either had negligible safe
   coverage or broke more correct base predictions than it repaired.
4. GTF1C/TREFIC/DGR-F1/GT-psF1: four preregistered reliability methods each
   failed a distinct validity, transfer or power gate.
5. Current nested audit: evaluation sampling accounts for 71.5%-100% of the
   estimated variance and can reverse seed ranks and baseline conclusions.

## Claim Boundary

Supported:

- dynamic fraud-graph evaluation variance can dominate training-seed variance;
- one-event F1 and raw-best reporting can change qualitative conclusions;
- aligned evidence sensitivity is not equivalent to corrective utility;
- rare-positive and temporally dependent regimes create a validity-power
  trade-off for F1 qualification;
- repeated streams, target-edge alignment and explicit abstention are
  necessary reporting components.

Not supported:

- a new fraud detector with stable improvement;
- a universal graph-dependent F1 confidence interval;
- a finite-sample or distribution-free guarantee;
- generalization beyond the AML simulator family;
- causal attribution of all variability to neighborhood sampling alone.

## Paper Consequence

The evidence is now sufficient to draft a transparent negative benchmark
paper around:

> Temporal evidence reliability under dynamic fraud-graph sampling.

The paper's contribution is an empirical reliability benchmark, failure
taxonomy and reproducible evaluation protocol. It must not be framed as a
successful new decoder or statistical certification method.

Machine-readable results:
`TEMPORAL_EVIDENCE_RELIABILITY_BENCHMARK_RESULTS.json`.

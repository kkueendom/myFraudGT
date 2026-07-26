# GT-psF1 Phase 0 Development Results

## Material Passport

- Origin skill: academic-research-suite / experiment-agent validate mode
- Branch: `feature/gt-psf1-graph-dependent-inference`
- Formal commit: `ffbf46c`
- Remote output:
  `/e/yky/FraudGT_cet_results/gt_psf1_phase0_formal_ffbf46c`
- Local mirror:
  `/Users/kun/FraudGT_experiment_workspace/gt_psf1_phase0_formal_ffbf46c`
- GPU tasks: seven nonduplicate dependence regimes on GPUs 0 through 6
- Formal experiments: 512 per scenario/template/effect
- Population-reference experiments: 4096 per scenario/template/effect
- AML train/validation/test loader iterations: 0/0/0
- Status: reproducibly completed; preregistered gate failed

## Decision

`STOP_GT_PSF1`.

GT-psF1 controlled directional type-I error, correctly abstained on dense hub
graphs and selected more neighborhood repeats under higher sampler noise. It
failed the registered coverage, power, cross-template transfer, allocation and
nontrivial novelty gates.

Per the stop rule, do not tune the effect size, dependence strength, graph
bandwidth, target count, replicate grid or diagnostic thresholds on these
scenarios. Do not replace the interval and reuse this screen.

## Gate

| Registered condition | Result |
|---|---|
| Null coverage in [0.94, 0.97] for every in-scope unit | **FAIL**, 4/12 units |
| Directional type-I error <= 0.05 | PASS |
| Coverage regret <= 0.01 versus valid robust comparator | PASS |
| Dense-hub abstention >= 0.90 with no claim | PASS, 1.00 |
| Positive-alternative power >= 0.80 | **FAIL**, 0/12 units |
| Harmful-alternative power >= 0.80 | **FAIL**, 0/12 units |
| Rare/moderate power gap <= 0.15 | **FAIL** |
| Naive inference gives a nontrivial false-claim benchmark | PASS |
| Dyadic coverage loses >= 0.03 under two-hop spillover | **FAIL** |
| Allocation cost regret <= 0.10 | **FAIL** |
| Low/high sampler regimes choose different repeats | PASS, 2 versus 4 |
| Novel value beyond generic graph HAC | **FAIL** |
| Overall | **FAIL** |

## Primary Results

The table reports the GT-psF1 interval. `False +` and `False -` are one-sided
directional claims under the nontrivial equal-F1 null.

| Scenario | Template | Null coverage | False + | False - | Positive power | Harm power |
|---|---|---:|---:|---:|---:|---:|
| IID deterministic | Moderate | 0.9492 | 0.0234 | 0.0273 | 0.6172 | 0.6543 |
| IID deterministic | Rare-sparse | 0.9805 | 0.0117 | 0.0078 | 0.2832 | 0.2344 |
| Endpoint dyadic | Moderate | 0.9375 | 0.0332 | 0.0293 | 0.6172 | 0.6035 |
| Endpoint dyadic | Rare-sparse | 0.9668 | 0.0137 | 0.0195 | 0.2852 | 0.2188 |
| Two-hop spillover | Moderate | 0.9160 | 0.0352 | 0.0488 | 0.5645 | 0.5273 |
| Two-hop spillover | Rare-sparse | 0.9609 | 0.0234 | 0.0156 | 0.2285 | 0.1777 |
| Temporal AR(1) | Moderate | 0.9414 | 0.0234 | 0.0352 | 0.6035 | 0.5566 |
| Temporal AR(1) | Rare-sparse | 0.9668 | 0.0156 | 0.0176 | 0.2188 | 0.1992 |
| Low sampler variance | Moderate | 0.9297 | 0.0410 | 0.0293 | 0.6191 | 0.6445 |
| Low sampler variance | Rare-sparse | 0.9746 | 0.0059 | 0.0195 | 0.2344 | 0.2031 |
| High sampler variance | Moderate | 0.9434 | 0.0234 | 0.0332 | 0.6602 | 0.5957 |
| High sampler variance | Rare-sparse | 0.9727 | 0.0176 | 0.0098 | 0.1914 | 0.1660 |

Coverage was not merely noisy around one boundary:

- two-hop Moderate was anti-conservative at 0.9160;
- IID Rare-sparse was over-conservative at 0.9805;
- low-sampler Moderate was anti-conservative at 0.9297;
- low- and high-sampler Rare-sparse were over-conservative.

This pattern shows that one fixed graph kernel did not adapt adequately across
dependence and rare-positive regimes.

## Power Failure

No in-scope unit reached 0.80 power in either direction.

- Moderate-template power ranged from 0.5273 to 0.6602.
- Rare-sparse power ranged from 0.1660 to 0.2852.
- The rare/moderate gap exceeded the registered 0.15 limit in the two-hop and
  temporal regimes.

The rare-positive failure is central to the intended fraud application. A
method that is valid only after increasing the registered effect or target
budget would not satisfy this development claim.

## Allocation Failure

The two-level decomposition behaved directionally as intended:

- low sampler variance selected `R=2`;
- high sampler variance selected `R=4`.

However, no registered target/repeat cell reached 80% power in either template.
Therefore no empirical cheapest feasible allocation existed, the cost regret
was undefined, and the allocation gate failed in all four units.

The direction of `R` is useful as a diagnostic but is not enough to support a
prospective power-planning method.

## Dense-Graph Diagnostic

The dense hub scenario returned `OUT_OF_SCOPE` for both templates:

- out-of-scope rate: 1.00;
- positive claims: 0;
- harmful claims: 0.

This validates the abstention implementation. It does not repair the failures
on graphs that the method declared in scope.

## Comparator Finding

Naive row-IID inference was strongly anti-conservative in multiple null units,
which confirms that the benchmark problem is real. However, endpoint-dyadic
coverage did not lose the preregistered 0.03 relative to GT-psF1 under two-hop
spillovers:

- Moderate: dyadic 0.9238 versus GT-psF1 0.9160;
- Rare-sparse: dyadic 0.9668 versus GT-psF1 0.9609.

GT-psF1 therefore did not demonstrate the required nontrivial advantage over a
simpler robust comparator. Its current-design interval is also identical to
generic graph HAC by construction. The failed allocation gate leaves no
remaining method contribution beyond that existing estimator.

## Engineering Validation

- 12/12 remote unit tests passed before smoke.
- Seven smoke manifests completed on one clean commit.
- Seven formal manifests completed on the same clean commit.
- Every task used its registered task and reference seed.
- No FraudGT model was trained and no AML validation/test loader was opened.
- Machine-readable audit:
  `GT_PSF1_PHASE0_DEVELOPMENT_RESULTS.json`.

## Research Consequence

The independent graph-dependent paired-F1 route does not supply the missing
strong methods-journal mainline.

The complete project evidence now rules out:

1. post-decoder evidence correction;
2. the current complementary temporal encoder;
3. self-supervised predictive surprise on the existing AML features;
4. four independently registered reliability/certification methods:
   GTF1C, TREFIC, DGR-F1 and GT-psF1.

The remaining defensible choices are:

- write the reproducible negative reliability benchmark; or
- obtain genuinely new external evidence/data and formulate a new predictive
  problem.

Another estimator selected on the completed scenarios would be post hoc and
cannot be treated as confirmatory evidence.

# GT-psF1 Phase 0 Development Plan

## Material Passport

- Origin skill: academic-research-suite / experiment-agent plan mode
- Status: preregistered, not started
- Branch: `feature/gt-psf1-graph-dependent-inference`
- Parent review: `GT_PSF1_THEORY_AND_NOVELTY_REVIEW.md`
- Experiment type: untouched synthetic graph and sampler simulation
- FraudGT/evidence training: none
- AML validation/test loader use: none
- GPU tasks: seven nonduplicate dependence regimes

## Fixed Research Object

Compare two fixed classifiers on the same target transactions under two sources
of evaluation randomness:

1. related target transactions are sampled from a graph-time population;
2. each target can be evaluated under repeated stochastic neighborhoods.

The estimand is the difference between the two F1 scores obtained from
population confusion moments. It is not the mean of batch-level F1 values.

## Fixed Candidate Method

`GT-psF1` uses:

1. six paired confusion contributions per target/neighborhood draw;
2. a multivariate F1-difference gradient;
3. equal target weighting after averaging neighborhood replicates;
4. graph-time HAC covariance over target-level influence values;
5. an explicit within-target sampler variance component;
6. a prospective target/replicate allocation rule.

For balanced replicate count `R`, planning uses:

```text
V(M,R) = a / M + b / (M R)
C(M,R) = M c_target + M R c_neighbor
R_star = sqrt(b c_target / (a c_neighbor)).
```

`R_star` is rounded to the registered feasible grid.

## Fixed Comparators

- `row_iid`: every target/neighborhood draw treated as independent;
- `target_iid`: target means treated as independent;
- `endpoint_dyadic`: direct shared-entity sandwich variance;
- `graph_hac`: graph-time HAC over target means without the explicit
  sampler-allocation output;
- `GT-psF1`: graph-time HAC plus two-level variance reporting, diagnostics and
  allocation.

The candidate cannot claim better coverage than `graph_hac` when both use the
same current-design covariance. Its required added value is valid target
weighting under unequal repeats and accurate prospective allocation.

## Fixed Synthetic Templates

| Template | Targets | Entities | Fraud prevalence | Replicate grid |
|---|---:|---:|---:|---|
| rare-sparse | 384 | 256 | approximately 0.02 | 2, 4, 8, 16 |
| moderate | 512 | 384 | approximately 0.10 | 2, 4, 8, 16 |

No AML confusion table or DGR-F1 scenario parameter is used to select these
templates.

Each scenario evaluates:

- a symmetric nontrivial null: classifiers differ but have equal population
  F1 by construction;
- a practical positive alternative;
- a symmetric practical harmful alternative.

Population truth is computed from a separate registered reference simulation,
not from the same Monte Carlo experiments used for coverage and power.

## Seven Registered Scenario Families

| GPU | Scenario | In-scope target |
|---:|---|---|
| 0 | independent targets, deterministic neighborhoods | reduction to paired-IID behavior |
| 1 | direct shared-endpoint dyadic dependence | endpoint dependence |
| 2 | decaying two-hop graph spillovers | network dependence beyond endpoints |
| 3 | chronological AR(1) common shocks | temporal dependence |
| 4 | graph-time dependence with low sampler variance | unique-target-heavy allocation |
| 5 | graph-time dependence with high sampler variance | repeat-neighborhood allocation |
| 6 | dense hub-dominated graph | out-of-scope diagnostic and abstention |

The seven tasks are scientifically distinct. No duplicate job may be launched
only to occupy a GPU.

## Fixed Monte Carlo Budget

- formal experiments per scenario/template/effect: 512;
- independent population-reference experiments: 4096;
- confidence level: 95%;
- one-sided test level: 0.05;
- desired power: 0.80;
- practical F1 margin: 0.005;
- formal replicate count: 4 unless the scenario is an allocation experiment;
- allocation grid: `R in {2, 4, 8, 16}`;
- task seeds: `88001` through `88007`;
- reference seeds: `98001` through `98007`;
- no parameter changes after the first formal task starts.

## Working-Assumption Diagnostic

The estimator must return `OUT_OF_SCOPE` rather than a confidence claim when
any registered condition holds:

- largest target-dependency degree exceeds `sqrt(M)`;
- one connected dependency component contains more than 50% of targets;
- effective target count is below 30;
- estimated F1 denominator for either classifier is below 10 expected positive
  confusion contributions;
- fewer than two neighborhood draws are available when sampler allocation is
  requested.

The dense-hub scenario is constructed to trigger at least one diagnostic.

## Mathematical Gate

Before formal simulation:

1. exact plug-in F1 difference agrees with direct confusion counts;
2. analytical gradients agree with central finite differences within `1e-6`;
3. row order and classifier order invariants pass;
4. equal-repeat target pooling agrees with pooled confusion moments;
5. unequal-repeat target pooling does not overweight highly repeated targets;
6. within-target variance is zero for deterministic neighborhoods;
7. estimated total variance is nonnegative;
8. the allocation rule chooses the boundary implied by `a=0` or `b=0` and is
   monotone in `b/a`;
9. duplicate target IDs reduce the effective target count rather than
   increasing it;
10. an intentionally dense hub graph returns `OUT_OF_SCOPE`.

## Formal Development Gate

All conditions are preregistered.

### Validity

1. In every in-scope null scenario/template:
   - empirical 95% coverage is between 0.94 and 0.97;
   - one-sided false improvement is at most 0.05;
   - one-sided false harm is at most 0.05.
2. GT-psF1 is not more than 0.01 worse in coverage than the best valid
   comparator in any in-scope unit.
3. The dense-hub scenario returns `OUT_OF_SCOPE` in at least 0.90 of
   experiments and makes no positive or harmful claim when out of scope.

### Power

4. Positive-alternative power is at least 0.80 in every in-scope
   scenario/template.
5. Harmful-alternative power is at least 0.80 in every in-scope
   scenario/template.
6. The rare-sparse template's power is not more than 0.15 below the moderate
   template's power in the graph-spillover and temporal scenarios.

### Nontrivial Value

7. `row_iid` or `target_iid` must exceed 0.08 type-I error in at least one
   graph, temporal or high-sampler null unit.
8. Endpoint-dyadic inference must lose at least 0.03 coverage in the
   two-hop-spillover null relative to GT-psF1 in at least one template.
9. In both sampler-variance scenarios, the registered allocation must achieve
   80% power at a total cost no more than 10% above the empirically cheapest
   allocation on the fixed grid.
10. Low- and high-sampler scenarios must select different replicate counts.

### Novelty Check

11. If GT-psF1's only successful result is identical current-design coverage
    to generic `graph_hac`, without successful allocation or diagnostic value,
    the route fails even if validity and power pass.

## Stop Rule

Any failed formal gate yields `STOP_GT_PSF1`.

After failure, do not tune:

- dependence strengths;
- graph bandwidth;
- temporal bandwidth;
- effect sizes;
- target counts;
- replicate grid;
- diagnostic thresholds;
- confidence level;
- Monte Carlo seeds.

Do not reuse the scenarios to select a replacement interval or bootstrap.

Passing Phase 0 authorizes theorem polishing and a separately preregistered
external benchmark. It does not authorize AML test access, a predictive-model
claim or a six-dataset FraudGT main table.

## GPU Launch Rule

Formal GPU jobs are authorized only after:

1. implementation and invariant tests are committed;
2. a four-experiment-per-unit smoke run succeeds;
3. every manifest records the same clean commit and this plan's specification;
4. the remote worktree is clean.

When authorized, use all seven available GPUs, one registered scenario per GPU.


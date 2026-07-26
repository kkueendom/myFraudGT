# Methods-Journal Mainline Completion Audit

## Material Passport

- Origin Skill: academic-research-suite / methodology review and integrity
  audit
- Audit date: 2026-07-26
- Target: a FraudGT mainline capable of supporting a strong
  methods-oriented journal
- Authoritative objective:
  `/Users/kun/.codex/attachments/0438ff2d-e89b-48df-97e3-4bd02864fcac/goal-objective.md`
- Current branch: `paper/temporal-evidence-reliability-benchmark`
- Current nested benchmark result commit: `eb04985`
- Current budget-convergence preregistration commit: `1344ef8`
- Audit verdict: target not achieved

## Executive Verdict

The project has completed the registered feasibility and reliability
experiments with strong protocol discipline. It has not produced a method
that passes the required two-scale gate. Therefore:

> The current repository does not yet support a strong predictive-method or
> statistical-method journal claim.

This is not an absence-of-testing conclusion. TIER OOF, CET-FraudGT, CPSE,
GTF1C, TREFIC, DGR-F1 and GT-psF1 each failed a different preregistered
requirement.
The failures collectively rule out immediate continuation through decoder
tuning, another history encoder with the same data, or another confidence
interval selected on the same scenarios.

## Requirement-by-Requirement Audit

### 1. Fixed Dynamic-Random Protocol

| Requirement | Evidence | Status |
|---|---|---|
| Dynamic train/val/test sampling | Runner manifests and protocol tests | Achieved |
| `shuffle=True` | Loader audits | Achieved |
| No fixed target panel | Protocol manifests | Achieved |
| No independent evaluation generator | Protocol manifests | Achieved |
| No sampler RNG restoration | Protocol manifests | Achieved |
| Initial A2 only, no Fixed-panel baseline | All primary result documents | Achieved |
| Val-selected primary; Raw-best secondary | CET and historical result tables | Achieved |
| No cross-metric comparison | Result aggregators | Achieved |

The dynamic protocol itself is not a blocker.

### 2. Phase A: Train-Only OOF Utility Qualification

Authoritative evidence:
`TIER_PHASE2B_OOF_UTILITY_RESULTS.md`.

| Dataset | Routed changes | Corrected/broken | Paired F1 Delta | Decision |
|---|---:|---:|---:|---|
| Small-LI | 9 | 3/6 | -0.01049 | Fail |
| Large-LI | 2 | 2/0 | +0.00189 | Fail |

Fold exclusion, edge alignment, zero val/test access and normal/shuffled/off
audits passed. The scientific gate failed:

- fewer than 50 interventions;
- nonpositive Small-LI utility;
- no policy gate on either scale.

Status: **completed but contradicted**. It does not authorize CrossFusion,
validation or test evaluation.

### 3. Phase B: Complementary Temporal Encoder

Authoritative evidence:
`CET_PHASEB_RESULTS_AND_STOP_DECISION.md`.

| Dataset | Variant | Val-selected F1 | Delta vs A2 |
|---|---|---:|---:|
| Small-LI | Encoder-only | 0.05290 | -0.40957 |
| Small-LI | Fusion | 0.28704 | -0.17543 |
| Large-LI | Encoder-only | 0.09742 | -0.20366 |
| Large-LI | Fusion | 0.31868 | +0.01760 |

The registered two-scale gate failed because:

- average fusion Delta was -0.07892;
- Small-LI normal-shuffled was only +0.00221;
- Large-LI changed only 17 predictions;
- corrected was below broken on both scales;
- the fusion mechanism was not useful across scales.

Status: **completed but contradicted**. Phase C expansion and Phase D formal
training are prohibited by the original stop rule.

### 4. Independent Self-Supervised Predictive Surprise

Authoritative evidence:
`CPSE_PHASE0B_OOF_RESULTS.md`.

CPSE already implements the apparently untried alternative:

- fraud-label-free causal next-transaction prediction;
- source/destination/global history pooling;
- amount and interarrival surprise;
- currency and payment negative log likelihood;
- normal/shuffled/off OOF evidence;
- six held-out train folds;
- zero validation/test access.

Results:

| Dataset | Changed | Corrected/broken | Summed/mean paired F1 result | Decision |
|---|---:|---:|---:|---|
| Small-LI | 0 | 0/0 | 0.0000 | Fail |
| Large-LI | 62 | 59/3 | negative, one fold 0.10 to 0.00 | Fail |

Large-LI normal and shuffled utility AUPRC differed by only 0.00274. The three
broken true positives outweighed 59 removed false positives because the base
F1 sensitivity ratio was very small.

Status: **completed but contradicted**. Reimplementing predictive surprise
under a new name would duplicate CPSE.

### 5. Reliability and Identifiability Methods

#### A2 and Multi-Model Reliability

- Six fixed A2 checkpoints: 96 dynamic events.
- Three evidence families: 31 manifests and 220 dynamic events.
- Sensitivity/utility mismatch: 9/10 model/dataset units.
- Useful aligned units: 0/10.

These audits establish the problem but not a successful method.

#### GTF1C

- 21,504 formal dataset/fold trials.
- Safe on IID/entity settings.
- Large-LI temporal-positive false qualification: 0.0944.
- Large-LI temporal-positive practical failure: 0.1068.

Decision: `STOP_GTF1C`.

#### TREFIC

- 2,688 development trials.
- False qualification controlled in all 14 units.
- Small-LI positive power: 4/4 regimes.
- Large-LI positive power: 0/4 regimes.

Decision: `STOP_TREFIC`.

#### DGR-F1

- Seven scenarios, two templates, 512 Monte Carlo experiments per unit.
- Stable positive power: 1.00 on both templates.
- Unstable-scenario abstention: 1.00.
- Large-LI null simultaneous mean coverage: 0.8672, below 0.94.
- Large-LI stable-harm power: 0.2949, below 0.80.

Decision: `STOP_DGR_F1`.

#### GT-psF1

- Seven untouched dependence regimes and two synthetic templates.
- 512 formal experiments and 4096 population-reference experiments per
  scenario/template/effect.
- Null coverage gate: 4/12 in-scope units.
- Positive and harmful power gate: 0/12 units in each direction.
- Dense-hub abstention: 1.00.
- Low/high sampler regimes selected 2 versus 4 repeats, but no registered
  allocation cell reached 0.80 power.
- No preregistered two-hop coverage advantage over endpoint-dyadic inference.

Decision: `STOP_GT_PSF1`.

Status: **the reliability question is established, but every proposed new
method failed at least one registered validity or power gate**.

### 6. Innovation Boundary

Completed reviews correctly prohibit claims of first use of:

- GNN, attention, motif or graph transformer;
- gate, residual, prototype or mixture-of-experts;
- temporal transaction encoding;
- counterfactual shuffle/off evaluation;
- OOF error prediction;
- conformal or generic risk control;
- confidence sequences or replication probability;
- cost-sensitive F1 optimization.

A new 2026 result further narrows the statistical claim:

- [Hsu, Liu and Shyr, comparative F1/F-beta inference](https://www.medrxiv.org/content/10.64898/2026.07.15.26358166v1.full)
  derives paired F1 inference and power calculations for correlated
  classifiers under independent instances.
- [Hsu et al., Statistics in Medicine 2026](https://onlinelibrary.wiley.com/doi/10.1002/sim.70557)
  provides single/comparative F-beta inference and sample-size planning.
- [Lam et al., F1 confidence intervals](https://arxiv.org/abs/2309.14621)
  compares analytical F1 interval constructions.
- [Malaviya et al., F1 coreset lower bounds](https://arxiv.org/abs/2312.09885)
  proves sampling lower bounds for non-decomposable classification measures.

Consequently, ordinary paired-F1 inference, power analysis, or a new interval
cannot be claimed as the contribution.

### 7. Six-Dataset Formal Experiment

The objective requires:

- six datasets;
- three model seeds;
- mean and standard deviation;
- module ablations;
- counterfactual evidence controls;
- calibration, complexity and failure analysis.

These experiments were intentionally not started because no candidate passed
the Small-LI/Large-LI Phase B gate.

Status: **not achieved and not authorized**.

### 8. Engineering Discipline

| Requirement | Status |
|---|---|
| Explain scope before edits | Achieved |
| Independent commit before experiment | Achieved |
| Unique output directories | Achieved |
| Commit/config/seed/protocol manifests | Achieved |
| OOF overlap and label-leakage tests | Achieved |
| Counterfactual and edge-alignment tests | Achieved |
| Dynamic protocol tests | Achieved |
| Distinct GPU tasks, no duplicate occupancy | Achieved |
| Low-frequency monitoring | Achieved |

Engineering quality is not the reason the mainline failed.

## Completion Matrix

| Objective deliverable | Evidence status | Completion |
|---|---|---|
| Valid two-scale evidence signal | Directly contradicted | No |
| Useful representation-level fusion | Directly contradicted | No |
| Stable improvement over A2 | Missing because candidate gate failed | No |
| Cross-scale mechanism | Directly contradicted | No |
| Successful reliability method | Directly contradicted, including GT-psF1 | No |
| Six-dataset three-seed main table | Not authorized | No |
| Coherent module ablation | Not authorized | No |
| Strong methods-journal novelty | Missing | No |
| Reproducible negative benchmark | Strong evidence exists | Yes |

## Current Publishable Scope

The completed nested dynamic reliability benchmark now supports:

> a reproducible negative benchmark showing that aligned temporal evidence,
> local corrective counts and one-run F1 gains do not imply stable corrective
> utility under dynamic fraud-graph sampling.

Additional six-dataset evidence is recorded in
`TEMPORAL_EVIDENCE_RELIABILITY_BENCHMARK_RESULTS.md`:

- 36 fixed-checkpoint tasks and 144 complete dynamic events;
- target-edge hashes aligned across three training seeds;
- dynamic sampling accounts for 71.5%-100% of estimated variance;
- one valid stream reverses a model-seed ranking on Medium-LI;
- event-level differences versus historical A2 cross zero on all six
  datasets;
- median raw-event maximum inflation is 0.0133-0.0838.

This is now suitable for drafting a benchmark, empirical-study or
negative-results paper. It is not a successful predictive-method paper, and a
strong methods-journal submission still requires external validation beyond
the six AML simulator variants.

## Only Scientifically Distinct Next Paths

### Path A: Negative Reliability Benchmark

The preregistered benchmark-readiness gate has passed. No new AML model
training is required. Consolidate:

- A2 dynamic stability;
- CET/TIER/COSTAR sensitivity-utility mismatch;
- CPSE, GTF1C, TREFIC and DGR-F1 failure boundaries;
- GT-psF1 validity and power boundary;
- nested model-seed/stream/event variance decomposition;
- a public reproducibility package and benchmark protocol.

This is the active paper path under the objective's strict stopping rule.

### Path B: Independent Graph-Dependent F1 Inference

The registered GT-psF1 development screen completed and failed coverage, power,
allocation and nontrivial-value gates. A replacement interval cannot be
selected on the same scenarios. Any future return to this path requires:

1. a dependency model that extends paired-F1 inference beyond independent
   instances to shared entities, chronological overlap and stochastic
   neighborhoods;
2. a derivation or proof supplied independently of the failed DGR-F1 and
   GT-psF1 scenarios;
3. new untouched simulations and at least one external graph benchmark;
4. prospective power calculations before fixing stream budgets;
5. comparison with psF1/psF1pair, cluster bootstrap and network bootstrap.

The existing AML, DGR-F1 and GT-psF1 scenarios may be used only as final
external applications, not to select the new estimator.

### Path C: New External Evidence

The current AML feature space has exhausted flow-role, local history,
subgraph, support and self-supervised predictive-surprise views. A genuinely
new predictive mainline requires information absent from the present dataset,
such as:

- device/IP linkage;
- account ownership or KYC;
- merchant/geolocation context;
- investigation alerts or case histories;
- an additional independent fraud graph dataset.

Without new information, another evidence encoder is unlikely to be
scientifically distinct from CET or CPSE.

## GPU Decision

No further predictive-model or failed-interval rerun is justified:

- TREFIC formal expansion is prohibited by its failed power gate.
- DGR-F1 reruns or interval replacement are prohibited by its stop rule.
- GT-psF1 reruns, effect enlargement or kernel replacement are prohibited by
  its stop rule.
- CET/CPSE additional seeds are prohibited by failed two-scale qualification.
- Phase C/D six-dataset training is not authorized.

One new non-training GPU experiment is now justified and preregistered in
`DYNAMIC_EVALUATION_BUDGET_CONVERGENCE_PLAN.md`. It extracts seven nested
evaluation budgets from each full dynamic trajectory across six datasets,
three fixed model seeds and two streams. This directly addresses the paper's
missing actionable recommendation: how much dynamic evaluation is needed
before F1 approaches the original 256-batch result. The 36 tasks are distinct
and may be distributed over all seven GPUs.

External GPU training remains blocked until a real dataset, official
preprocessing and access terms are verified. The comparison and resource plan
are recorded in `EXTERNAL_RELIABILITY_VALIDATION_FEASIBILITY.md`.

## Final Audit Verdict

`STRONG_METHODS_MAINLINE_NOT_ACHIEVED`.

The active target cannot be honestly marked complete. The next decision is
not a hyperparameter choice; it is whether to:

1. approve the negative benchmark paper scope;
2. supply/obtain independent graph data or external evidence;
3. undertake a new statistical-theory project extending paired-F1 inference
   under graph/time dependence.

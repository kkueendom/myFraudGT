# Graph-Time Paired Risk Method: Preregistered Feasibility Plan

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Status: preregistered, not started
- Working method name: GTPRC
- Sampling protocol for AML experiments: `dynamic_random`
- Validation/test labels allowed in Phase 0: no
- Full FraudGT training allowed in Phase 0: no
- Formal baseline: registered historical initial A2

## Research Question

Can an auxiliary causal evidence view alter a frozen FraudGT prediction with
nonzero corrective coverage while controlling harm under transaction-level
graph and temporal dependence?

The experiment has two independent gates. Passing only one is insufficient.

## Method Object

For target transaction \(i\), let \(b_i\) be the frozen A2 decision, \(e_i\)
an evidence proposal, and \(a_i^d(\lambda)\) an intervention policy for
direction \(d \in \{\mathrm{add},\mathrm{remove}\}\).

The paired break loss and correction utility are:

\[
L_{\mathrm{break},i}^d(\lambda)
=
\mathbb{1}[a_i^d(\lambda)=1]
\mathbb{1}[b_i=y_i]
\mathbb{1}[e_i\ne y_i],
\]

\[
U_{\mathrm{correct},i}^d(\lambda)
=
\mathbb{1}[a_i^d(\lambda)=1]
\mathbb{1}[b_i\ne y_i]
\mathbb{1}[e_i=y_i].
\]

Rows are linked in a dependency graph when they share a source or destination
entity, refer to the same unique target transaction, or fall within a
preregistered temporal dependence window. GTPRC selects the least restrictive
policy whose dependency-adjusted upper confidence bound on break risk is at
most \(\alpha=0.40\), then maximizes corrective coverage. The registered
confidence failure probability is \(\delta=0.05\).

Add and remove policies are fitted and calibrated separately. Validation and
test labels never define policies, thresholds, blocks, or confidence bounds.

No finite-sample guarantee may be claimed until the dependency assumption and
bound are proved. Phase 0 measures empirical validity first.

## Phase 0A: Controlled Dependence Validation

### Purpose

Test whether GTPRC controls harm when ordinary row-IID calibration is
anti-conservative and determine whether it retains useful coverage.

### Seven Scientifically Distinct GPU Tasks

| GPU | Dependence regime |
|---:|---|
| 0 | IID reference |
| 1 | shared-entity clusters |
| 2 | temporal autocorrelation |
| 3 | combined entity and temporal dependence |
| 4 | fraud-prevalence drift |
| 5 | evidence-alignment drift |
| 6 | rare-positive and duplicate-exposure stress |

Each task runs the same preregistered policy grid over independently generated
replicates. These are distinct dependence regimes, not duplicate seeds added
to occupy GPUs.

### Compared Methods

1. row-IID empirical or Wilson upper bound;
2. time-block-only bound;
3. entity-cluster-only bound;
4. GTPRC graph-time dependency bound;
5. oracle policy using population risk, for coverage reference only.

### Primary Metrics

- empirical probability that true break risk exceeds \(\alpha\);
- intervention coverage;
- corrected-minus-broken;
- coverage as a fraction of oracle coverage;
- false qualification rate when evidence is shuffled or harmful;
- abstention rate when no policy is safe.

### Phase 0A Gate

GTPRC advances only if:

- violation probability is at most \(\delta + 0.02=0.07\) in every dependent
  regime;
- false qualification is at most 0.05 in shuffled and harmful controls;
- median coverage is at least 60% of oracle coverage in at least five of seven
  regimes;
- it is not uniformly more conservative than both single-axis block methods;
- the result is stable across preregistered replicate batches.

Failure stops the method or requires a mathematical redesign before AML use.

## Phase 0B: Independent Evidence-Source Qualification

### Evidence Source

Use a Causal Predictive Surprise Encoder (CPSE), not an evidence-only fraud
classifier and not a CET fusion head.

For each target transaction, CPSE summarizes only prior endpoint histories and
predicts target-event properties:

- log amount and amount-change bucket;
- time since prior source and destination activity;
- payment-format and currency distributions;
- counterparty novelty;
- source and destination activity intensity.

The evidence representation is the predictive residual or surprise of the
observed target relative to both endpoint histories. Fraud labels are not used
to train CPSE. Target causality is enforced by timestamp and edge-ID order.

This source is independent in purpose from TIER:

- TIER directly predicts fraud from hand-organized history tokens;
- CPSE learns normal causal behavior through self-supervised next-event
  prediction;
- only the detached surprise representation enters the train-only OOF utility
  qualification.

### Six OOF Tasks

| GPU | Dataset | Held-out fold |
|---:|---|---:|
| 0 | Small-LI | 0 |
| 1 | Small-LI | 1 |
| 2 | Small-LI | 2 |
| 3 | Large-LI | 0 |
| 4 | Large-LI | 1 |
| 5 | Large-LI | 2 |

GPU 6 runs Phase 0A or deterministic aggregation while the six OOF tasks are
active. It must not run a duplicate OOF fold.

### Fold Contract

- train targets are assigned to exactly one held-out fold by global edge ID;
- held-out fraud labels do not enter CPSE or utility-probe training;
- held-out transactions may remain unlabeled causal context for later events;
- validation/test loaders are never instantiated;
- source/destination IDs and timestamps are stored for dependency auditing;
- normal, shuffled, and off evidence share identical target rows;
- all outputs record `sampling_protocol=dynamic_random`.

### Phase 0B Gate

Both Small-LI and Large-LI must satisfy:

- at least 50 held-out interventions in total;
- corrected greater than broken;
- corrected/broken at least 1.5;
- positive summed held-out paired F1 delta;
- normal paired F1 exceeds shuffled by at least 0.01 or ten net corrections;
- at least two of three held-out folds have positive net correction;
- CPSE normal-shuffled utility AUPRC exceeds prevalence by at least 0.05;
- GTPRC selects a nonempty policy with its registered harm bound satisfied.

If either dataset fails, stop CPSE and do not tune thresholds, add a decoder,
or open validation/test.

## Advancement Logic

| Phase 0A | Phase 0B | Decision |
|---|---|---|
| Fail | any | redesign dependence method; no AML claim |
| Pass | Fail | retain a controlled-method result only; stop predictive claim |
| Fail | Pass | evidence is promising but risk method is invalid; do not open test |
| Pass | Pass | lock GTPRC and CPSE, then begin two-scale dynamic validation |

## Later Phases

### Phase 1: Two-Scale Dynamic Validation

- Small-LI seed 42 and Large-LI seed 44;
- frozen A2 plus locked CPSE/GTPRC;
- normal, shuffled, off, row-IID and single-axis block controls;
- Val-selected Test F1 primary, Raw-best secondary;
- repeated dynamic streams for stability;
- no method retuning.

Advance only if both datasets have positive Val-selected delta greater than
their measured sampling uncertainty, nonzero safe coverage, and positive
corrected-minus-broken.

### Phase 2: Six-Dataset Confirmation

Only after Phase 1 passes:

- six AML datasets;
- three model seeds;
- mean and standard deviation;
- paired same-batch mechanism audit;
- method ablations, risk-coverage curves, complexity, and failure analysis.

## Required Artifacts

Every task must record:

- dataset or simulation regime;
- variant and seed;
- Git commit and configuration;
- checkpoint and source OOF folds;
- dependency-block definition;
- normal/shuffled/off/base metrics;
- changed, corrected, broken, coverage, and harm upper bound;
- Val-selected and Raw-best F1 when validation/test is allowed;
- delta against the initial A2 using the same metric;
- loader audit and `sampling_protocol=dynamic_random`;
- runtime, peak GPU memory, and failure reason.

## Engineering Rules

- Commit each code change before any execution.
- Smoke test every distinct task before formal launch.
- Preserve existing output directories.
- Use all free GPUs only for distinct preregistered tasks.
- Poll infrequently and keep logs concise.
- Stop immediately when a hard gate fails.

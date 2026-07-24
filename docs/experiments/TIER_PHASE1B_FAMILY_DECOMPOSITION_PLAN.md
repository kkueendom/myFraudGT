# TIER Phase 1b Evidence Family Decomposition

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Stage: preregistered mechanism diagnosis
- Branch: `feature/tier-independent-evidence-routing`
- Experiment configuration commit: `26dea79`
- Parent result root: `formal_204b3075`
- Sampling protocol: `dynamic_random`
- Primary metric: Val-selected Test F1
- Fixed-panel A2: excluded

## Why This Experiment Exists

The Phase 1 all-evidence classifier established that raw historical
transaction evidence is counterfactually active: normal evidence strongly
outperforms shuffled and off evidence on both datasets. It did not satisfy the
safety criterion for replacing A2 predictions:

| Dataset | Selection | Corrected | Broken | Corrected / broken | Decision |
|---|---|---:|---:|---:|---|
| Small-LI | recent | 40 | 121 | 0.331 | fail |
| Small-LI | role_motif | 36 | 81 | 0.444 | fail |
| Large-LI | role_motif | 41 | 46 | 0.891 | fail |

The last Large-LI `recent` result remains part of Phase 1 and must be audited
before Phase 1b is launched. Phase 1b does not relax any qualification
criterion and does not implement CrossFusion or ErrorRouter. It separates the
evidence channels so that later design decisions are driven by the source of
corrections and breaks, rather than by a headline F1 search.

## Hypotheses

1. A subset of evidence channels will retain a material normal-versus-shuffled
   gap, establishing that its signal is carried by the assigned historical
   transactions.
2. The same subset may improve the corrected-to-broken ratio by removing
   channels that produce plausible but unsafe false-positive evidence.
3. A family that is predictive but still breaks more A2-correct samples than
   it corrects is evidence for a future conditional intervention problem, not
   evidence that it should enter the final model.

## Controlled Factor

All Phase 1b runs preserve the evidence encoder architecture, optimizer,
training budget, token cap, selection pool, A2 checkpoint, dynamic sampling
protocol, thresholding procedure, and paired same-batch diagnostic from Phase
1. The only changed factor is the input-channel family:

| Family | Enabled token channels | Disabled token channels |
|---|---|---|
| `structure` | endpoint roles, reciprocal/relay/cycle motifs | timestamp, amount, currency, payment format |
| `temporal` | absolute timestamp and target-context time difference | amount, categorical attributes, roles, motifs |
| `flow_role` | amount, currency, payment format, endpoint roles | timestamp, time difference, motifs |

The controlled rerun also applies the same family boundary to support:

- `structure`: context count, endpoint-role coverage, and motif counts;
- `temporal`: context count and time span;
- `flow_role`: context count and endpoint-role coverage;
- `all`: all six support values.

The first executed family matrix masked token channels but not support
channels. Its family attribution is therefore superseded; its broad
counterfactual and safety findings remain descriptive.

The target transaction's raw attributes are intentionally unchanged in every
family. The normal, shuffled, and off paths remain same-batch
counterfactuals. `off` zeros all contextual evidence and support, while the
target path stays fixed.

## Matrix

Each family contains these four tasks:

| Dataset | Seed | Context selection | Batch size | Validation/test iterations |
|---|---:|---|---:|---:|
| AML Small-LI | 42 | `recent` | 2048 | 256 |
| AML Small-LI | 42 | `role_motif` | 2048 | 256 |
| AML Large-LI | 44 | `recent` | 1024 | 256 |
| AML Large-LI | 44 | `role_motif` | 1024 | 256 |

There are 12 tasks in total. They use 80 maximum epochs, validation every two
epochs, and the existing 12-evaluation early-stop rule. The reproducible
specifications are:

- `run/tier_phase1b_structure_spec.json`
- `run/tier_phase1b_temporal_spec.json`
- `run/tier_phase1b_flow_role_spec.json`

`run/tier_phase1b_family_decomposition_7gpu.sh` places all seven available
GPUs into the first wave and queues the remaining five tasks on those workers.
It writes a worker PID list and one low-volume stdout log per GPU; the
per-task trajectory and manifest remain the authoritative result records.

## Required Per-Task Record

Every manifest must contain:

- dataset, seed, family, context selection, Git commit, config, and checkpoint;
- `sampling_protocol=dynamic_random`;
- Val-selected Test F1 and Raw-best Test F1;
- like-for-like deltas against the initial A2 table;
- normal, shuffled, and off metrics in one selected test event;
- sampled-instance and unique-edge A2 pairing statistics;
- evidence coverage, token counts, role/motif activation, and qualification
  checks.

The audit rejects a manifest if it uses a fixed panel, a mismatched historical
A2 metric, or missing counterfactual/paired diagnostics.

## Decision Rules

For each task, retain the existing Phase 1 qualification gate unchanged:

1. normal minus shuffled Test F1 >= `0.010`;
2. normal minus off Test F1 >= `0.005`;
3. correction rate on A2 errors >= `10%`;
4. corrected/broken >= `1.5` and corrected minus broken > `0`;
5. changed predictions >= `max(50, 10% of A2 errors)`;
6. overall and per-class coverage >= `0.80`.

Family-level interpretation is fixed before launch:

| Outcome | Interpretation | Next action |
|---|---|---|
| A family passes on Small-LI and Large-LI | Cross-scale safe evidence candidate | Implement frozen-A2 CrossFusion and train-only OOF utility router |
| Passes only on Large-LI or only on Small-LI | Scale-dependent evidence | Run source-level error-subset analysis; do not claim a universal family |
| Counterfactual gap is material but corrected/broken fails | Informative but unsafe evidence | Diagnose error strata and route supervision before any fusion implementation |
| Shuffled/off gap fails | No demonstrated context-dependent signal | Remove the family from the TIER candidate set |

No family advances because of raw-best F1 alone. An absolute delta under
`0.005` against initial A2 is marked as potentially within dynamic-sampling
variation. Phase 1b is not a test of full-model F1, and no Phase 2/3 training
starts until the evidence gate has been evaluated across both scales.

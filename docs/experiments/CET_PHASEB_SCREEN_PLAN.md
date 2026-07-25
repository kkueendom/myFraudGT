# CET-FraudGT Phase B Two-Scale Screen

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Status: preregistered, not yet executed
- Branch: `feature/cet-fraudgt-complementary-encoder`
- Baseline: initial A2, not Fixed-panel A2
- Sampling protocol: `dynamic_random`
- Primary metric: Val-selected Test F1
- Supplementary metric: Raw-best Test F1

## Hypothesis

A representation learned from causal transaction history can complement a
frozen FraudGT representation when it is fused before classification and
trained to remain sensitive to aligned history.

## Model

For target transaction \(e_i=(u_i,v_i,t_i)\), the read-only history index
returns only events satisfying:

\[
t_j < t_i
\quad\text{or}\quad
(t_j=t_i \land j<i).
\]

The evidence encoder independently embeds amount, currency, payment format,
relative time, endpoint role and reciprocal/relay/cycle flags. It produces
separate source-history, destination-history and global-history summaries:

\[
h_i^{evi} =
E_\theta(H_i^{src},H_i^{dst},H_i^{global},s_i,x_i).
\]

The frozen A2 encoder supplies:

\[
h_i^{base} =
[h_{u_i}\Vert e'_{u_iv_i}\Vert h_{v_i}].
\]

The full variant performs cross-attention before classification:

\[
c_i=\operatorname{MHA}(W_bh_i^{base},H_i^{evi},H_i^{evi}),
\]

\[
\tilde h_i^{base}=
\operatorname{LN}(W_bh_i^{base}+W_cc_i),
\]

\[
z_i^{fusion}
=C_\phi[
\tilde h_i^{base}\Vert h_i^{evi}
\Vert|\tilde h_i^{base}-h_i^{evi}|
\Vert\tilde h_i^{base}\odot h_i^{evi}
].
\]

This is a full candidate prediction at the representation layer. It is not a
scalar residual added after \(z_{base}\).

## Training Objective

\[
\mathcal L =
\mathcal L_{fraud}
+\lambda_{cf}\mathcal L_{counterfactual}
+\lambda_{comp}\mathcal L_{OOF-error}
+\lambda_{drop}\mathcal L_{off}
+\lambda_{distill}\mathcal L_{base-retention}
+\lambda_{aux}\mathcal L_{evidence}.
\]

- `fraud`: weighted fraud classification on aligned history;
- `counterfactual`: aligned history must have lower per-sample loss than the
  average shuffled/off path by a margin;
- `OOF-error`: extra classification pressure only on train edges that an OOF
  A2 teacher misclassified;
- `off`: the fusion classifier remains usable without history;
- `base-retention`: off-history fusion matches the frozen A2 probability;
- `evidence`: evidence representation retains direct prediction signal.

Validation and test labels never define OOF-error targets.

## Registered Tasks

| Task | GPU | Dataset | Variant | Seed |
|---|---:|---|---|---:|
| 0 | 0 | Small-LI | fusion | 42 |
| 1 | 1 | Large-LI | fusion | 44 |
| 2 | 2 | Small-LI | encoder-only | 42 |
| 3 | 3 | Large-LI | encoder-only | 44 |

The four primary tasks are scientifically distinct.

At the user's request to use all currently available GPUs, three Small-LI
diagnostic ablations run concurrently on GPUs 4-6:

| GPU | Label | Removed objective |
|---:|---|---|
| 4 | `no_oof` | OOF base-error emphasis |
| 5 | `no_counterfactual` | normal-versus-shuffled/off ranking |
| 6 | `no_base_retention` | off-history classification and A2 distillation |

These diagnostics are not substitutes for the two-scale advancement gate.
They are interpreted only after the registered full fusion task, and their
purpose is to diagnose a pass or failure without running duplicate seeds.

## Fixed Runtime

- maximum 80 epochs;
- evaluate every 4 epochs;
- earliest stop after 20 epochs;
- stop after five evaluation events without a material validation gain;
- 48 most recent causal history events;
- one 64-dimensional history encoder layer;
- frozen epoch-499 A2 checkpoint;
- OOF supervision from Phase 2b folds only.

Before formal execution, all four tasks run a four-epoch smoke test with two
train and evaluation steps per epoch in isolated output directories. Four
epochs are required to trigger the registered evaluation period.

## Dynamic Sampling Contract

- train, val and test use the original dynamic loader;
- `shuffle=True`;
- `val.fixed_target_panel=False`;
- no fixed target edge panel;
- no independent evaluation generator;
- no RNG state restoration;
- original batch size and `val.iter_per_epoch=256`;
- normal, shuffled, off and frozen-base scores are computed on the same
  dynamic batches at every evaluation event.

## Required Outputs

Every task records dataset, variant, seed, commit, config, checkpoint, A2
checkpoint, OOF source, loader audit, epochs, actual val/test iterations,
Val-selected and Raw-best Test F1, both same-metric deltas, normal/shuffled/off
results, coverage, changed/corrected/broken predictions, representation norms,
query time, total time and `sampling_protocol=dynamic_random`.

## Advancement Gate

Only the two fusion tasks determine advancement. Both datasets must satisfy:

- Small-LI Val-selected Test F1 at least `0.46747`;
- Large-LI Val-selected Test F1 at least `0.30608`;
- average delta versus initial A2 greater than zero;
- normal minus shuffled Test F1 at least `0.010`;
- normal minus off Test F1 at least `0.005`;
- at least 50 changed predictions;
- corrected predictions greater than broken predictions;
- nonzero, input-dependent fusion gain norm.

An absolute delta below `0.005` is labelled likely dynamic-sampling variation.
If either dataset fails, do not tune thresholds or expand to six datasets.

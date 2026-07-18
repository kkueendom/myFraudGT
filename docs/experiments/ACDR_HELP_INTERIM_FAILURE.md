# ACDR-Help Interim Failure Conclusion

## Material Passport

- Branch: `feature/acdr-marginal-help-critic`
- Method revision: `faf972865181ad6e97a318052011617dc079e94a`
- Decision date: 2026-07-19
- Primary metric: Test F1 at the checkpoint selected by validation F1
- Decision: **FAIL; stop the paired screen and do not expand to later datasets**
- Scope: mechanism-level interim rejection, not a completed 500-epoch result

## 1. Authoritative Interim Evidence

The matched audit through epoch 165 gives:

| Dataset | Val-selected Test F1 delta vs A2 | Raw-best Test F1 delta vs A2 |
|---|---:|---:|
| Small-LI | **-0.01660** | **-0.01594** |
| Large-LI | **+0.03971** | **+0.03971** |

The latest validation diagnostics before termination show:

| Dataset | Mean help probability | Mean dose | Dose std | Interpretation |
|---|---:|---:|---:|---|
| Large-LI | 0.9966 | 1.9931 | 0.0379 | Almost every sample is routed to the upper bound 2 |
| Small-LI | 0.9992 | 1.9984 | 0.0192 | Almost every sample is routed to the upper bound 2 |

The Large-LI task was stopped around epoch 171 and the Small-LI task around
epoch 268. Neither task is a completed 500-epoch run. Continuing them would not
repair the already observed mechanism collapse or the failed Small-LI gate.

## 2. What Worked

The A2-centered target recovered the absolute direction that CAMPR's centered
target removed. Large-LI improved materially under a stronger prototype
residual, so the result supports this narrower finding:

> On Large-LI, increasing the global A2 prototype-residual scale can help.

This is useful diagnostic evidence about residual strength, but it does not
validate sample-adaptive routing.

## 3. Why the Method Fails

ACDR-Help was intended to learn different residual doses for different
samples. In both datasets, however, the binary critic converged to
`help_probability` near 1 and `dose` near its maximum of 2. The router therefore
behaves as an almost constant global amplifier:

\[
z_i \approx z_{base,i} + 2\beta r_i d_i.
\]

This duplicates the role already owned by the learned global scale `beta`.
The critic is not allocating residual strength according to sample-specific
evidence, so the claimed sample-level innovation is not empirically supported.

The dataset asymmetry also matters. Global amplification helps Large-LI but
hurts Small-LI by `-0.01660` on the primary metric and `-0.01594` on raw-best.
Because the pre-registered funnel requires both difficult datasets to beat A2,
ACDR-Help fails and must not advance to Medium-HI/Large-HI or the remaining
datasets.

## 4. Final Interpretation

The rejected hypothesis is:

> A binary marginal help/harm critic will produce meaningful sample-level A2
> residual doses across the difficult LI datasets.

The retained insight is:

> A2-centered supervision can recover useful absolute residual direction, but
> unconstrained routing absorbs global scale and becomes redundant with beta.

This failure is structural rather than a request for more epochs or another
auxiliary-loss weight. The next method must make the sample-adaptive component
mathematically unable to express a global rescaling of the A2 residual.

## 5. Preserved Evidence

The stopped outputs are retained without modification at:

```text
results/acdr_help_pair500_faf972865181/
  AML-Large-LI-ACDRHelpPair500-Seed44-faf972865181-gpu0/
  AML-Small-LI-ACDRHelpPair500-Seed42-faf972865181-gpu0/
```

Related provenance files are retained at:

```text
.acdr_help_pair500_faf972865181.events
.acdr_help_pair500_faf972865181_active/
.acdr_help_pair500_AML-Large-LI-ACDRHelpPair500-Seed44-faf972865181_gpu5.log
.acdr_help_pair500_AML-Small-LI-ACDRHelpPair500-Seed42-faf972865181_gpu6.log
```

The active-marker directory is historical queue state after manual task
termination; it is not evidence that training should be resumed.

## 6. Motivation for COSTAR Orthogonalization

COSTAR should separate two effects that ACDR-Help allowed one critic to mix:

1. `beta` remains the sole owner of global prototype-residual magnitude.
2. The sample router is restricted to an orthogonal, centered redistribution
   component that cannot increase every sample's dose together.

Conceptually, the next prediction should preserve the A2 global component and
add only a constrained heterogeneous correction:

\[
z_i = z_i^{A2} + \gamma r_i \tilde u_i d_i,
\qquad \tilde u \perp \text{global-scale direction}.
\]

The orthogonalization constraint is motivated directly by the observed
`dose approximately 2` collapse. If the router has no genuine sample-level
signal, its orthogonal correction should vanish and the model should fall back
to A2, rather than becoming another global beta. A future COSTAR experiment
must verify nontrivial sample variance, a non-saturated route distribution, and
improvement on both Small-LI and Large-LI before expansion.

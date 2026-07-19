# DABR s15 Conservative Follow-up

## Material Passport

- Branch: `feature/dabr-s15-conservative-screen`
- Parent method commit: `0985b718`
- Primary metric: validation-selected Test F1
- Datasets: Small-LI seed 42 and Large-LI seed 44
- Budget: 500 epochs from epoch 0, no early stopping
- Route strength: fixed `rho=0.15`
- Doses: `{0.85, 1.00, 1.15}`

## Decision evidence

At the matched epoch-163 audit:

| Variant | Small-LI delta | Large-LI delta | Mean raw delta |
|---|---:|---:|---:|
| DABR s25 | -0.02338 | -0.01298 | -0.00596 |
| DABR s50 | -0.03733 | -0.12618 | -0.01235 |

The stronger route substantially worsens Large-LI, while s25 remains close to
A2 and has already reached `0.46565` selected Test F1 on Small-LI when compared
against the full A2 reference. This supports a monotone risk diagnosis:
counterfactual direction may be useful, but residual-dose magnitude is too
large on out-of-sample Large-LI edges.

`s15` tests the narrower hypothesis that a conservative action can retain the
sample-specific direction while reducing train-to-test overcorrection. It is a
new prospective run, not continuation or rescaling of an existing trajectory.

## Fixed protocol

- Code and model are unchanged from DABR except this versioned protocol.
- `model.dabr_route_strength=0.15` is fixed before epoch 0.
- The original A2 weighted-CE anchor remains independently optimized.
- Router construction preserves global RNG and router/A2 clipping is separate.
- GPU0 is excluded; OCR is non-blocking.
- Raw-best remains diagnostic only.

The formal pair gate remains:

| Dataset | A2 selected Test F1 | Required s15 F1 |
|---|---:|---:|
| Small-LI | 0.46247 | at least 0.46747 |
| Large-LI | 0.30108 | at least 0.30608 |

Both complete 500-epoch rows must pass before expansion.

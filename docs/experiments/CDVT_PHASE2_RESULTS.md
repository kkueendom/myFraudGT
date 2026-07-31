# CDVT Phase 2 Result

Status date: 2026-07-31

## Decision

The frozen CDVT model passed the preregistered six-dataset seed-42 gate under
the `dynamic_random` protocol:

- primary baseline: PE-FraudGT from FraudGT Table 2;
- wins: 5/6;
- mean val-selected test F1: 0.59224;
- mean delta versus PE-FraudGT: +0.02716;
- Phase 3 decision: advance.

## Main Results

| Dataset | PE-FraudGT | CDVT | Delta |
|---|---:|---:|---:|
| Small-LI | 0.45810 | 0.43798 | -0.02012 |
| Small-HI | 0.76410 | 0.76940 | +0.00530 |
| Medium-LI | 0.43530 | 0.44711 | +0.01181 |
| Medium-HI | 0.74220 | 0.77301 | +0.03081 |
| Large-LI | 0.30440 | 0.33803 | +0.03363 |
| Large-HI | 0.68640 | 0.78793 | +0.10153 |
| Mean | 0.56508 | 0.59224 | +0.02716 |

CDVT also wins 4/6 against Multi-FraudGT, with mean delta +0.00241. This mean
is below the predefined 0.005 dynamic-sampling band and is not interpreted as
stable overall superiority.

Against the initial A2 internal comparator, CDVT has mean delta -0.00104. It
does not dominate A2 overall, but improves Large-LI by +0.03695 and Large-HI
by +0.05896. The paired representative multi-seed experiments test whether
this scale-dependent advantage persists.

## Mechanism Evidence

| Condition | Small-LI | Medium-LI | Large-LI | Mean |
|---|---:|---:|---:|---:|
| Normal | 0.43798 | 0.44711 | 0.33803 | 0.40771 |
| Shuffled | 0.09350 | 0.12058 | 0.01596 | 0.07668 |
| Off | 0.27294 | 0.03012 | 0.00000 | 0.10102 |

Normal event context exceeds shuffled and off controls on all three
representative scales. This supports use of target-aligned event history but
is not a causal-effect claim.

## Provenance

- architecture freeze: `9038f85`;
- Phase 1 execution commit: `9c18cfdd`;
- Phase 2 portable execution commit: `2fb3333`;
- Phase 1 root:
  `/e/yky/FraudGT_cdvt_results/phase1_formal_9c18cfdd`;
- Phase 2 root:
  `/e/yky/FraudGT_cdvt_results/phase2_2fb3333`;
- authoritative JSON:
  `results/CDVT_Phase2_Six_Dataset_Summary.json`;
- JSON SHA-256:
  `d348b0f3278a8a532d9a5751f2cb4fad69ea0332eb06490c33834d6f01b182d0`;
- generated Markdown:
  `results/CDVT_Phase2_Six_Dataset_Summary.md`;
- Markdown SHA-256:
  `c6a5a4eb3de673e351787f11aa4f97ef21963cbd901bfa9ab56216c13a1fc284`.

The follow-up execution source is `34456ab`; its parentless portable snapshot
is `f7209f2`. It runs paired seeds 43/44, missing core controls, no-relation and
K=2 ablations, and normal-only runtime benchmarks.

# CDVT Phase 2 Six-Dataset Summary

Frozen model: causal event graph plus dual-view cross-attention, without sampling consistency.

Sampling protocol: `dynamic_random`. Formal selection uses validation F1 against PE-FraudGT, the direct parent architecture reported in FraudGT Table 2. Multi-FraudGT is the stronger published reference. Raw-best is supplementary and is compared only with A2 raw-best.

## Published FraudGT references

| Dataset | PE-FraudGT | CDVT | Delta vs PE | Status | Multi-FraudGT | Delta vs Multi |
|---|---:|---:|---:|---|---:|---:|
| Small-LI | 0.45810 | 0.43798 | -0.02012 | clear_loss | 0.47010 | -0.03212 |
| Small-HI | 0.76410 | 0.76940 | +0.00530 | clear_gain | 0.76130 | +0.00810 |
| Medium-LI | 0.43530 | 0.44711 | +0.01181 | clear_gain | 0.44060 | +0.00651 |
| Medium-HI | 0.74220 | 0.77301 | +0.03081 | clear_gain | 0.75930 | +0.01371 |
| Large-LI | 0.30440 | 0.33803 | +0.03363 | clear_gain | 0.37430 | -0.03627 |
| Large-HI | 0.68640 | 0.78793 | +0.10153 | clear_gain | 0.73340 | +0.05453 |

## Internal A2 reference

| Dataset | A2 val-selected | CDVT val-selected | Delta | Status | A2 raw-best | CDVT raw-best | Delta |
|---|---:|---:|---:|---|---:|---:|---:|
| Small-LI | 0.46247 | 0.43798 | -0.02449 | clear_loss | 0.50667 | 0.47687 | -0.02980 |
| Small-HI | 0.77984 | 0.76940 | -0.01044 | clear_loss | 0.79497 | 0.79128 | -0.00369 |
| Medium-LI | 0.51163 | 0.44711 | -0.06452 | clear_loss | 0.59031 | 0.51029 | -0.08002 |
| Medium-HI | 0.77574 | 0.77301 | -0.00273 | possible_sampling_variation | 0.78940 | 0.80088 | +0.01148 |
| Large-LI | 0.30108 | 0.33803 | +0.03695 | clear_gain | 0.44720 | 0.44816 | +0.00096 |
| Large-HI | 0.72897 | 0.78793 | +0.05896 | clear_gain | 0.76223 | 0.79284 | +0.03061 |

Val-selected vs PE-FraudGT: 5 wins, 1 losses, mean delta +0.02716.
Val-selected vs Multi-FraudGT: 4 wins, 2 losses, mean delta +0.00241.
Val-selected vs A2: 2 wins, 4 losses, mean delta -0.00104.
Raw-best vs A2: 3 wins, 3 losses, mean delta -0.01174.

Advance to Phase 3: True.

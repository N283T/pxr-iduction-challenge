# Submission Preflight: `ens_id55_combo_gate_rank3.csv`

Verdict: **PASS**

## Inputs

- candidate: `track1_activity/submissions/ens_id55_combo_gate_rank3.csv`
- anchor: `track1_activity/submissions/ens_id51_top500_potent46_t40_soft_g35.csv`

## CSV Sanity

- rows: 513 / 513
- SMILES order match: True
- Molecule Name order match: True

## Anchor Shift

- Pearson vs anchor: 0.999736
- Spearman vs anchor: 0.999460
- mean shift: +0.008288
- mean abs shift: 0.012900
- p90 abs shift: 0.034483
- max abs shift: 0.086024
- |shift| > 0.05: 21
- |shift| > 0.10: 0
- |shift| > 0.20: 0

## Prediction Distribution

- anchor mean/std: 4.798350 / 0.770987
- candidate mean/std: 4.806638 / 0.777810

## Known Bad Axis

| label           |   pearson |   spearman |   candidate_projection |
|:----------------|----------:|-----------:|-----------------------:|
| id56_minus_id55 |  0.326764 |   0.393740 |               0.083228 |
| id56_minus_id51 |  0.526488 |   0.567360 |               0.155113 |

## Experiment Metadata

No matching `experiment_summary` row found for this CSV path.

## Reasons

- small_anchor_shift

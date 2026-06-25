# Submission Preflight: `phase2_as1_aug_top500_id55blend_a0p4_pairrankchembl_q95_g0p15_labels_as1.csv`

Verdict: **PASS**

## Inputs

- candidate: `track1_activity/submissions/phase2_as1_aug_top500_id55blend_a0p4_pairrankchembl_q95_g0p15_labels_as1.csv`
- anchor: `track1_activity/submissions/phase2_as1_aug_top500_id55blend_a0p4_labels_as1.csv`

## CSV Sanity

- rows: 513 / 513
- SMILES order match: True
- Molecule Name order match: True

## Anchor Shift

- Pearson vs anchor: 0.999204
- Spearman vs anchor: 0.997954
- mean shift: +0.009357
- mean abs shift: 0.009357
- p90 abs shift: 0.000000
- max abs shift: 0.150000
- |shift| > 0.05: 32
- |shift| > 0.10: 32
- |shift| > 0.20: 0

## Prediction Distribution

- anchor mean/std: 4.760491 / 0.895318
- candidate mean/std: 4.769848 / 0.901172

## Known Bad Axis

| label           |   pearson |   spearman |   candidate_projection |
|:----------------|----------:|-----------:|-----------------------:|
| id56_minus_id55 | -0.006969 |   0.001879 |              -0.008765 |
| id56_minus_id51 |  0.042412 |   0.043386 |               0.020334 |

## Experiment Metadata

No matching `experiment_summary` row found for this CSV path.

## Reasons

- small_anchor_shift

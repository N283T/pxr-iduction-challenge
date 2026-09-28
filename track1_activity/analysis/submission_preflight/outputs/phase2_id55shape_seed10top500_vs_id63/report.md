# Submission Preflight: `phase2_as1_aug_suite_id55shape_seed10top500_t40_soft_g35_labels_as1.csv`

Verdict: **PASS**

## Inputs

- candidate: `track1_activity/submissions/phase2_as1_aug_suite_id55shape_seed10top500_t40_soft_g35_labels_as1.csv`
- anchor: `track1_activity/submissions/phase2_as1_aug_top500_id55blend_a0p45_pairrankchembl_q95_g0p15_plus_combo_new_h0p15_l0p15_labels_as1.csv`

## CSV Sanity

- rows: 513 / 513
- SMILES order match: True
- Molecule Name order match: True

## Anchor Shift

- Pearson vs anchor: 0.997957
- Spearman vs anchor: 0.996918
- mean shift: +0.005423
- mean abs shift: 0.032328
- p90 abs shift: 0.094313
- max abs shift: 0.283472
- |shift| > 0.05: 128
- |shift| > 0.10: 48
- |shift| > 0.20: 5

## Prediction Distribution

- anchor mean/std: 4.768881 / 0.901369
- candidate mean/std: 4.774304 / 0.899600

## Known Bad Axis

| label           |   pearson |   spearman |   candidate_projection |
|:----------------|----------:|-----------:|-----------------------:|
| id56_minus_id55 |  0.131231 |   0.157717 |               0.103551 |
| id56_minus_id51 |  0.060584 |   0.092426 |               0.052833 |

## Experiment Metadata

No matching `experiment_summary` row found for this CSV path.

## Reasons

- small_anchor_shift

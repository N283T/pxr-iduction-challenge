# Submission Preflight: `phase2_as1_aug_top500_id55blend_a0p4_pairrankchembl_q95_g0p15_plus_combo_new_h0p10_l0p10_labels_as1.csv`

Verdict: **PASS**

## Inputs

- candidate: `track1_activity/submissions/phase2_as1_aug_top500_id55blend_a0p4_pairrankchembl_q95_g0p15_plus_combo_new_h0p10_l0p10_labels_as1.csv`
- anchor: `track1_activity/submissions/phase2_as1_aug_top500_id55blend_a0p4_pairrankchembl_q95_g0p15_labels_as1.csv`

## CSV Sanity

- rows: 513 / 513
- SMILES order match: True
- Molecule Name order match: True

## Anchor Shift

- Pearson vs anchor: 0.999976
- Spearman vs anchor: 0.999909
- mean shift: +0.000390
- mean abs shift: 0.000390
- p90 abs shift: 0.000000
- max abs shift: 0.100000
- |shift| > 0.05: 2
- |shift| > 0.10: 1
- |shift| > 0.20: 0

## Prediction Distribution

- anchor mean/std: 4.769848 / 0.901172
- candidate mean/std: 4.770238 / 0.901404

## Known Bad Axis

| label           |   pearson |   spearman |   candidate_projection |
|:----------------|----------:|-----------:|-----------------------:|
| id56_minus_id55 | -0.003476 |  -0.001852 |              -0.000522 |
| id56_minus_id51 | -0.008124 |  -0.004612 |              -0.000968 |

## Experiment Metadata

No matching `experiment_summary` row found for this CSV path.

## Reasons

- small_anchor_shift

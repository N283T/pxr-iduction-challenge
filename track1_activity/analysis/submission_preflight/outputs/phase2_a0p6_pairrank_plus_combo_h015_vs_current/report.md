# Submission Preflight: `phase2_as1_aug_top500_id55blend_a0p6_pairrankchembl_q95_g0p15_plus_combo_new_h0p15_l0p15_labels_as1.csv`

Verdict: **PASS**

## Inputs

- candidate: `track1_activity/submissions/phase2_as1_aug_top500_id55blend_a0p6_pairrankchembl_q95_g0p15_plus_combo_new_h0p15_l0p15_labels_as1.csv`
- anchor: `track1_activity/submissions/phase2_as1_aug_top500_id55blend_a0p4_pairrankchembl_q95_g0p15_labels_as1.csv`

## CSV Sanity

- rows: 513 / 513
- SMILES order match: True
- Molecule Name order match: True

## Anchor Shift

- Pearson vs anchor: 0.999653
- Spearman vs anchor: 0.999439
- mean shift: -0.005623
- mean abs shift: 0.013029
- p90 abs shift: 0.037000
- max abs shift: 0.137834
- |shift| > 0.05: 28
- |shift| > 0.10: 4
- |shift| > 0.20: 0

## Prediction Distribution

- anchor mean/std: 4.769848 / 0.901172
- candidate mean/std: 4.764225 / 0.901080

## Known Bad Axis

| label           |   pearson |   spearman |   candidate_projection |
|:----------------|----------:|-----------:|-----------------------:|
| id56_minus_id55 | -0.197396 |  -0.162732 |              -0.062991 |
| id56_minus_id51 | -0.123554 |  -0.090978 |              -0.043986 |

## Experiment Metadata

No matching `experiment_summary` row found for this CSV path.

## Reasons

- small_anchor_shift

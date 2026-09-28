# Submission Preflight: `phase2_as1_aug_top500_id55blend_a0p4_combohtchemcp02_q98_h0p20_cpabs01lowq98_l0p20_labels_as1.csv`

Verdict: **HOLD**

## Inputs

- candidate: `track1_activity/submissions/phase2_as1_aug_top500_id55blend_a0p4_combohtchemcp02_q98_h0p20_cpabs01lowq98_l0p20_labels_as1.csv`
- anchor: `track1_activity/submissions/ens_id51_top500_potent46_t40_soft_g35.csv`

## CSV Sanity

- rows: 513 / 513
- SMILES order match: True
- Molecule Name order match: True

## Anchor Shift

- Pearson vs anchor: 0.888684
- Spearman vs anchor: 0.919412
- mean shift: -0.035130
- mean abs shift: 0.228073
- p90 abs shift: 0.615996
- max abs shift: 2.875959
- |shift| > 0.05: 329
- |shift| > 0.10: 233
- |shift| > 0.20: 169

## Prediction Distribution

- anchor mean/std: 4.798350 / 0.770987
- candidate mean/std: 4.763220 / 0.896905

## Known Bad Axis

| label           |   pearson |   spearman |   candidate_projection |
|:----------------|----------:|-----------:|-----------------------:|
| id56_minus_id55 | -0.015195 |  -0.061294 |              -0.068770 |
| id56_minus_id51 |  0.014327 |  -0.034658 |               0.108391 |

## Experiment Metadata

No matching `experiment_summary` row found for this CSV path.

## Reasons

- large_anchor_shift
- extreme_single_compound_shift
- prediction_scale_changed
- rank_order_changed

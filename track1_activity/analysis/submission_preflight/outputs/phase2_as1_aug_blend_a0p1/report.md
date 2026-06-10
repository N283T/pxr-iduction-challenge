# Submission Preflight: `phase2_as1_aug_top500_id55blend_a0p1_labels_as1.csv`

Verdict: **HOLD**

## Inputs

- candidate: `track1_activity/submissions/phase2_as1_aug_top500_id55blend_a0p1_labels_as1.csv`
- anchor: `track1_activity/submissions/ens_id51_top500_potent46_t40_soft_g35.csv`

## CSV Sanity

- rows: 513 / 513
- SMILES order match: True
- Molecule Name order match: True

## Anchor Shift

- Pearson vs anchor: 0.890334
- Spearman vs anchor: 0.921857
- mean shift: -0.028547
- mean abs shift: 0.206870
- p90 abs shift: 0.615996
- max abs shift: 2.875959
- |shift| > 0.05: 224
- |shift| > 0.10: 202
- |shift| > 0.20: 162

## Prediction Distribution

- anchor mean/std: 4.798350 / 0.770987
- candidate mean/std: 4.769803 / 0.896998

## Known Bad Axis

| label           |   pearson |   spearman |   candidate_projection |
|:----------------|----------:|-----------:|-----------------------:|
| id56_minus_id55 |  0.005924 |  -0.043484 |               0.050048 |
| id56_minus_id51 |  0.024900 |  -0.026000 |               0.173368 |

## Experiment Metadata

No matching `experiment_summary` row found for this CSV path.

## Reasons

- large_anchor_shift
- extreme_single_compound_shift
- prediction_scale_changed
- rank_order_changed

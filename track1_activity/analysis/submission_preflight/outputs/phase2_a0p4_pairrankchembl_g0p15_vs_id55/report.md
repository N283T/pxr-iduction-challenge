# Submission Preflight: `phase2_as1_aug_top500_id55blend_a0p4_pairrankchembl_q95_g0p15_labels_as1.csv`

Verdict: **HOLD**

## Inputs

- candidate: `track1_activity/submissions/phase2_as1_aug_top500_id55blend_a0p4_pairrankchembl_q95_g0p15_labels_as1.csv`
- anchor: `track1_activity/submissions/ens_id51_top500_potent46_t40_soft_g35.csv`

## CSV Sanity

- rows: 513 / 513
- SMILES order match: True
- Molecule Name order match: True

## Anchor Shift

- Pearson vs anchor: 0.889367
- Spearman vs anchor: 0.919755
- mean shift: -0.028502
- mean abs shift: 0.231361
- p90 abs shift: 0.615996
- max abs shift: 2.875959
- |shift| > 0.05: 344
- |shift| > 0.10: 249
- |shift| > 0.20: 166

## Prediction Distribution

- anchor mean/std: 4.798350 / 0.770987
- candidate mean/std: 4.769848 / 0.901172

## Known Bad Axis

| label           |   pearson |   spearman |   candidate_projection |
|:----------------|----------:|-----------:|-----------------------:|
| id56_minus_id55 | -0.011650 |  -0.059597 |              -0.052028 |
| id56_minus_id51 |  0.018019 |  -0.034342 |               0.129901 |

## Experiment Metadata

No matching `experiment_summary` row found for this CSV path.

## Reasons

- large_anchor_shift
- extreme_single_compound_shift
- prediction_scale_changed
- rank_order_changed

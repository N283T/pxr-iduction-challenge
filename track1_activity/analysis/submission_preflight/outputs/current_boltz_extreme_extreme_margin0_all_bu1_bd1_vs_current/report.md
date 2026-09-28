# Submission Preflight: `phase2_current_boltz_extreme_extreme_margin0_all_bu1_bd1_labels_as1.csv`

Verdict: **HOLD**

## Inputs

- candidate: `track1_activity/submissions/phase2_current_boltz_extreme_extreme_margin0_all_bu1_bd1_labels_as1.csv`
- anchor: `track1_activity/submissions/phase2_as1_aug_top500_id55blend_a0p4_pairrankchembl_q95_g0p15_labels_as1.csv`

## CSV Sanity

- rows: 513 / 513
- SMILES order match: True
- Molecule Name order match: True

## Anchor Shift

- Pearson vs anchor: 0.956353
- Spearman vs anchor: 0.933350
- mean shift: -0.037566
- mean abs shift: 0.145299
- p90 abs shift: 0.455725
- max abs shift: 1.164410
- |shift| > 0.05: 221
- |shift| > 0.10: 201
- |shift| > 0.20: 144

## Prediction Distribution

- anchor mean/std: 4.769848 / 0.901172
- candidate mean/std: 4.732282 / 0.853517

## Known Bad Axis

| label           |   pearson |   spearman |   candidate_projection |
|:----------------|----------:|-----------:|-----------------------:|
| id56_minus_id55 | -0.034404 |  -0.121320 |              -0.106937 |
| id56_minus_id51 | -0.180575 |  -0.232907 |              -0.735981 |

## Experiment Metadata

No matching `experiment_summary` row found for this CSV path.

## Reasons

- large_anchor_shift
- extreme_single_compound_shift
- rank_order_changed

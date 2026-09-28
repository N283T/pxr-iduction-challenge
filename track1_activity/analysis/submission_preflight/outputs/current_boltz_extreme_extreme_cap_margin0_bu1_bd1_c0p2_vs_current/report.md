# Submission Preflight: `phase2_current_boltz_extreme_extreme_cap_margin0_bu1_bd1_c0p2_labels_as1.csv`

Verdict: **HOLD**

## Inputs

- candidate: `track1_activity/submissions/phase2_current_boltz_extreme_extreme_cap_margin0_bu1_bd1_c0p2_labels_as1.csv`
- anchor: `track1_activity/submissions/phase2_as1_aug_top500_id55blend_a0p4_pairrankchembl_q95_g0p15_labels_as1.csv`

## CSV Sanity

- rows: 513 / 513
- SMILES order match: True
- Molecule Name order match: True

## Anchor Shift

- Pearson vs anchor: 0.991892
- Spearman vs anchor: 0.984874
- mean shift: -0.021997
- mean abs shift: 0.076188
- p90 abs shift: 0.200000
- max abs shift: 0.200000
- |shift| > 0.05: 221
- |shift| > 0.10: 201
- |shift| > 0.20: 135

## Prediction Distribution

- anchor mean/std: 4.769848 / 0.901172
- candidate mean/std: 4.747851 / 0.874649

## Known Bad Axis

| label           |   pearson |   spearman |   candidate_projection |
|:----------------|----------:|-----------:|-----------------------:|
| id56_minus_id55 | -0.081010 |  -0.121656 |              -0.120420 |
| id56_minus_id51 | -0.214525 |  -0.230272 |              -0.384381 |

## Experiment Metadata

No matching `experiment_summary` row found for this CSV path.

## Reasons

- large_anchor_shift
- rank_order_changed

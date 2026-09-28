# Submission Preflight: `phase2_current_boltz_extreme_extreme_cap_margin0_bu1_bd0p75_c0p15_labels_as1.csv`

Verdict: **HOLD**

## Inputs

- candidate: `track1_activity/submissions/phase2_current_boltz_extreme_extreme_cap_margin0_bu1_bd0p75_c0p15_labels_as1.csv`
- anchor: `track1_activity/submissions/phase2_as1_aug_top500_id55blend_a0p4_pairrankchembl_q95_g0p15_labels_as1.csv`

## CSV Sanity

- rows: 513 / 513
- SMILES order match: True
- Molecule Name order match: True

## Anchor Shift

- Pearson vs anchor: 0.995258
- Spearman vs anchor: 0.990876
- mean shift: -0.014845
- mean abs shift: 0.058794
- p90 abs shift: 0.150000
- max abs shift: 0.150000
- |shift| > 0.05: 219
- |shift| > 0.10: 191
- |shift| > 0.20: 0

## Prediction Distribution

- anchor mean/std: 4.769848 / 0.901172
- candidate mean/std: 4.755003 / 0.880372

## Known Bad Axis

| label           |   pearson |   spearman |   candidate_projection |
|:----------------|----------:|-----------:|-----------------------:|
| id56_minus_id55 | -0.074850 |  -0.118876 |              -0.085894 |
| id56_minus_id51 | -0.207735 |  -0.228104 |              -0.286567 |

## Experiment Metadata

No matching `experiment_summary` row found for this CSV path.

## Reasons

- large_anchor_shift
- rank_order_changed

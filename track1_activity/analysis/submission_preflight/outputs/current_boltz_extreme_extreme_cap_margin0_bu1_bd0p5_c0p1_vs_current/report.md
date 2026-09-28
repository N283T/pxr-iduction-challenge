# Submission Preflight: `phase2_current_boltz_extreme_extreme_cap_margin0_bu1_bd0p5_c0p1_labels_as1.csv`

Verdict: **PASS**

## Inputs

- candidate: `track1_activity/submissions/phase2_current_boltz_extreme_extreme_cap_margin0_bu1_bd0p5_c0p1_labels_as1.csv`
- anchor: `track1_activity/submissions/phase2_as1_aug_top500_id55blend_a0p4_pairrankchembl_q95_g0p15_labels_as1.csv`

## CSV Sanity

- rows: 513 / 513
- SMILES order match: True
- Molecule Name order match: True

## Anchor Shift

- Pearson vs anchor: 0.997801
- Spearman vs anchor: 0.995769
- mean shift: -0.008673
- mean abs shift: 0.040419
- p90 abs shift: 0.100000
- max abs shift: 0.100000
- |shift| > 0.05: 211
- |shift| > 0.10: 32
- |shift| > 0.20: 0

## Prediction Distribution

- anchor mean/std: 4.769848 / 0.901172
- candidate mean/std: 4.761175 / 0.886546

## Known Bad Axis

| label           |   pearson |   spearman |   candidate_projection |
|:----------------|----------:|-----------:|-----------------------:|
| id56_minus_id55 | -0.070790 |  -0.127256 |              -0.056118 |
| id56_minus_id51 | -0.202990 |  -0.232746 |              -0.192188 |

## Experiment Metadata

No matching `experiment_summary` row found for this CSV path.

## Reasons

- small_anchor_shift

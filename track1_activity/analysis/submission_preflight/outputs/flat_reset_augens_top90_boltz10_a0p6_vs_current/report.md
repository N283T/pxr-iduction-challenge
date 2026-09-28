# Submission Preflight: `phase2_flat_reset_augens_top90_boltz10_id55blend_a0p6_pairrank_q95_g0p15_labels_as1.csv`

Verdict: **PASS**

## Inputs

- candidate: `track1_activity/submissions/phase2_flat_reset_augens_top90_boltz10_id55blend_a0p6_pairrank_q95_g0p15_labels_as1.csv`
- anchor: `track1_activity/submissions/phase2_as1_aug_top500_id55blend_a0p4_pairrankchembl_q95_g0p15_labels_as1.csv`

## CSV Sanity

- rows: 513 / 513
- SMILES order match: True
- Molecule Name order match: True

## Anchor Shift

- Pearson vs anchor: 0.999766
- Spearman vs anchor: 0.999698
- mean shift: -0.006821
- mean abs shift: 0.011889
- p90 abs shift: 0.035840
- max abs shift: 0.099518
- |shift| > 0.05: 20
- |shift| > 0.10: 0
- |shift| > 0.20: 0

## Prediction Distribution

- anchor mean/std: 4.769848 / 0.901172
- candidate mean/std: 4.763027 / 0.896053

## Known Bad Axis

| label           |   pearson |   spearman |   candidate_projection |
|:----------------|----------:|-----------:|-----------------------:|
| id56_minus_id55 | -0.224298 |  -0.256352 |              -0.059752 |
| id56_minus_id51 | -0.260943 |  -0.281585 |              -0.079956 |

## Experiment Metadata

No matching `experiment_summary` row found for this CSV path.

## Reasons

- small_anchor_shift

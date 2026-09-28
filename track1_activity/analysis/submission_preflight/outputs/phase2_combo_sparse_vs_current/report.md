# Submission Preflight: `phase2_as1_aug_top500_id55blend_a0p4_combohtchemcp02_q98_h0p20_cpabs01lowq98_l0p20_labels_as1.csv`

Verdict: **PASS**

## Inputs

- candidate: `track1_activity/submissions/phase2_as1_aug_top500_id55blend_a0p4_combohtchemcp02_q98_h0p20_cpabs01lowq98_l0p20_labels_as1.csv`
- anchor: `track1_activity/submissions/phase2_as1_aug_top500_id55blend_a0p4_pairrankchembl_q95_g0p15_labels_as1.csv`

## CSV Sanity

- rows: 513 / 513
- SMILES order match: True
- Molecule Name order match: True

## Anchor Shift

- Pearson vs anchor: 0.999193
- Spearman vs anchor: 0.997863
- mean shift: -0.006628
- mean abs shift: 0.009162
- p90 abs shift: 0.000000
- max abs shift: 0.200000
- |shift| > 0.05: 29
- |shift| > 0.10: 29
- |shift| > 0.20: 2

## Prediction Distribution

- anchor mean/std: 4.769848 / 0.901172
- candidate mean/std: 4.763220 / 0.896905

## Known Bad Axis

| label           |   pearson |   spearman |   candidate_projection |
|:----------------|----------:|-----------:|-----------------------:|
| id56_minus_id55 | -0.039820 |  -0.063134 |              -0.016742 |
| id56_minus_id51 | -0.042349 |  -0.037410 |              -0.021511 |

## Experiment Metadata

No matching `experiment_summary` row found for this CSV path.

## Reasons

- small_anchor_shift

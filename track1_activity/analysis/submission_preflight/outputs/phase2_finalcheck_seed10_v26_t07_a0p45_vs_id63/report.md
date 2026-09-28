# Submission Preflight: `phase2_finalcheck_seed10_v2_6_t0p7_a0p45_pairrank_q95_combo_new_h015_labels_as1.csv`

Verdict: **PASS**

## Inputs

- candidate: `track1_activity/submissions/phase2_finalcheck_seed10_v2_6_t0p7_a0p45_pairrank_q95_combo_new_h015_labels_as1.csv`
- anchor: `track1_activity/submissions/phase2_as1_aug_top500_id55blend_a0p45_pairrankchembl_q95_g0p15_plus_combo_new_h0p15_l0p15_labels_as1.csv`

## CSV Sanity

- rows: 513 / 513
- SMILES order match: True
- Molecule Name order match: True

## Anchor Shift

- Pearson vs anchor: 0.999276
- Spearman vs anchor: 0.999195
- mean shift: -0.001461
- mean abs shift: 0.016967
- p90 abs shift: 0.049652
- max abs shift: 0.207625
- |shift| > 0.05: 51
- |shift| > 0.10: 14
- |shift| > 0.20: 1

## Prediction Distribution

- anchor mean/std: 4.768881 / 0.901369
- candidate mean/std: 4.767420 / 0.900985

## Known Bad Axis

| label           |   pearson |   spearman |   candidate_projection |
|:----------------|----------:|-----------:|-----------------------:|
| id56_minus_id55 |  0.089411 |   0.020433 |               0.044037 |
| id56_minus_id51 |  0.016120 |  -0.034275 |               0.009364 |

## Experiment Metadata

No matching `experiment_summary` row found for this CSV path.

## Reasons

- small_anchor_shift

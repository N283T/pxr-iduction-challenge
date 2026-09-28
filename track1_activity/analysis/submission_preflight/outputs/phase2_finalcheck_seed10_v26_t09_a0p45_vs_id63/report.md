# Submission Preflight: `phase2_finalcheck_seed10_v2_6_t0p9_a0p45_pairrank_q95_combo_new_h015_labels_as1.csv`

Verdict: **PASS**

## Inputs

- candidate: `track1_activity/submissions/phase2_finalcheck_seed10_v2_6_t0p9_a0p45_pairrank_q95_combo_new_h015_labels_as1.csv`
- anchor: `track1_activity/submissions/phase2_as1_aug_top500_id55blend_a0p45_pairrankchembl_q95_g0p15_plus_combo_new_h0p15_l0p15_labels_as1.csv`

## CSV Sanity

- rows: 513 / 513
- SMILES order match: True
- Molecule Name order match: True

## Anchor Shift

- Pearson vs anchor: 0.999229
- Spearman vs anchor: 0.999153
- mean shift: -0.002964
- mean abs shift: 0.017420
- p90 abs shift: 0.051133
- max abs shift: 0.225537
- |shift| > 0.05: 55
- |shift| > 0.10: 19
- |shift| > 0.20: 1

## Prediction Distribution

- anchor mean/std: 4.768881 / 0.901369
- candidate mean/std: 4.765917 / 0.900624

## Known Bad Axis

| label           |   pearson |   spearman |   candidate_projection |
|:----------------|----------:|-----------:|-----------------------:|
| id56_minus_id55 |  0.093001 |   0.024296 |               0.048061 |
| id56_minus_id51 |  0.016408 |  -0.030687 |               0.010453 |

## Experiment Metadata

No matching `experiment_summary` row found for this CSV path.

## Reasons

- small_anchor_shift

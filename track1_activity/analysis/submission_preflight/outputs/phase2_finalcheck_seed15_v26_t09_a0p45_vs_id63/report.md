# Submission Preflight: `phase2_finalcheck_seed15_v2_6_t0p9_a0p45_pairrank_q95_combo_new_h015_labels_as1.csv`

Verdict: **PASS**

## Inputs

- candidate: `track1_activity/submissions/phase2_finalcheck_seed15_v2_6_t0p9_a0p45_pairrank_q95_combo_new_h015_labels_as1.csv`
- anchor: `track1_activity/submissions/phase2_as1_aug_top500_id55blend_a0p45_pairrankchembl_q95_g0p15_plus_combo_new_h0p15_l0p15_labels_as1.csv`

## CSV Sanity

- rows: 513 / 513
- SMILES order match: True
- Molecule Name order match: True

## Anchor Shift

- Pearson vs anchor: 0.999144
- Spearman vs anchor: 0.999013
- mean shift: +0.000535
- mean abs shift: 0.018082
- p90 abs shift: 0.056306
- max abs shift: 0.341390
- |shift| > 0.05: 64
- |shift| > 0.10: 15
- |shift| > 0.20: 1

## Prediction Distribution

- anchor mean/std: 4.768881 / 0.901369
- candidate mean/std: 4.769415 / 0.899562

## Known Bad Axis

| label           |   pearson |   spearman |   candidate_projection |
|:----------------|----------:|-----------:|-----------------------:|
| id56_minus_id55 |  0.109907 |   0.035195 |               0.057523 |
| id56_minus_id51 |  0.037527 |  -0.012898 |               0.021905 |

## Experiment Metadata

No matching `experiment_summary` row found for this CSV path.

## Reasons

- small_anchor_shift

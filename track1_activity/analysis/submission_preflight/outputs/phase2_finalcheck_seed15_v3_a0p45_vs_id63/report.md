# Submission Preflight: `phase2_finalcheck_seed15_v3_t0p7_a0p45_pairrank_q95_combo_new_h015_labels_as1.csv`

Verdict: **PASS**

## Inputs

- candidate: `track1_activity/submissions/phase2_finalcheck_seed15_v3_t0p7_a0p45_pairrank_q95_combo_new_h015_labels_as1.csv`
- anchor: `track1_activity/submissions/phase2_as1_aug_top500_id55blend_a0p45_pairrankchembl_q95_g0p15_plus_combo_new_h0p15_l0p15_labels_as1.csv`

## CSV Sanity

- rows: 513 / 513
- SMILES order match: True
- Molecule Name order match: True

## Anchor Shift

- Pearson vs anchor: 0.999692
- Spearman vs anchor: 0.999581
- mean shift: +0.001926
- mean abs shift: 0.011076
- p90 abs shift: 0.034408
- max abs shift: 0.166858
- |shift| > 0.05: 21
- |shift| > 0.10: 5
- |shift| > 0.20: 0

## Prediction Distribution

- anchor mean/std: 4.768881 / 0.901369
- candidate mean/std: 4.770807 / 0.899846

## Known Bad Axis

| label           |   pearson |   spearman |   candidate_projection |
|:----------------|----------:|-----------:|-----------------------:|
| id56_minus_id55 | -0.063670 |  -0.016021 |              -0.021175 |
| id56_minus_id51 | -0.109998 |  -0.051766 |              -0.039770 |

## Experiment Metadata

No matching `experiment_summary` row found for this CSV path.

## Reasons

- small_anchor_shift

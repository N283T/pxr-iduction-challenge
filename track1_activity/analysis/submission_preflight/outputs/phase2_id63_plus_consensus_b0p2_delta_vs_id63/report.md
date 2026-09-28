# Submission Preflight: `phase2_id63_plus_consensus_boltz_top500_b0p2_delta_labels_as1.csv`

Verdict: **PASS**

## Inputs

- candidate: `track1_activity/submissions/phase2_id63_plus_consensus_boltz_top500_b0p2_delta_labels_as1.csv`
- anchor: `track1_activity/submissions/phase2_as1_aug_top500_id55blend_a0p45_pairrankchembl_q95_g0p15_plus_combo_new_h0p15_l0p15_labels_as1.csv`

## CSV Sanity

- rows: 513 / 513
- SMILES order match: True
- Molecule Name order match: True

## Anchor Shift

- Pearson vs anchor: 0.999816
- Spearman vs anchor: 0.999665
- mean shift: -0.005814
- mean abs shift: 0.008088
- p90 abs shift: 0.035488
- max abs shift: 0.085424
- |shift| > 0.05: 25
- |shift| > 0.10: 0
- |shift| > 0.20: 0

## Prediction Distribution

- anchor mean/std: 4.768881 / 0.901369
- candidate mean/std: 4.763067 / 0.896623

## Known Bad Axis

| label           |   pearson |   spearman |   candidate_projection |
|:----------------|----------:|-----------:|-----------------------:|
| id56_minus_id55 | -0.190559 |  -0.220373 |              -0.044811 |
| id56_minus_id51 | -0.252776 |  -0.252729 |              -0.068955 |

## Experiment Metadata

No matching `experiment_summary` row found for this CSV path.

## Reasons

- small_anchor_shift

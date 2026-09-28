# Submission Preflight: `phase2_current_plus_boltz_agree_two_top500s_toward_consensus_boltz_top500_b0p2_labels_as1.csv`

Verdict: **PASS**

## Inputs

- candidate: `track1_activity/submissions/phase2_current_plus_boltz_agree_two_top500s_toward_consensus_boltz_top500_b0p2_labels_as1.csv`
- anchor: `track1_activity/submissions/phase2_as1_aug_top500_id55blend_a0p4_pairrankchembl_q95_g0p15_labels_as1.csv`

## CSV Sanity

- rows: 513 / 513
- SMILES order match: True
- Molecule Name order match: True

## Anchor Shift

- Pearson vs anchor: 0.999816
- Spearman vs anchor: 0.999675
- mean shift: -0.005814
- mean abs shift: 0.008088
- p90 abs shift: 0.035488
- max abs shift: 0.085424
- |shift| > 0.05: 25
- |shift| > 0.10: 0
- |shift| > 0.20: 0

## Prediction Distribution

- anchor mean/std: 4.769848 / 0.901172
- candidate mean/std: 4.764034 / 0.896385

## Known Bad Axis

| label           |   pearson |   spearman |   candidate_projection |
|:----------------|----------:|-----------:|-----------------------:|
| id56_minus_id55 | -0.190559 |  -0.220373 |              -0.044811 |
| id56_minus_id51 | -0.252776 |  -0.252729 |              -0.068955 |

## Experiment Metadata

No matching `experiment_summary` row found for this CSV path.

## Reasons

- small_anchor_shift

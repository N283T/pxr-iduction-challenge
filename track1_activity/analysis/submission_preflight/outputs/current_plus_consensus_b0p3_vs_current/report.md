# Submission Preflight: `phase2_current_plus_boltz_agree_two_top500s_toward_consensus_boltz_top500_b0p3_labels_as1.csv`

Verdict: **PASS**

## Inputs

- candidate: `track1_activity/submissions/phase2_current_plus_boltz_agree_two_top500s_toward_consensus_boltz_top500_b0p3_labels_as1.csv`
- anchor: `track1_activity/submissions/phase2_as1_aug_top500_id55blend_a0p4_pairrankchembl_q95_g0p15_labels_as1.csv`

## CSV Sanity

- rows: 513 / 513
- SMILES order match: True
- Molecule Name order match: True

## Anchor Shift

- Pearson vs anchor: 0.999585
- Spearman vs anchor: 0.999309
- mean shift: -0.008720
- mean abs shift: 0.012132
- p90 abs shift: 0.053232
- max abs shift: 0.128137
- |shift| > 0.05: 56
- |shift| > 0.10: 9
- |shift| > 0.20: 0

## Prediction Distribution

- anchor mean/std: 4.769848 / 0.901172
- candidate mean/std: 4.761128 / 0.894116

## Known Bad Axis

| label           |   pearson |   spearman |   candidate_projection |
|:----------------|----------:|-----------:|-----------------------:|
| id56_minus_id55 | -0.190559 |  -0.220373 |              -0.067216 |
| id56_minus_id51 | -0.252776 |  -0.252729 |              -0.103433 |

## Experiment Metadata

No matching `experiment_summary` row found for this CSV path.

## Reasons

- small_anchor_shift

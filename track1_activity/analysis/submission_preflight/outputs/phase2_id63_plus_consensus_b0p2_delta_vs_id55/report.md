# Submission Preflight: `phase2_id63_plus_consensus_boltz_top500_b0p2_delta_labels_as1.csv`

Verdict: **HOLD**

## Inputs

- candidate: `track1_activity/submissions/phase2_id63_plus_consensus_boltz_top500_b0p2_delta_labels_as1.csv`
- anchor: `track1_activity/submissions/ens_id51_top500_potent46_t40_soft_g35.csv`

## CSV Sanity

- rows: 513 / 513
- SMILES order match: True
- Molecule Name order match: True

## Anchor Shift

- Pearson vs anchor: 0.887243
- Spearman vs anchor: 0.917840
- mean shift: -0.035283
- mean abs shift: 0.239808
- p90 abs shift: 0.615996
- max abs shift: 2.875959
- |shift| > 0.05: 387
- |shift| > 0.10: 260
- |shift| > 0.20: 173

## Prediction Distribution

- anchor mean/std: 4.798350 / 0.770987
- candidate mean/std: 4.763067 / 0.896623

## Known Bad Axis

| label           |   pearson |   spearman |   candidate_projection |
|:----------------|----------:|-----------:|-----------------------:|
| id56_minus_id55 | -0.022727 |  -0.082906 |              -0.113174 |
| id56_minus_id51 |  0.005150 |  -0.067846 |               0.048860 |

## Experiment Metadata

No matching `experiment_summary` row found for this CSV path.

## Reasons

- large_anchor_shift
- extreme_single_compound_shift
- prediction_scale_changed
- rank_order_changed

# Submission Preflight: `ens_conservative_probe_simplex_mae_anchor_l2_0p1.csv`

Verdict: **HOLD**

## Inputs

- candidate: `track1_activity/submissions/ens_conservative_probe_simplex_mae_anchor_l2_0p1.csv`
- anchor: `track1_activity/submissions/ens_id51_top500_potent46_t40_soft_g35.csv`

## CSV Sanity

- rows: 513 / 513
- SMILES order match: True
- Molecule Name order match: True

## Anchor Shift

- Pearson vs anchor: 0.992123
- Spearman vs anchor: 0.991467
- mean shift: -0.060667
- mean abs shift: 0.097531
- p90 abs shift: 0.182756
- max abs shift: 0.399183
- |shift| > 0.05: 377
- |shift| > 0.10: 217
- |shift| > 0.20: 43

## Prediction Distribution

- anchor mean/std: 4.798350 / 0.770987
- candidate mean/std: 4.737683 / 0.732774

## Known Bad Axis

| label           |   pearson |   spearman |   candidate_projection |
|:----------------|----------:|-----------:|-----------------------:|
| id56_minus_id55 |  0.824237 |   0.793483 |               1.216583 |
| id56_minus_id51 |  0.764286 |   0.752939 |               1.255790 |

## Experiment Metadata

No matching `experiment_summary` row found for this CSV path.

## Reasons

- large_anchor_shift
- extreme_single_compound_shift
- aligned_with_known_bad_axis
- rank_order_changed

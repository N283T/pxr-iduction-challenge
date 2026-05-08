# Submission Preflight: `ens_id51_top500_potent46_t40_soft_g50.csv`

Verdict: **PASS**

## Inputs

- candidate: `track1_activity/submissions/ens_id51_top500_potent46_t40_soft_g50.csv`
- anchor: `track1_activity/submissions/ens_id51_top500_potent46_t40_soft_g35.csv`

## CSV Sanity

- rows: 513 / 513
- SMILES order match: True
- Molecule Name order match: True

## Anchor Shift

- Pearson vs anchor: 0.999915
- Spearman vs anchor: 0.999873
- mean shift: +0.000460
- mean abs shift: 0.005642
- p90 abs shift: 0.016707
- max abs shift: 0.067850
- |shift| > 0.05: 3
- |shift| > 0.10: 0
- |shift| > 0.20: 0

## Prediction Distribution

- anchor mean/std: 4.798350 / 0.770987
- candidate mean/std: 4.798810 / 0.774569

## Known Bad Axis

| label           |   pearson |   spearman |   candidate_projection |
|:----------------|----------:|-----------:|-----------------------:|
| id56_minus_id55 | -0.464488 |  -0.267353 |              -0.070314 |
| id56_minus_id51 | -0.125479 |   0.007009 |              -0.021417 |

## Experiment Metadata

No matching `experiment_summary` row found for this CSV path.

## Reasons

- small_anchor_shift

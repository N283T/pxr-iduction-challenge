# Submission Preflight: `ens_id55_shap_pos_familygap_highsoftq20_g25.csv`

Verdict: **PASS**

## Inputs

- candidate: `track1_activity/submissions/ens_id55_shap_pos_familygap_highsoftq20_g25.csv`
- anchor: `track1_activity/submissions/ens_id51_top500_potent46_t40_soft_g35.csv`

## CSV Sanity

- rows: 513 / 513
- SMILES order match: True
- Molecule Name order match: True

## Anchor Shift

- Pearson vs anchor: 0.999815
- Spearman vs anchor: 0.999636
- mean shift: +0.009661
- mean abs shift: 0.009661
- p90 abs shift: 0.032535
- max abs shift: 0.086024
- |shift| > 0.05: 18
- |shift| > 0.10: 0
- |shift| > 0.20: 0

## Prediction Distribution

- anchor mean/std: 4.798350 / 0.770987
- candidate mean/std: 4.808011 / 0.776782

## Known Bad Axis

| label           |   pearson |   spearman |   candidate_projection |
|:----------------|----------:|-----------:|-----------------------:|
| id56_minus_id55 |  0.349711 |   0.495361 |               0.073448 |
| id56_minus_id51 |  0.535051 |   0.668179 |               0.131056 |

## Experiment Metadata

No matching `experiment_summary` row found for this CSV path.

## Reasons

- small_anchor_shift

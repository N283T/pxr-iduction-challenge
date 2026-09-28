# Submission Preflight: `id63_plus_pairwise_low_pooled_q050_s0p03_as2only.csv`

Verdict: **PASS**

## Inputs

- candidate: `track1_activity/analysis/boltz_affinity_embedding_probe/outputs_pairwise_gate/safe_low_gate_scan/id63_plus_pairwise_low_pooled_q050_s0p03_as2only.csv`
- anchor: `track1_activity/submissions/phase2_as1_aug_top500_id55blend_a0p45_pairrankchembl_q95_g0p15_plus_combo_new_h0p15_l0p15_labels_as1.csv`

## CSV Sanity

- rows: 513 / 513
- SMILES order match: True
- Molecule Name order match: True

## Anchor Shift

- Pearson vs anchor: 0.999985
- Spearman vs anchor: 0.999996
- mean shift: -0.000877
- mean abs shift: 0.000877
- p90 abs shift: 0.000000
- max abs shift: 0.030000
- |shift| > 0.05: 0
- |shift| > 0.10: 0
- |shift| > 0.20: 0

## Prediction Distribution

- anchor mean/std: 4.768881 / 0.901369
- candidate mean/std: 4.768004 / 0.902695

## Known Bad Axis

| label           |   pearson |   spearman |   candidate_projection |
|:----------------|----------:|-----------:|-----------------------:|
| id56_minus_id55 |  0.010271 |  -0.010266 |               0.001220 |
| id56_minus_id51 |  0.014955 |  -0.012050 |               0.001572 |

## Experiment Metadata

No matching `experiment_summary` row found for this CSV path.

## Reasons

- small_anchor_shift

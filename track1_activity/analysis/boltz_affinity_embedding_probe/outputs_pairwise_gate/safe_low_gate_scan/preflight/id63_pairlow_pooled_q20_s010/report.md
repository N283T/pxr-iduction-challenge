# Submission Preflight: `id63_plus_pairwise_low_pooled_q200_s0p1_as2only.csv`

Verdict: **PASS**

## Inputs

- candidate: `track1_activity/analysis/boltz_affinity_embedding_probe/outputs_pairwise_gate/safe_low_gate_scan/id63_plus_pairwise_low_pooled_q200_s0p1_as2only.csv`
- anchor: `track1_activity/submissions/phase2_as1_aug_top500_id55blend_a0p45_pairrankchembl_q95_g0p15_plus_combo_new_h0p15_l0p15_labels_as1.csv`

## CSV Sanity

- rows: 513 / 513
- SMILES order match: True
- Molecule Name order match: True

## Anchor Shift

- Pearson vs anchor: 0.999562
- Spearman vs anchor: 0.999765
- mean shift: -0.008967
- mean abs shift: 0.008967
- p90 abs shift: 0.000000
- max abs shift: 0.100000
- |shift| > 0.05: 46
- |shift| > 0.10: 27
- |shift| > 0.20: 0

## Prediction Distribution

- anchor mean/std: 4.768881 / 0.901369
- candidate mean/std: 4.759914 / 0.911269

## Known Bad Axis

| label           |   pearson |   spearman |   candidate_projection |
|:----------------|----------:|-----------:|-----------------------:|
| id56_minus_id55 |  0.020108 |  -0.012809 |               0.013089 |
| id56_minus_id51 |  0.039300 |   0.003584 |               0.021612 |

## Experiment Metadata

No matching `experiment_summary` row found for this CSV path.

## Reasons

- small_anchor_shift

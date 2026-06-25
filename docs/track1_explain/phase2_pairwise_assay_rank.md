# Phase 2 pairwise assay-rank probe

Prototype of Boltz-style same-assay pairwise learning. The model is trained
to classify which compound is stronger within the same assay, then challenge
compounds are scored by pairwise comparison to assay reference compounds.

## Config

- Sources: `chembl,htchem,single_conc`
- ChEMBL scope: `all_pxr`
- Min assay n/std/unique: `5` / `0.25` / `5`
- Max pairs per assay: `1500`
- Score top-k refs: `64`

## Assay Sources

| source      |   assays |   rows |    mean_n |
|:------------|---------:|-------:|----------:|
| chembl      |       50 |   1127 |   22.5400 |
| htchem      |        2 |    441 |  220.5000 |
| single_conc |        3 |  20985 | 6995.0000 |

## Pair Training Rows

| source      |   assays |   training_rows |   mean_compounds |
|:------------|---------:|----------------:|-----------------:|
| chembl      |       50 |           27472 |          22.5400 |
| htchem      |        2 |            6000 |         220.5000 |
| single_conc |        3 |            9000 |        6995.0000 |

## Score Evaluation

| score                | slice        |    n |   spearman_pec50 |   mean_score |   gte6_auc |   gte6_ap |   lt3_auc |   lt3_ap |
|:---------------------|:-------------|-----:|-----------------:|-------------:|-----------:|----------:|----------:|---------:|
| pairrank_htchem      | all          | 4393 |           0.4962 |       0.4275 |     0.6675 |    0.0621 |    0.8165 |   0.4788 |
| pairrank_single_conc | all          | 4393 |           0.4453 |       0.5556 |     0.6423 |    0.0410 |    0.7795 |   0.4174 |
| pairrank_chembl      | all          | 4393 |           0.3847 |       0.3481 |     0.6470 |    0.0370 |    0.7382 |   0.3101 |
| pairrank_all         | all          | 4393 |           0.4449 |       0.5478 |     0.6287 |    0.0291 |    0.7806 |   0.4117 |
| pairrank_chembl      | source_as1   |  253 |           0.3484 |       0.3516 |     0.8169 |    0.3144 |    0.5853 |   0.1433 |
| pairrank_htchem      | source_as1   |  253 |           0.5013 |       0.4825 |     0.7909 |    0.3092 |    0.7029 |   0.2743 |
| pairrank_all         | source_as1   |  253 |           0.3957 |       0.5594 |     0.7712 |    0.1756 |    0.6416 |   0.2370 |
| pairrank_single_conc | source_as1   |  253 |           0.3996 |       0.5741 |     0.7642 |    0.1617 |    0.6545 |   0.2500 |
| pairrank_htchem      | source_train | 4140 |           0.4941 |       0.4242 |     0.6469 |    0.0515 |    0.8192 |   0.4860 |
| pairrank_single_conc | source_train | 4140 |           0.4485 |       0.5545 |     0.6246 |    0.0381 |    0.7829 |   0.4255 |
| pairrank_chembl      | source_train | 4140 |           0.3894 |       0.3479 |     0.6214 |    0.0283 |    0.7440 |   0.3209 |
| pairrank_all         | source_train | 4140 |           0.4487 |       0.5471 |     0.6087 |    0.0250 |    0.7848 |   0.4212 |

## Best AS1 Gate Rows

| score           | mode      |   quantile |   threshold |   shift |   as1_mae |   n_flags |   n_true_high_flags |   n_true_low_flags |
|:----------------|:----------|-----------:|------------:|--------:|----------:|----------:|--------------------:|-------------------:|
| pairrank_chembl | high_lift |     0.9500 |      0.5750 |  0.3000 |    0.4014 |        13 |                   5 |                  0 |
| pairrank_chembl | high_lift |     0.9500 |      0.5750 |  0.2000 |    0.4017 |        13 |                   5 |                  0 |
| pairrank_chembl | high_lift |     0.9500 |      0.5750 |  0.1500 |    0.4024 |        13 |                   5 |                  0 |
| pairrank_htchem | high_lift |     0.9500 |      0.6936 |  0.2000 |    0.4034 |        13 |                   3 |                  0 |
| pairrank_htchem | high_lift |     0.9500 |      0.6936 |  0.1500 |    0.4034 |        13 |                   3 |                  0 |
| pairrank_chembl | high_lift |     0.9500 |      0.5750 |  0.1000 |    0.4034 |        13 |                   5 |                  0 |
| pairrank_htchem | high_lift |     0.9500 |      0.6936 |  0.1000 |    0.4040 |        13 |                   3 |                  0 |
| pairrank_htchem | high_lift |     0.9000 |      0.6669 |  0.1500 |    0.4045 |        26 |                   6 |                  1 |
| pairrank_htchem | high_lift |     0.9000 |      0.6669 |  0.1000 |    0.4045 |        26 |                   6 |                  1 |
| pairrank_htchem | high_lift |     0.9500 |      0.6936 |  0.3000 |    0.4047 |        13 |                   3 |                  0 |
| pairrank_chembl | high_lift |     0.9500 |      0.5750 |  0.0500 |    0.4048 |        13 |                   5 |                  0 |
| pairrank_htchem | high_lift |     0.9000 |      0.6669 |  0.0500 |    0.4049 |        26 |                   6 |                  1 |
| pairrank_htchem | high_lift |     0.9500 |      0.6936 |  0.0500 |    0.4050 |        13 |                   3 |                  0 |
| pairrank_htchem | high_lift |     0.9000 |      0.6669 |  0.2000 |    0.4050 |        26 |                   6 |                  1 |
| pairrank_htchem | high_lift |     0.8500 |      0.6386 |  0.0500 |    0.4055 |        38 |                   7 |                  3 |
| pairrank_chembl | low_drop  |     0.1000 |      0.1732 | -0.1000 |    0.4060 |        26 |                   0 |                  4 |
| pairrank_chembl | low_drop  |     0.1000 |      0.1732 | -0.1500 |    0.4060 |        26 |                   0 |                  4 |
| pairrank_chembl | low_drop  |     0.1000 |      0.1732 | -0.2000 |    0.4061 |        26 |                   0 |                  4 |
| pairrank_chembl | low_drop  |     0.1000 |      0.1732 | -0.0500 |    0.4062 |        26 |                   0 |                  4 |
| pairrank_htchem | high_lift |     0.8500 |      0.6386 |  0.1000 |    0.4063 |        38 |                   7 |                  3 |

## Read

This is a diagnostic scalar. A useful result would show stronger AS1 gte6 AP
or a gate that improves id55 AS1 without broad shifts.

## Generated Files

- `track1_activity/analysis/phase2_classifier_gate/outputs/pairwise_assay_rank/all_pxr_chembl_htchem_single_conc_mpa1500_top64/pool_pairrank_scores.csv`
- `track1_activity/analysis/phase2_classifier_gate/outputs/pairwise_assay_rank/all_pxr_chembl_htchem_single_conc_mpa1500_top64/test_pairrank_scores.csv`
- `track1_activity/analysis/phase2_classifier_gate/outputs/pairwise_assay_rank/all_pxr_chembl_htchem_single_conc_mpa1500_top64/score_summary.csv`
- `track1_activity/analysis/phase2_classifier_gate/outputs/pairwise_assay_rank/all_pxr_chembl_htchem_single_conc_mpa1500_top64/gate_scan.csv`

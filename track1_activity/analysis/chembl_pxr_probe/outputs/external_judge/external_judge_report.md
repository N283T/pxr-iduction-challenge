# ChEMBL External Judge

Exact challenge train/test InChIKeys were excluded from the external ChEMBL
PXR activation reference before nearest-neighbor scoring.

## External Set

- raw filtered ChEMBL PXR activation molecules: 267
- exact challenge overlaps removed: 12
- external molecules after exclusion: 255

## Coverage

| split   |    n |   exact_after_exclusion |   nn_ge_0.25 |   nn_ge_0.30 |   nn_ge_0.35 |   nn_ge_0.40 |   nn_max |   nn_median |
|:--------|-----:|------------------------:|-------------:|-------------:|-------------:|-------------:|---------:|------------:|
| train   | 4140 |                       0 |          790 |          161 |           55 |           36 |   1.0000 |      0.2154 |
| test    |  513 |                       0 |          121 |           27 |            4 |            0 |   0.3939 |      0.2222 |

## Train Signal

|   threshold |         n |   frac |   nn_sim_median |   pec50_mean |   external_pchembl_mean |   pearson_pec50_vs_external |   spearman_pec50_vs_external |   mae_direct_external |   centered_mae_external |
|------------:|----------:|-------:|----------------:|-------------:|------------------------:|----------------------------:|-----------------------------:|----------------------:|------------------------:|
|      0.0000 | 4140.0000 | 1.0000 |          0.2154 |       4.3208 |                  5.3236 |                      0.0566 |                       0.0936 |                1.2181 |                  1.0212 |
|      0.2000 | 2967.0000 | 0.7167 |          0.2283 |       4.4287 |                  5.3506 |                      0.0531 |                       0.0924 |                1.1419 |                  0.9699 |
|      0.2500 |  790.0000 | 0.1908 |          0.2701 |       4.4039 |                  5.4486 |                      0.0705 |                       0.1559 |                1.2461 |                  1.0229 |
|      0.3000 |  161.0000 | 0.0389 |          0.3281 |       4.2632 |                  5.4848 |                      0.1215 |                       0.2391 |                1.4090 |                  1.1303 |
|      0.3500 |   55.0000 | 0.0133 |          0.4483 |       4.1966 |                  5.5788 |                     -0.0204 |                       0.1400 |                1.6200 |                  1.3324 |
|      0.4000 |   36.0000 | 0.0087 |          0.5317 |       3.9621 |                  5.5797 |                     -0.2102 |                       0.0758 |                1.8865 |                  1.5946 |

## Candidate Judge at Tanimoto >= 0.30

| candidate                  | anchor        |   threshold |   n |   pred_mean |   external_mean |   pred_external_spearman |   pred_external_pearson |   direct_mae_to_external |   centered_mae_to_external |   direct_mae_delta_vs_anchor |   centered_mae_delta_vs_anchor |   mean_shift_vs_anchor |   mean_abs_shift_vs_anchor |   alignment_dot_vs_anchor |
|:---------------------------|:--------------|------------:|----:|------------:|----------------:|-------------------------:|------------------------:|-------------------------:|---------------------------:|-----------------------------:|-------------------------------:|-----------------------:|---------------------------:|--------------------------:|
| id55_soft_g35              | id57_soft_g50 |     0.30000 |  27 |     4.66575 |         5.86407 |                  0.23358 |                 0.05695 |                  1.21908 |                    1.00301 |                     -0.00083 |                       -0.00150 |                0.00083 |                    0.00188 |                   0.00244 |
| id57_soft_g50              | id57_soft_g50 |     0.30000 |  27 |     4.66491 |         5.86407 |                  0.23358 |                 0.05680 |                  1.21991 |                    1.00451 |                      0.00000 |                        0.00000 |                0.00000 |                    0.00000 |                   0.00000 |
| shap_numrings_pos_g50      | id57_soft_g50 |     0.30000 |  27 |     4.66965 |         5.86407 |                  0.23733 |                 0.05687 |                  1.21661 |                    1.00460 |                     -0.00331 |                        0.00008 |                0.00473 |                    0.00473 |                   0.00468 |
| combo_rank2_numrings       | id57_soft_g50 |     0.30000 |  27 |     4.66965 |         5.86407 |                  0.23733 |                 0.05687 |                  1.21661 |                    1.00460 |                     -0.00331 |                        0.00008 |                0.00473 |                    0.00473 |                   0.00468 |
| id58_combo_rank1           | id57_soft_g50 |     0.30000 |  27 |     4.66922 |         5.86407 |                  0.23733 |                 0.05680 |                  1.21661 |                    1.00462 |                     -0.00330 |                        0.00010 |                0.00430 |                    0.00486 |                   0.00431 |
| combo_rank3_familygap      | id57_soft_g50 |     0.30000 |  27 |     4.66922 |         5.86407 |                  0.23733 |                 0.05652 |                  1.21696 |                    1.00470 |                     -0.00296 |                        0.00019 |                0.00430 |                    0.00535 |                   0.00408 |
| log2fc_gate_optuna_q60_g50 | id57_soft_g50 |     0.30000 |  27 |     4.66851 |         5.86407 |                  0.23733 |                 0.05676 |                  1.21632 |                    1.00486 |                     -0.00360 |                        0.00034 |                0.00360 |                    0.00438 |                   0.00367 |

## Interpretation

Coverage is intentionally reported before using this as a decision aid.
If test coverage is sparse or train correlation is weak, ChEMBL should be
treated as a qualitative warning light rather than a submission gate.

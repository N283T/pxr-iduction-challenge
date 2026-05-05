# Selectivity Axis Report

Internal-only axis using counter-assay-derived selectivity labels.

## Auxiliary Predictor Diagnostics

|   counter_active_auc |   nonselective_auc_active_rows |   selectivity_delta_mae_active_rows |   n_counter_active |   n_nonselective |
|---------------------:|-------------------------------:|------------------------------------:|-------------------:|-----------------:|
|             0.861529 |                       0.636362 |                            0.996295 |        2648.000000 |       315.000000 |

## Residual Correction Diagnostics

|   raw_corr_std_train |   raw_corr_std_test |   raw_corr_residual_r |
|---------------------:|--------------------:|----------------------:|
|             0.050411 |            0.021997 |              0.066006 |

## Candidate Summary

| name                                   | path                                                                   | mode      |   shrink |     clip |   oof_mae |   oof_delta_mae |   oof_spearman |   oof_delta_spearman |   test_mean_shift |   test_mean_abs_shift |   test_p90_abs_shift |   test_max_abs_shift |   projection_on_id50_direction |   pearson_vs_id48 |
|:---------------------------------------|:-----------------------------------------------------------------------|:----------|---------:|---------:|----------:|----------------:|---------------:|---------------------:|------------------:|----------------------:|---------------------:|---------------------:|-------------------------------:|------------------:|
| ens_selectivity_axis_raw_s30_c05       | track1_activity/submissions/ens_selectivity_axis_raw_s30_c05.csv       | raw       | 0.300000 | 0.050000 |  0.398357 |       -0.000696 |       0.842391 |            -0.000134 |         -0.012616 |              0.012936 |             0.015000 |             0.015000 |                       0.306470 |          0.999983 |
| ens_selectivity_axis_raw_s20_c05       | track1_activity/submissions/ens_selectivity_axis_raw_s20_c05.csv       | raw       | 0.200000 | 0.050000 |  0.398546 |       -0.000508 |       0.842433 |            -0.000092 |         -0.008411 |              0.008624 |             0.010000 |             0.010000 |                       0.204313 |          0.999992 |
| ens_selectivity_axis_raw_s30_c03       | track1_activity/submissions/ens_selectivity_axis_raw_s30_c03.csv       | raw       | 0.300000 | 0.030000 |  0.398581 |       -0.000473 |       0.842460 |            -0.000065 |         -0.008214 |              0.008484 |             0.009000 |             0.009000 |                       0.203678 |          0.999994 |
| ens_selectivity_axis_centered_s30_c05  | track1_activity/submissions/ens_selectivity_axis_centered_s30_c05.csv  | centered  | 0.300000 | 0.050000 |  0.398626 |       -0.000428 |       0.842315 |            -0.000210 |         -0.000188 |              0.004777 |             0.009087 |             0.015000 |                      -0.000725 |          0.999970 |
| ens_selectivity_axis_anti_id50_s30_c05 | track1_activity/submissions/ens_selectivity_axis_anti_id50_s30_c05.csv | anti_id50 | 0.300000 | 0.050000 |  0.398626 |       -0.000428 |       0.842315 |            -0.000210 |         -0.000209 |              0.004780 |             0.009080 |             0.015000 |                       0.000354 |          0.999970 |
| ens_selectivity_axis_centered_s30_c03  | track1_activity/submissions/ens_selectivity_axis_centered_s30_c03.csv  | centered  | 0.300000 | 0.030000 |  0.398669 |       -0.000384 |       0.842367 |            -0.000158 |         -0.000479 |              0.004399 |             0.009000 |             0.009000 |                       0.003216 |          0.999978 |
| ens_selectivity_axis_anti_id50_s30_c03 | track1_activity/submissions/ens_selectivity_axis_anti_id50_s30_c03.csv | anti_id50 | 0.300000 | 0.030000 |  0.398669 |       -0.000384 |       0.842367 |            -0.000158 |         -0.000498 |              0.004402 |             0.009000 |             0.009000 |                       0.004241 |          0.999978 |
| ens_selectivity_axis_raw_s20_c03       | track1_activity/submissions/ens_selectivity_axis_raw_s20_c03.csv       | raw       | 0.200000 | 0.030000 |  0.398725 |       -0.000328 |       0.842486 |            -0.000039 |         -0.005476 |              0.005656 |             0.006000 |             0.006000 |                       0.135785 |          0.999997 |
| ens_selectivity_axis_centered_s20_c05  | track1_activity/submissions/ens_selectivity_axis_centered_s20_c05.csv  | centered  | 0.200000 | 0.050000 |  0.398750 |       -0.000303 |       0.842387 |            -0.000138 |         -0.000125 |              0.003185 |             0.006058 |             0.010000 |                      -0.000484 |          0.999987 |
| ens_selectivity_axis_anti_id50_s20_c05 | track1_activity/submissions/ens_selectivity_axis_anti_id50_s20_c05.csv | anti_id50 | 0.200000 | 0.050000 |  0.398750 |       -0.000303 |       0.842387 |            -0.000138 |         -0.000139 |              0.003186 |             0.006053 |             0.010000 |                       0.000236 |          0.999987 |
| ens_selectivity_axis_raw_s10_c05       | track1_activity/submissions/ens_selectivity_axis_raw_s10_c05.csv       | raw       | 0.100000 | 0.050000 |  0.398783 |       -0.000271 |       0.842489 |            -0.000036 |         -0.004205 |              0.004312 |             0.005000 |             0.005000 |                       0.102157 |          0.999998 |
| ens_selectivity_axis_centered_s20_c03  | track1_activity/submissions/ens_selectivity_axis_centered_s20_c03.csv  | centered  | 0.200000 | 0.030000 |  0.398788 |       -0.000265 |       0.842422 |            -0.000103 |         -0.000319 |              0.002932 |             0.006000 |             0.006000 |                       0.002144 |          0.999990 |
| ens_selectivity_axis_anti_id50_s20_c03 | track1_activity/submissions/ens_selectivity_axis_anti_id50_s20_c03.csv | anti_id50 | 0.200000 | 0.030000 |  0.398788 |       -0.000265 |       0.842422 |            -0.000103 |         -0.000332 |              0.002935 |             0.006000 |             0.006000 |                       0.002828 |          0.999990 |
| ens_selectivity_axis_raw_s10_c03       | track1_activity/submissions/ens_selectivity_axis_raw_s10_c03.csv       | raw       | 0.100000 | 0.030000 |  0.398881 |       -0.000172 |       0.842511 |            -0.000015 |         -0.002738 |              0.002828 |             0.003000 |             0.003000 |                       0.067893 |          0.999999 |
| ens_selectivity_axis_centered_s10_c05  | track1_activity/submissions/ens_selectivity_axis_centered_s10_c05.csv  | centered  | 0.100000 | 0.050000 |  0.398893 |       -0.000160 |       0.842459 |            -0.000066 |         -0.000063 |              0.001592 |             0.003029 |             0.005000 |                      -0.000242 |          0.999997 |
| ens_selectivity_axis_anti_id50_s10_c05 | track1_activity/submissions/ens_selectivity_axis_anti_id50_s10_c05.csv | anti_id50 | 0.100000 | 0.050000 |  0.398893 |       -0.000160 |       0.842459 |            -0.000066 |         -0.000070 |              0.001593 |             0.003027 |             0.005000 |                       0.000118 |          0.999997 |
| ens_selectivity_axis_centered_s10_c03  | track1_activity/submissions/ens_selectivity_axis_centered_s10_c03.csv  | centered  | 0.100000 | 0.030000 |  0.398914 |       -0.000140 |       0.842470 |            -0.000055 |         -0.000160 |              0.001466 |             0.003000 |             0.003000 |                       0.001072 |          0.999998 |
| ens_selectivity_axis_anti_id50_s10_c03 | track1_activity/submissions/ens_selectivity_axis_anti_id50_s10_c03.csv | anti_id50 | 0.100000 | 0.030000 |  0.398914 |       -0.000140 |       0.842470 |            -0.000055 |         -0.000166 |              0.001467 |             0.003000 |             0.003000 |                       0.001414 |          0.999998 |

## Read

Prefer candidates with negative OOF delta, non-negative Spearman delta,
test mean_abs_shift <= 0.02, and low projection on the failed id50 direction.

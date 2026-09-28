# Phase 2 candidate scorer

This is a diagnostic scorecard. It does not train models, change OOF
predictions, or generate a submission.

## Submission CSV summary

| candidate                               | path                                                                                                                                                 |   as1_mae |   as1_delta_mae_vs_anchor |   as1_bias_pred_minus_true |   as1_spearman |   test_pearson_vs_anchor |   test_mean_abs_shift |   test_p90_abs_shift |   test_max_abs_shift |   bad_axis_id56_projection |
|:----------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------|----------:|--------------------------:|---------------------------:|---------------:|-------------------------:|----------------------:|---------------------:|---------------------:|---------------------------:|
| combo_q98_lift03__cpabs01_lowq95_drop03 | track1_activity/analysis/phase2_classifier_gate/outputs/composite_pairrank_chemprop_train_quantile_gates/combo_q98_lift03__cpabs01_lowq95_drop03.csv |   0.39804 |                  -0.00852 |                    0.05396 |        0.85272 |                  0.99646 |               0.01520 |              0.00000 |              0.30000 |                    0.04446 |
| combo_q98_lift03__cpabs02_lowq95_drop03 | track1_activity/analysis/phase2_classifier_gate/outputs/composite_pairrank_chemprop_train_quantile_gates/combo_q98_lift03__cpabs02_lowq95_drop03.csv |   0.39851 |                  -0.00805 |                    0.04922 |        0.85295 |                  0.99635 |               0.01579 |              0.00000 |              0.30000 |                    0.03499 |
| combo_q98_lift02__cpabs01_lowq95_drop03 | track1_activity/analysis/phase2_classifier_gate/outputs/composite_pairrank_chemprop_train_quantile_gates/combo_q98_lift02__cpabs01_lowq95_drop03.csv |   0.39882 |                  -0.00774 |                    0.05041 |        0.85202 |                  0.99764 |               0.01209 |              0.00000 |              0.30000 |                    0.04857 |

## AS2 shift slices

| slice                               |   n |    frac |   mean_shift |   mean_abs_shift |   p90_abs_shift |   max_abs_shift |   n_abs_gt_005 |   n_abs_gt_010 | candidate                               |
|:------------------------------------|----:|--------:|-------------:|-----------------:|----------------:|----------------:|---------------:|---------------:|:----------------------------------------|
| all_test                            | 513 | 1.00000 |      0.00351 |          0.01520 |         0.00000 |         0.30000 |             26 |             26 | combo_q98_lift03__cpabs01_lowq95_drop03 |
| AS1                                 | 253 | 0.49318 |      0.00237 |          0.01897 |         0.00000 |         0.30000 |             16 |             16 | combo_q98_lift03__cpabs01_lowq95_drop03 |
| AS2                                 | 260 | 0.50682 |      0.00462 |          0.01154 |         0.00000 |         0.30000 |             10 |             10 | combo_q98_lift03__cpabs01_lowq95_drop03 |
| AS2_overall_risk_ge_0p80            |  26 | 0.05068 |      0.01154 |          0.01154 |         0.00000 |         0.30000 |              1 |              1 | combo_q98_lift03__cpabs01_lowq95_drop03 |
| AS2_tag_potent_neighbor_low_support |  87 | 0.16959 |      0.01724 |          0.02414 |         0.00000 |         0.30000 |              7 |              7 | combo_q98_lift03__cpabs01_lowq95_drop03 |
| AS2_tag_high_lf_saturated           |  37 | 0.07212 |      0.00811 |          0.00811 |         0.00000 |         0.30000 |              1 |              1 | combo_q98_lift03__cpabs01_lowq95_drop03 |
| AS2_tag_high_lf_but_not_high_pred   |  22 | 0.04288 |      0.00000 |          0.00000 |         0.00000 |         0.00000 |              0 |              0 | combo_q98_lift03__cpabs01_lowq95_drop03 |
| AS2_tag_member_disagreement         |  24 | 0.04678 |      0.01250 |          0.03750 |         0.21000 |         0.30000 |              3 |              3 | combo_q98_lift03__cpabs01_lowq95_drop03 |
| all_test                            | 513 | 1.00000 |      0.00292 |          0.01579 |         0.00000 |         0.30000 |             27 |             27 | combo_q98_lift03__cpabs02_lowq95_drop03 |
| AS1                                 | 253 | 0.49318 |     -0.00237 |          0.02372 |         0.00000 |         0.30000 |             20 |             20 | combo_q98_lift03__cpabs02_lowq95_drop03 |
| AS2                                 | 260 | 0.50682 |      0.00808 |          0.00808 |         0.00000 |         0.30000 |              7 |              7 | combo_q98_lift03__cpabs02_lowq95_drop03 |
| AS2_overall_risk_ge_0p80            |  26 | 0.05068 |      0.01154 |          0.01154 |         0.00000 |         0.30000 |              1 |              1 | combo_q98_lift03__cpabs02_lowq95_drop03 |
| AS2_tag_potent_neighbor_low_support |  87 | 0.16959 |      0.02069 |          0.02069 |         0.00000 |         0.30000 |              6 |              6 | combo_q98_lift03__cpabs02_lowq95_drop03 |
| AS2_tag_high_lf_saturated           |  37 | 0.07212 |      0.00811 |          0.00811 |         0.00000 |         0.30000 |              1 |              1 | combo_q98_lift03__cpabs02_lowq95_drop03 |
| AS2_tag_high_lf_but_not_high_pred   |  22 | 0.04288 |      0.00000 |          0.00000 |         0.00000 |         0.00000 |              0 |              0 | combo_q98_lift03__cpabs02_lowq95_drop03 |
| AS2_tag_member_disagreement         |  24 | 0.04678 |      0.02500 |          0.02500 |         0.00000 |         0.30000 |              2 |              2 | combo_q98_lift03__cpabs02_lowq95_drop03 |
| all_test                            | 513 | 1.00000 |      0.00039 |          0.01209 |         0.00000 |         0.30000 |             26 |             26 | combo_q98_lift02__cpabs01_lowq95_drop03 |
| AS1                                 | 253 | 0.49318 |     -0.00119 |          0.01542 |         0.00000 |         0.30000 |             16 |             16 | combo_q98_lift02__cpabs01_lowq95_drop03 |
| AS2                                 | 260 | 0.50682 |      0.00192 |          0.00885 |         0.00000 |         0.30000 |             10 |             10 | combo_q98_lift02__cpabs01_lowq95_drop03 |
| AS2_overall_risk_ge_0p80            |  26 | 0.05068 |      0.00769 |          0.00769 |         0.00000 |         0.20000 |              1 |              1 | combo_q98_lift02__cpabs01_lowq95_drop03 |
| AS2_tag_potent_neighbor_low_support |  87 | 0.16959 |      0.01034 |          0.01724 |         0.00000 |         0.30000 |              7 |              7 | combo_q98_lift02__cpabs01_lowq95_drop03 |
| AS2_tag_high_lf_saturated           |  37 | 0.07212 |      0.00541 |          0.00541 |         0.00000 |         0.20000 |              1 |              1 | combo_q98_lift02__cpabs01_lowq95_drop03 |
| AS2_tag_high_lf_but_not_high_pred   |  22 | 0.04288 |      0.00000 |          0.00000 |         0.00000 |         0.00000 |              0 |              0 | combo_q98_lift02__cpabs01_lowq95_drop03 |
| AS2_tag_member_disagreement         |  24 | 0.04678 |      0.00417 |          0.02917 |         0.14000 |         0.30000 |              3 |              3 | combo_q98_lift02__cpabs01_lowq95_drop03 |

## AS1 true-bin replay

| candidate                               | true_bin   |   n |     mae |   bias_pred_minus_true |   spearman |   pred_mean |   pred_std |   delta_mae_vs_anchor |
|:----------------------------------------|:-----------|----:|--------:|-----------------------:|-----------:|------------:|-----------:|----------------------:|
| combo_q98_lift03__cpabs01_lowq95_drop03 | lt3        |  24 | 1.13128 |                1.11115 |    0.12614 |     3.43032 |    0.63888 |              -0.01250 |
| combo_q98_lift03__cpabs01_lowq95_drop03 | 3to4       |  31 | 0.46491 |                0.10071 |    0.30326 |     3.66522 |    0.59552 |              -0.03871 |
| combo_q98_lift03__cpabs01_lowq95_drop03 | 4to5       |  86 | 0.34426 |                0.01177 |    0.48652 |     4.65578 |    0.50869 |               0.00281 |
| combo_q98_lift03__cpabs01_lowq95_drop03 | 5to6       | 102 | 0.24645 |               -0.12884 |    0.46452 |     5.28871 |    0.31358 |               0.00208 |
| combo_q98_lift03__cpabs01_lowq95_drop03 | gte6       |  10 | 0.43970 |               -0.40071 |   -0.27273 |     5.78779 |    0.34041 |              -0.11101 |
| combo_q98_lift03__cpabs02_lowq95_drop03 | lt3        |  24 | 1.14098 |                1.09865 |    0.08873 |     3.41782 |    0.64295 |              -0.00280 |
| combo_q98_lift03__cpabs02_lowq95_drop03 | 3to4       |  31 | 0.46491 |                0.08135 |    0.28995 |     3.64587 |    0.60703 |              -0.03871 |
| combo_q98_lift03__cpabs02_lowq95_drop03 | 4to5       |  86 | 0.34294 |                0.00828 |    0.48833 |     4.65229 |    0.51196 |               0.00149 |
| combo_q98_lift03__cpabs02_lowq95_drop03 | 5to6       | 102 | 0.24645 |               -0.12884 |    0.46452 |     5.28871 |    0.31358 |               0.00208 |
| combo_q98_lift03__cpabs02_lowq95_drop03 | gte6       |  10 | 0.43970 |               -0.40071 |   -0.27273 |     5.78779 |    0.34041 |              -0.11101 |
| combo_q98_lift02__cpabs01_lowq95_drop03 | lt3        |  24 | 1.13128 |                1.11115 |    0.12614 |     3.43032 |    0.63888 |              -0.01250 |
| combo_q98_lift02__cpabs01_lowq95_drop03 | 3to4       |  31 | 0.46491 |                0.10071 |    0.30326 |     3.66522 |    0.59552 |              -0.03871 |
| combo_q98_lift02__cpabs01_lowq95_drop03 | 4to5       |  86 | 0.34426 |                0.01177 |    0.48652 |     4.65578 |    0.50869 |               0.00281 |
| combo_q98_lift02__cpabs01_lowq95_drop03 | 5to6       | 102 | 0.24645 |               -0.13276 |    0.45880 |     5.28479 |    0.30748 |               0.00208 |
| combo_q98_lift02__cpabs01_lowq95_drop03 | gte6       |  10 | 0.45946 |               -0.45071 |   -0.27273 |     5.73779 |    0.29903 |              -0.09126 |

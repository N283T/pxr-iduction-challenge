# Phase 2 candidate scorer

This is a diagnostic scorecard. It does not train models, change OOF
predictions, or generate a submission.

## Submission CSV summary

| candidate                               | path                                                                                                                                                 |   as1_mae |   as1_delta_mae_vs_anchor |   as1_bias_pred_minus_true |   as1_spearman |   test_pearson_vs_anchor |   test_mean_abs_shift |   test_p90_abs_shift |   test_max_abs_shift |   bad_axis_id56_projection |
|:----------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------|----------:|--------------------------:|---------------------------:|---------------:|-------------------------:|----------------------:|---------------------:|---------------------:|---------------------------:|
| combo_q98_lift03__cpabs01_lowq95_drop02 | track1_activity/analysis/phase2_classifier_gate/outputs/composite_pairrank_chemprop_train_quantile_gates/combo_q98_lift03__cpabs01_lowq95_drop02.csv |   0.39923 |                  -0.00734 |                    0.05673 |        0.85281 |                  0.99721 |               0.01326 |              0.00000 |              0.30000 |                    0.02553 |
| combo_q98_lift02__cpabs02_lowq95_drop02 | track1_activity/analysis/phase2_classifier_gate/outputs/composite_pairrank_chemprop_train_quantile_gates/combo_q98_lift02__cpabs02_lowq95_drop02.csv |   0.39974 |                  -0.00682 |                    0.05001 |        0.85227 |                  0.99835 |               0.01053 |              0.00000 |              0.20000 |                    0.02333 |
| combo_q98_lift02__cpabs01_lowq95_drop02 | track1_activity/analysis/phase2_classifier_gate/outputs/composite_pairrank_chemprop_train_quantile_gates/combo_q98_lift02__cpabs01_lowq95_drop02.csv |   0.40001 |                  -0.00656 |                    0.05317 |        0.85211 |                  0.99840 |               0.01014 |              0.00000 |              0.20000 |                    0.02964 |

## AS2 shift slices

| slice                               |   n |    frac |   mean_shift |   mean_abs_shift |   p90_abs_shift |   max_abs_shift |   n_abs_gt_005 |   n_abs_gt_010 | candidate                               |
|:------------------------------------|----:|--------:|-------------:|-----------------:|----------------:|----------------:|---------------:|---------------:|:----------------------------------------|
| all_test                            | 513 | 1.00000 |      0.00234 |          0.01014 |         0.00000 |         0.20000 |             26 |             26 | combo_q98_lift02__cpabs01_lowq95_drop02 |
| AS1                                 | 253 | 0.49318 |      0.00158 |          0.01265 |         0.00000 |         0.20000 |             16 |             16 | combo_q98_lift02__cpabs01_lowq95_drop02 |
| AS2                                 | 260 | 0.50682 |      0.00308 |          0.00769 |         0.00000 |         0.20000 |             10 |             10 | combo_q98_lift02__cpabs01_lowq95_drop02 |
| AS2_overall_risk_ge_0p80            |  26 | 0.05068 |      0.00769 |          0.00769 |         0.00000 |         0.20000 |              1 |              1 | combo_q98_lift02__cpabs01_lowq95_drop02 |
| AS2_tag_potent_neighbor_low_support |  87 | 0.16959 |      0.01149 |          0.01609 |         0.00000 |         0.20000 |              7 |              7 | combo_q98_lift02__cpabs01_lowq95_drop02 |
| AS2_tag_high_lf_saturated           |  37 | 0.07212 |      0.00541 |          0.00541 |         0.00000 |         0.20000 |              1 |              1 | combo_q98_lift02__cpabs01_lowq95_drop02 |
| AS2_tag_high_lf_but_not_high_pred   |  22 | 0.04288 |      0.00000 |          0.00000 |         0.00000 |         0.00000 |              0 |              0 | combo_q98_lift02__cpabs01_lowq95_drop02 |
| AS2_tag_member_disagreement         |  24 | 0.04678 |      0.00833 |          0.02500 |         0.14000 |         0.20000 |              3 |              3 | combo_q98_lift02__cpabs01_lowq95_drop02 |
| all_test                            | 513 | 1.00000 |      0.00546 |          0.01326 |         0.00000 |         0.30000 |             26 |             26 | combo_q98_lift03__cpabs01_lowq95_drop02 |
| AS1                                 | 253 | 0.49318 |      0.00514 |          0.01621 |         0.00000 |         0.30000 |             16 |             16 | combo_q98_lift03__cpabs01_lowq95_drop02 |
| AS2                                 | 260 | 0.50682 |      0.00577 |          0.01038 |         0.00000 |         0.30000 |             10 |             10 | combo_q98_lift03__cpabs01_lowq95_drop02 |
| AS2_overall_risk_ge_0p80            |  26 | 0.05068 |      0.01154 |          0.01154 |         0.00000 |         0.30000 |              1 |              1 | combo_q98_lift03__cpabs01_lowq95_drop02 |
| AS2_tag_potent_neighbor_low_support |  87 | 0.16959 |      0.01839 |          0.02299 |         0.00000 |         0.30000 |              7 |              7 | combo_q98_lift03__cpabs01_lowq95_drop02 |
| AS2_tag_high_lf_saturated           |  37 | 0.07212 |      0.00811 |          0.00811 |         0.00000 |         0.30000 |              1 |              1 | combo_q98_lift03__cpabs01_lowq95_drop02 |
| AS2_tag_high_lf_but_not_high_pred   |  22 | 0.04288 |      0.00000 |          0.00000 |         0.00000 |         0.00000 |              0 |              0 | combo_q98_lift03__cpabs01_lowq95_drop02 |
| AS2_tag_member_disagreement         |  24 | 0.04678 |      0.01667 |          0.03333 |         0.14000 |         0.30000 |              3 |              3 | combo_q98_lift03__cpabs01_lowq95_drop02 |
| all_test                            | 513 | 1.00000 |      0.00195 |          0.01053 |         0.00000 |         0.20000 |             27 |             27 | combo_q98_lift02__cpabs02_lowq95_drop02 |
| AS1                                 | 253 | 0.49318 |     -0.00158 |          0.01581 |         0.00000 |         0.20000 |             20 |             20 | combo_q98_lift02__cpabs02_lowq95_drop02 |
| AS2                                 | 260 | 0.50682 |      0.00538 |          0.00538 |         0.00000 |         0.20000 |              7 |              7 | combo_q98_lift02__cpabs02_lowq95_drop02 |
| AS2_overall_risk_ge_0p80            |  26 | 0.05068 |      0.00769 |          0.00769 |         0.00000 |         0.20000 |              1 |              1 | combo_q98_lift02__cpabs02_lowq95_drop02 |
| AS2_tag_potent_neighbor_low_support |  87 | 0.16959 |      0.01379 |          0.01379 |         0.00000 |         0.20000 |              6 |              6 | combo_q98_lift02__cpabs02_lowq95_drop02 |
| AS2_tag_high_lf_saturated           |  37 | 0.07212 |      0.00541 |          0.00541 |         0.00000 |         0.20000 |              1 |              1 | combo_q98_lift02__cpabs02_lowq95_drop02 |
| AS2_tag_high_lf_but_not_high_pred   |  22 | 0.04288 |      0.00000 |          0.00000 |         0.00000 |         0.00000 |              0 |              0 | combo_q98_lift02__cpabs02_lowq95_drop02 |
| AS2_tag_member_disagreement         |  24 | 0.04678 |      0.01667 |          0.01667 |         0.00000 |         0.20000 |              2 |              2 | combo_q98_lift02__cpabs02_lowq95_drop02 |

## AS1 true-bin replay

| candidate                               | true_bin   |   n |     mae |   bias_pred_minus_true |   spearman |   pred_mean |   pred_std |   delta_mae_vs_anchor |
|:----------------------------------------|:-----------|----:|--------:|-----------------------:|-----------:|------------:|-----------:|----------------------:|
| combo_q98_lift02__cpabs01_lowq95_drop02 | lt3        |  24 | 1.13544 |                1.11532 |    0.12614 |     3.43449 |    0.63223 |              -0.00833 |
| combo_q98_lift02__cpabs01_lowq95_drop02 | 3to4       |  31 | 0.47782 |                0.11361 |    0.32907 |     3.67813 |    0.60248 |              -0.02581 |
| combo_q98_lift02__cpabs01_lowq95_drop02 | 4to5       |  86 | 0.34310 |                0.01293 |    0.48873 |     4.65694 |    0.50840 |               0.00165 |
| combo_q98_lift02__cpabs01_lowq95_drop02 | 5to6       | 102 | 0.24547 |               -0.13178 |    0.45746 |     5.28577 |    0.30589 |               0.00110 |
| combo_q98_lift02__cpabs01_lowq95_drop02 | gte6       |  10 | 0.45946 |               -0.45071 |   -0.27273 |     5.73779 |    0.29903 |              -0.09126 |
| combo_q98_lift03__cpabs01_lowq95_drop02 | lt3        |  24 | 1.13544 |                1.11532 |    0.12614 |     3.43449 |    0.63223 |              -0.00833 |
| combo_q98_lift03__cpabs01_lowq95_drop02 | 3to4       |  31 | 0.47782 |                0.11361 |    0.32907 |     3.67813 |    0.60248 |              -0.02581 |
| combo_q98_lift03__cpabs01_lowq95_drop02 | 4to5       |  86 | 0.34310 |                0.01293 |    0.48873 |     4.65694 |    0.50840 |               0.00165 |
| combo_q98_lift03__cpabs01_lowq95_drop02 | 5to6       | 102 | 0.24547 |               -0.12786 |    0.46318 |     5.28969 |    0.31202 |               0.00110 |
| combo_q98_lift03__cpabs01_lowq95_drop02 | gte6       |  10 | 0.43970 |               -0.40071 |   -0.27273 |     5.78779 |    0.34041 |              -0.11101 |
| combo_q98_lift02__cpabs02_lowq95_drop02 | lt3        |  24 | 1.14098 |                1.10699 |    0.09134 |     3.42615 |    0.63437 |              -0.00280 |
| combo_q98_lift02__cpabs02_lowq95_drop02 | 3to4       |  31 | 0.47782 |                0.10071 |    0.31939 |     3.66522 |    0.60935 |              -0.02581 |
| combo_q98_lift02__cpabs02_lowq95_drop02 | 4to5       |  86 | 0.34077 |                0.01060 |    0.49016 |     4.65461 |    0.51036 |              -0.00068 |
| combo_q98_lift02__cpabs02_lowq95_drop02 | 5to6       | 102 | 0.24547 |               -0.13178 |    0.45746 |     5.28577 |    0.30589 |               0.00110 |
| combo_q98_lift02__cpabs02_lowq95_drop02 | gte6       |  10 | 0.45946 |               -0.45071 |   -0.27273 |     5.73779 |    0.29903 |              -0.09126 |

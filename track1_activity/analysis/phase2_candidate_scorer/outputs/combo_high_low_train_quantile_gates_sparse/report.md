# Phase 2 candidate scorer

This is a diagnostic scorecard. It does not train models, change OOF
predictions, or generate a submission.

## Submission CSV summary

| candidate                                 | path                                                                                                                                                   |   as1_mae |   as1_delta_mae_vs_anchor |   as1_bias_pred_minus_true |   as1_spearman |   test_pearson_vs_anchor |   test_mean_abs_shift |   test_p90_abs_shift |   test_max_abs_shift |   bad_axis_id56_projection |
|:------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------|----------:|--------------------------:|---------------------------:|---------------:|-------------------------:|----------------------:|---------------------:|---------------------:|---------------------------:|
| combo_q98_lift02__cp_abs01_lowq98_drop02  | track1_activity/analysis/phase2_classifier_gate/outputs/composite_pairrank_chemprop_train_quantile_gates/combo_q98_lift02__cp_abs01_lowq98_drop02.csv  |   0.40182 |                  -0.00475 |                    0.05792 |        0.85188 |                  0.99895 |               0.00663 |              0.00000 |              0.20000 |                    0.00049 |
| combo_q98_lift02__cp_abs01_lowq98_drop015 | track1_activity/analysis/phase2_classifier_gate/outputs/composite_pairrank_chemprop_train_quantile_gates/combo_q98_lift02__cp_abs01_lowq98_drop015.csv |   0.40202 |                  -0.00455 |                    0.05811 |        0.85187 |                  0.99898 |               0.00653 |              0.00000 |              0.20000 |                   -0.00169 |

## AS2 shift slices

| slice                               |   n |    frac |   mean_shift |   mean_abs_shift |   p90_abs_shift |   max_abs_shift |   n_abs_gt_005 |   n_abs_gt_010 | candidate                                 |
|:------------------------------------|----:|--------:|-------------:|-----------------:|----------------:|----------------:|---------------:|---------------:|:------------------------------------------|
| all_test                            | 513 | 1.00000 |      0.00585 |          0.00663 |         0.00000 |         0.20000 |             17 |             17 | combo_q98_lift02__cp_abs01_lowq98_drop02  |
| AS1                                 | 253 | 0.49318 |      0.00632 |          0.00791 |         0.00000 |         0.20000 |             10 |             10 | combo_q98_lift02__cp_abs01_lowq98_drop02  |
| AS2                                 | 260 | 0.50682 |      0.00538 |          0.00538 |         0.00000 |         0.20000 |              7 |              7 | combo_q98_lift02__cp_abs01_lowq98_drop02  |
| AS2_overall_risk_ge_0p80            |  26 | 0.05068 |      0.00769 |          0.00769 |         0.00000 |         0.20000 |              1 |              1 | combo_q98_lift02__cp_abs01_lowq98_drop02  |
| AS2_tag_potent_neighbor_low_support |  87 | 0.16959 |      0.01379 |          0.01379 |         0.00000 |         0.20000 |              6 |              6 | combo_q98_lift02__cp_abs01_lowq98_drop02  |
| AS2_tag_high_lf_saturated           |  37 | 0.07212 |      0.00541 |          0.00541 |         0.00000 |         0.20000 |              1 |              1 | combo_q98_lift02__cp_abs01_lowq98_drop02  |
| AS2_tag_high_lf_but_not_high_pred   |  22 | 0.04288 |      0.00000 |          0.00000 |         0.00000 |         0.00000 |              0 |              0 | combo_q98_lift02__cp_abs01_lowq98_drop02  |
| AS2_tag_member_disagreement         |  24 | 0.04678 |      0.01667 |          0.01667 |         0.00000 |         0.20000 |              2 |              2 | combo_q98_lift02__cp_abs01_lowq98_drop02  |
| all_test                            | 513 | 1.00000 |      0.00595 |          0.00653 |         0.00000 |         0.20000 |             17 |             17 | combo_q98_lift02__cp_abs01_lowq98_drop015 |
| AS1                                 | 253 | 0.49318 |      0.00652 |          0.00771 |         0.00000 |         0.20000 |             10 |             10 | combo_q98_lift02__cp_abs01_lowq98_drop015 |
| AS2                                 | 260 | 0.50682 |      0.00538 |          0.00538 |         0.00000 |         0.20000 |              7 |              7 | combo_q98_lift02__cp_abs01_lowq98_drop015 |
| AS2_overall_risk_ge_0p80            |  26 | 0.05068 |      0.00769 |          0.00769 |         0.00000 |         0.20000 |              1 |              1 | combo_q98_lift02__cp_abs01_lowq98_drop015 |
| AS2_tag_potent_neighbor_low_support |  87 | 0.16959 |      0.01379 |          0.01379 |         0.00000 |         0.20000 |              6 |              6 | combo_q98_lift02__cp_abs01_lowq98_drop015 |
| AS2_tag_high_lf_saturated           |  37 | 0.07212 |      0.00541 |          0.00541 |         0.00000 |         0.20000 |              1 |              1 | combo_q98_lift02__cp_abs01_lowq98_drop015 |
| AS2_tag_high_lf_but_not_high_pred   |  22 | 0.04288 |      0.00000 |          0.00000 |         0.00000 |         0.00000 |              0 |              0 | combo_q98_lift02__cp_abs01_lowq98_drop015 |
| AS2_tag_member_disagreement         |  24 | 0.04678 |      0.01667 |          0.01667 |         0.00000 |         0.20000 |              2 |              2 | combo_q98_lift02__cp_abs01_lowq98_drop015 |

## AS1 true-bin replay

| candidate                                 | true_bin   |   n |     mae |   bias_pred_minus_true |   spearman |   pred_mean |   pred_std |   delta_mae_vs_anchor |
|:------------------------------------------|:-----------|----:|--------:|-----------------------:|-----------:|------------:|-----------:|----------------------:|
| combo_q98_lift02__cp_abs01_lowq98_drop02  | lt3        |  24 | 1.14378 |                1.12365 |    0.09961 |     3.44282 |    0.62072 |               0.00000 |
| combo_q98_lift02__cp_abs01_lowq98_drop02  | 3to4       |  31 | 0.49717 |                0.13297 |    0.34399 |     3.69748 |    0.61549 |              -0.00645 |
| combo_q98_lift02__cp_abs01_lowq98_drop02  | 4to5       |  86 | 0.34145 |                0.01525 |    0.48875 |     4.65927 |    0.50850 |               0.00000 |
| combo_q98_lift02__cp_abs01_lowq98_drop02  | 5to6       | 102 | 0.24351 |               -0.12982 |    0.45408 |     5.28773 |    0.30367 |              -0.00086 |
| combo_q98_lift02__cp_abs01_lowq98_drop02  | gte6       |  10 | 0.45946 |               -0.45071 |   -0.27273 |     5.73779 |    0.29903 |              -0.09126 |
| combo_q98_lift02__cp_abs01_lowq98_drop015 | lt3        |  24 | 1.14378 |                1.12365 |    0.09961 |     3.44282 |    0.62072 |               0.00000 |
| combo_q98_lift02__cp_abs01_lowq98_drop015 | 3to4       |  31 | 0.49878 |                0.13458 |    0.34399 |     3.69910 |    0.61687 |              -0.00484 |
| combo_q98_lift02__cp_abs01_lowq98_drop015 | 4to5       |  86 | 0.34145 |                0.01525 |    0.48875 |     4.65927 |    0.50850 |               0.00000 |
| combo_q98_lift02__cp_abs01_lowq98_drop015 | 5to6       | 102 | 0.24351 |               -0.12982 |    0.45408 |     5.28773 |    0.30367 |              -0.00086 |
| combo_q98_lift02__cp_abs01_lowq98_drop015 | gte6       |  10 | 0.45946 |               -0.45071 |   -0.27273 |     5.73779 |    0.29903 |              -0.09126 |

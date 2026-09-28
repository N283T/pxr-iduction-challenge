# Phase 2 candidate scorer

This is a diagnostic scorecard. It does not train models, change OOF
predictions, or generate a submission.

## Submission CSV summary

| candidate               | path                                                                                                                            |   as1_mae |   as1_delta_mae_vs_anchor |   as1_bias_pred_minus_true |   as1_spearman |   test_pearson_vs_anchor |   test_mean_abs_shift |   test_p90_abs_shift |   test_max_abs_shift |   bad_axis_id56_projection |
|:------------------------|:--------------------------------------------------------------------------------------------------------------------------------|----------:|--------------------------:|---------------------------:|---------------:|-------------------------:|----------------------:|---------------------:|---------------------:|---------------------------:|
| test_residual_candidate | track1_activity/analysis/phase2_classifier_gate/outputs/composite_pairrank_chemprop_residual_cap010/test_residual_candidate.csv |   0.39635 |                  -0.01021 |                    0.08059 |        0.85712 |                  0.99855 |               0.04649 |              0.10000 |              0.10000 |                   -0.02713 |

## AS2 shift slices

| slice                               |   n |    frac |   mean_shift |   mean_abs_shift |   p90_abs_shift |   max_abs_shift |   n_abs_gt_005 |   n_abs_gt_010 | candidate               |
|:------------------------------------|----:|--------:|-------------:|-----------------:|----------------:|----------------:|---------------:|---------------:|:------------------------|
| all_test                            | 513 | 1.00000 |      0.02744 |          0.04649 |         0.10000 |         0.10000 |            208 |             13 | test_residual_candidate |
| AS1                                 | 253 | 0.49318 |      0.02900 |          0.04923 |         0.10000 |         0.10000 |            120 |              7 | test_residual_candidate |
| AS2                                 | 260 | 0.50682 |      0.02592 |          0.04382 |         0.09857 |         0.10000 |             88 |              6 | test_residual_candidate |
| AS2_overall_risk_ge_0p80            |  26 | 0.05068 |      0.05681 |          0.05732 |         0.10000 |         0.10000 |             17 |              0 | test_residual_candidate |
| AS2_tag_potent_neighbor_low_support |  87 | 0.16959 |      0.03696 |          0.04853 |         0.10000 |         0.10000 |             38 |              0 | test_residual_candidate |
| AS2_tag_high_lf_saturated           |  37 | 0.07212 |      0.05977 |          0.06013 |         0.10000 |         0.10000 |             22 |              0 | test_residual_candidate |
| AS2_tag_high_lf_but_not_high_pred   |  22 | 0.04288 |      0.03478 |          0.03864 |         0.07474 |         0.10000 |              7 |              0 | test_residual_candidate |
| AS2_tag_member_disagreement         |  24 | 0.04678 |     -0.00838 |          0.04540 |         0.09809 |         0.10000 |              9 |              2 | test_residual_candidate |

## AS1 true-bin replay

| candidate               | true_bin   |   n |     mae |   bias_pred_minus_true |   spearman |   pred_mean |   pred_std |   delta_mae_vs_anchor |
|:------------------------|:-----------|----:|--------:|-----------------------:|-----------:|------------:|-----------:|----------------------:|
| test_residual_candidate | lt3        |  24 | 1.12393 |                1.11000 |    0.08525 |     3.42916 |    0.62785 |              -0.01985 |
| test_residual_candidate | 3to4       |  31 | 0.47514 |                0.12623 |    0.36375 |     3.69075 |    0.59627 |              -0.02848 |
| test_residual_candidate | 4to5       |  86 | 0.34671 |                0.03831 |    0.51362 |     4.68232 |    0.52075 |               0.00526 |
| test_residual_candidate | 5to6       | 102 | 0.23466 |               -0.08468 |    0.46130 |     5.33287 |    0.30707 |              -0.00971 |
| test_residual_candidate | gte6       |  10 | 0.48208 |               -0.48208 |   -0.32121 |     5.70642 |    0.26244 |              -0.06863 |

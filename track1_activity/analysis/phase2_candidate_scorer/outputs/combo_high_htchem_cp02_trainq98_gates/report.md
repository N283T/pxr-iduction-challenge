# Phase 2 candidate scorer

This is a diagnostic scorecard. It does not train models, change OOF
predictions, or generate a submission.

## Submission CSV summary

| candidate                          | path                                                                                                                                            |   as1_mae |   as1_delta_mae_vs_anchor |   as1_bias_pred_minus_true |   as1_spearman |   test_pearson_vs_anchor |   test_mean_abs_shift |   test_p90_abs_shift |   test_max_abs_shift |   bad_axis_id56_projection |
|:-----------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------|----------:|--------------------------:|---------------------------:|---------------:|-------------------------:|----------------------:|---------------------:|---------------------:|---------------------------:|
| combo_high_htchem_cp02_q98_lift030 | track1_activity/analysis/phase2_classifier_gate/outputs/composite_pairrank_chemprop_train_quantile_gates/combo_high_htchem_cp02_q98_lift030.csv |   0.40183 |                  -0.00474 |                    0.06226 |        0.85245 |                  0.99781 |               0.00936 |              0.00000 |              0.30000 |                   -0.01232 |
| combo_high_htchem_cp02_q98_lift020 | track1_activity/analysis/phase2_classifier_gate/outputs/composite_pairrank_chemprop_train_quantile_gates/combo_high_htchem_cp02_q98_lift020.csv |   0.40261 |                  -0.00396 |                    0.05871 |        0.85176 |                  0.99902 |               0.00624 |              0.00000 |              0.20000 |                   -0.00821 |

## AS2 shift slices

| slice                               |   n |    frac |   mean_shift |   mean_abs_shift |   p90_abs_shift |   max_abs_shift |   n_abs_gt_005 |   n_abs_gt_010 | candidate                          |
|:------------------------------------|----:|--------:|-------------:|-----------------:|----------------:|----------------:|---------------:|---------------:|:-----------------------------------|
| all_test                            | 513 | 1.00000 |      0.00624 |          0.00624 |         0.00000 |         0.20000 |             16 |             16 | combo_high_htchem_cp02_q98_lift020 |
| AS1                                 | 253 | 0.49318 |      0.00711 |          0.00711 |         0.00000 |         0.20000 |              9 |              9 | combo_high_htchem_cp02_q98_lift020 |
| AS2                                 | 260 | 0.50682 |      0.00538 |          0.00538 |         0.00000 |         0.20000 |              7 |              7 | combo_high_htchem_cp02_q98_lift020 |
| AS2_overall_risk_ge_0p80            |  26 | 0.05068 |      0.00769 |          0.00769 |         0.00000 |         0.20000 |              1 |              1 | combo_high_htchem_cp02_q98_lift020 |
| AS2_tag_potent_neighbor_low_support |  87 | 0.16959 |      0.01379 |          0.01379 |         0.00000 |         0.20000 |              6 |              6 | combo_high_htchem_cp02_q98_lift020 |
| AS2_tag_high_lf_saturated           |  37 | 0.07212 |      0.00541 |          0.00541 |         0.00000 |         0.20000 |              1 |              1 | combo_high_htchem_cp02_q98_lift020 |
| AS2_tag_high_lf_but_not_high_pred   |  22 | 0.04288 |      0.00000 |          0.00000 |         0.00000 |         0.00000 |              0 |              0 | combo_high_htchem_cp02_q98_lift020 |
| AS2_tag_member_disagreement         |  24 | 0.04678 |      0.01667 |          0.01667 |         0.00000 |         0.20000 |              2 |              2 | combo_high_htchem_cp02_q98_lift020 |
| all_test                            | 513 | 1.00000 |      0.00936 |          0.00936 |         0.00000 |         0.30000 |             16 |             16 | combo_high_htchem_cp02_q98_lift030 |
| AS1                                 | 253 | 0.49318 |      0.01067 |          0.01067 |         0.00000 |         0.30000 |              9 |              9 | combo_high_htchem_cp02_q98_lift030 |
| AS2                                 | 260 | 0.50682 |      0.00808 |          0.00808 |         0.00000 |         0.30000 |              7 |              7 | combo_high_htchem_cp02_q98_lift030 |
| AS2_overall_risk_ge_0p80            |  26 | 0.05068 |      0.01154 |          0.01154 |         0.00000 |         0.30000 |              1 |              1 | combo_high_htchem_cp02_q98_lift030 |
| AS2_tag_potent_neighbor_low_support |  87 | 0.16959 |      0.02069 |          0.02069 |         0.00000 |         0.30000 |              6 |              6 | combo_high_htchem_cp02_q98_lift030 |
| AS2_tag_high_lf_saturated           |  37 | 0.07212 |      0.00811 |          0.00811 |         0.00000 |         0.30000 |              1 |              1 | combo_high_htchem_cp02_q98_lift030 |
| AS2_tag_high_lf_but_not_high_pred   |  22 | 0.04288 |      0.00000 |          0.00000 |         0.00000 |         0.00000 |              0 |              0 | combo_high_htchem_cp02_q98_lift030 |
| AS2_tag_member_disagreement         |  24 | 0.04678 |      0.02500 |          0.02500 |         0.00000 |         0.30000 |              2 |              2 | combo_high_htchem_cp02_q98_lift030 |

## AS1 true-bin replay

| candidate                          | true_bin   |   n |     mae |   bias_pred_minus_true |   spearman |   pred_mean |   pred_std |   delta_mae_vs_anchor |
|:-----------------------------------|:-----------|----:|--------:|-----------------------:|-----------:|------------:|-----------:|----------------------:|
| combo_high_htchem_cp02_q98_lift020 | lt3        |  24 | 1.14378 |                1.12365 |    0.09961 |     3.44282 |    0.62072 |               0.00000 |
| combo_high_htchem_cp02_q98_lift020 | 3to4       |  31 | 0.50362 |                0.13942 |    0.35326 |     3.70393 |    0.62178 |               0.00000 |
| combo_high_htchem_cp02_q98_lift020 | 4to5       |  86 | 0.34145 |                0.01525 |    0.48875 |     4.65927 |    0.50850 |               0.00000 |
| combo_high_htchem_cp02_q98_lift020 | 5to6       | 102 | 0.24351 |               -0.12982 |    0.45408 |     5.28773 |    0.30367 |              -0.00086 |
| combo_high_htchem_cp02_q98_lift020 | gte6       |  10 | 0.45946 |               -0.45071 |   -0.27273 |     5.73779 |    0.29903 |              -0.09126 |
| combo_high_htchem_cp02_q98_lift030 | lt3        |  24 | 1.14378 |                1.12365 |    0.09961 |     3.44282 |    0.62072 |               0.00000 |
| combo_high_htchem_cp02_q98_lift030 | 3to4       |  31 | 0.50362 |                0.13942 |    0.35326 |     3.70393 |    0.62178 |               0.00000 |
| combo_high_htchem_cp02_q98_lift030 | 4to5       |  86 | 0.34145 |                0.01525 |    0.48875 |     4.65927 |    0.50850 |               0.00000 |
| combo_high_htchem_cp02_q98_lift030 | 5to6       | 102 | 0.24351 |               -0.12590 |    0.45981 |     5.29165 |    0.30981 |              -0.00086 |
| combo_high_htchem_cp02_q98_lift030 | gte6       |  10 | 0.43970 |               -0.40071 |   -0.27273 |     5.78779 |    0.34041 |              -0.11101 |

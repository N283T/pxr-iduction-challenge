# Phase 2 candidate scorer

This is a diagnostic scorecard. It does not train models, change OOF
predictions, or generate a submission.

## Submission CSV summary

| candidate                   | path                                                                                                                                     |   as1_mae |   as1_delta_mae_vs_anchor |   as1_bias_pred_minus_true |   as1_spearman |   test_pearson_vs_anchor |   test_mean_abs_shift |   test_p90_abs_shift |   test_max_abs_shift |   bad_axis_id56_projection |
|:----------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------|----------:|--------------------------:|---------------------------:|---------------:|-------------------------:|----------------------:|---------------------:|---------------------:|---------------------------:|
| pairrank_chembl_q95_lift030 | track1_activity/analysis/phase2_classifier_gate/outputs/composite_pairrank_chemprop_train_quantile_gates/pairrank_chembl_q95_lift030.csv |   0.40106 |                  -0.00550 |                    0.06582 |        0.85379 |                  0.99458 |               0.02573 |              0.00000 |              0.30000 |                    0.01139 |
| pairrank_chembl_q95_lift020 | track1_activity/analysis/phase2_classifier_gate/outputs/composite_pairrank_chemprop_train_quantile_gates/pairrank_chembl_q95_lift020.csv |   0.40177 |                  -0.00480 |                    0.06108 |        0.85273 |                  0.99755 |               0.01715 |              0.00000 |              0.20000 |                    0.00759 |

## AS2 shift slices

| slice                               |   n |    frac |   mean_shift |   mean_abs_shift |   p90_abs_shift |   max_abs_shift |   n_abs_gt_005 |   n_abs_gt_010 | candidate                   |
|:------------------------------------|----:|--------:|-------------:|-----------------:|----------------:|----------------:|---------------:|---------------:|:----------------------------|
| all_test                            | 513 | 1.00000 |      0.01715 |          0.01715 |         0.00000 |         0.20000 |             44 |             44 | pairrank_chembl_q95_lift020 |
| AS1                                 | 253 | 0.49318 |      0.00949 |          0.00949 |         0.00000 |         0.20000 |             12 |             12 | pairrank_chembl_q95_lift020 |
| AS2                                 | 260 | 0.50682 |      0.02462 |          0.02462 |         0.20000 |         0.20000 |             32 |             32 | pairrank_chembl_q95_lift020 |
| AS2_overall_risk_ge_0p80            |  26 | 0.05068 |      0.01538 |          0.01538 |         0.00000 |         0.20000 |              2 |              2 | pairrank_chembl_q95_lift020 |
| AS2_tag_potent_neighbor_low_support |  87 | 0.16959 |      0.02299 |          0.02299 |         0.20000 |         0.20000 |             10 |             10 | pairrank_chembl_q95_lift020 |
| AS2_tag_high_lf_saturated           |  37 | 0.07212 |      0.03243 |          0.03243 |         0.20000 |         0.20000 |              6 |              6 | pairrank_chembl_q95_lift020 |
| AS2_tag_high_lf_but_not_high_pred   |  22 | 0.04288 |      0.01818 |          0.01818 |         0.00000 |         0.20000 |              2 |              2 | pairrank_chembl_q95_lift020 |
| AS2_tag_member_disagreement         |  24 | 0.04678 |      0.01667 |          0.01667 |         0.00000 |         0.20000 |              2 |              2 | pairrank_chembl_q95_lift020 |
| all_test                            | 513 | 1.00000 |      0.02573 |          0.02573 |         0.00000 |         0.30000 |             44 |             44 | pairrank_chembl_q95_lift030 |
| AS1                                 | 253 | 0.49318 |      0.01423 |          0.01423 |         0.00000 |         0.30000 |             12 |             12 | pairrank_chembl_q95_lift030 |
| AS2                                 | 260 | 0.50682 |      0.03692 |          0.03692 |         0.30000 |         0.30000 |             32 |             32 | pairrank_chembl_q95_lift030 |
| AS2_overall_risk_ge_0p80            |  26 | 0.05068 |      0.02308 |          0.02308 |         0.00000 |         0.30000 |              2 |              2 | pairrank_chembl_q95_lift030 |
| AS2_tag_potent_neighbor_low_support |  87 | 0.16959 |      0.03448 |          0.03448 |         0.30000 |         0.30000 |             10 |             10 | pairrank_chembl_q95_lift030 |
| AS2_tag_high_lf_saturated           |  37 | 0.07212 |      0.04865 |          0.04865 |         0.30000 |         0.30000 |              6 |              6 | pairrank_chembl_q95_lift030 |
| AS2_tag_high_lf_but_not_high_pred   |  22 | 0.04288 |      0.02727 |          0.02727 |         0.00000 |         0.30000 |              2 |              2 | pairrank_chembl_q95_lift030 |
| AS2_tag_member_disagreement         |  24 | 0.04678 |      0.02500 |          0.02500 |         0.00000 |         0.30000 |              2 |              2 | pairrank_chembl_q95_lift030 |

## AS1 true-bin replay

| candidate                   | true_bin   |   n |     mae |   bias_pred_minus_true |   spearman |   pred_mean |   pred_std |   delta_mae_vs_anchor |
|:----------------------------|:-----------|----:|--------:|-----------------------:|-----------:|------------:|-----------:|----------------------:|
| pairrank_chembl_q95_lift020 | lt3        |  24 | 1.14378 |                1.12365 |    0.09961 |     3.44282 |    0.62072 |               0.00000 |
| pairrank_chembl_q95_lift020 | 3to4       |  31 | 0.50362 |                0.13942 |    0.35326 |     3.70393 |    0.62178 |               0.00000 |
| pairrank_chembl_q95_lift020 | 4to5       |  86 | 0.34145 |                0.01990 |    0.49431 |     4.66392 |    0.51254 |               0.00000 |
| pairrank_chembl_q95_lift020 | 5to6       | 102 | 0.24143 |               -0.12786 |    0.46205 |     5.28969 |    0.29999 |              -0.00295 |
| pairrank_chembl_q95_lift020 | gte6       |  10 | 0.45946 |               -0.45071 |   -0.07879 |     5.73779 |    0.25193 |              -0.09126 |
| pairrank_chembl_q95_lift030 | lt3        |  24 | 1.14378 |                1.12365 |    0.09961 |     3.44282 |    0.62072 |               0.00000 |
| pairrank_chembl_q95_lift030 | 3to4       |  31 | 0.50362 |                0.13942 |    0.35326 |     3.70393 |    0.62178 |               0.00000 |
| pairrank_chembl_q95_lift030 | 4to5       |  86 | 0.34282 |                0.02223 |    0.49819 |     4.66624 |    0.51522 |               0.00137 |
| pairrank_chembl_q95_lift030 | 5to6       | 102 | 0.24046 |               -0.12296 |    0.46730 |     5.29459 |    0.30484 |              -0.00392 |
| pairrank_chembl_q95_lift030 | gte6       |  10 | 0.43970 |               -0.40071 |   -0.07879 |     5.78779 |    0.27740 |              -0.11101 |

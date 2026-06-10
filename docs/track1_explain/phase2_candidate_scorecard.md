# Phase 2 candidate scorer

This is a diagnostic scorecard. It does not train models, change OOF
predictions, or generate a submission.

## Submission CSV summary

| candidate                                       | path                                                                            |   as1_mae |   as1_delta_mae_vs_anchor |   as1_bias_pred_minus_true |   as1_spearman |   test_pearson_vs_anchor |   test_mean_abs_shift |   test_p90_abs_shift |   test_max_abs_shift |   bad_axis_id56_projection |
|:------------------------------------------------|:--------------------------------------------------------------------------------|----------:|--------------------------:|---------------------------:|---------------:|-------------------------:|----------------------:|---------------------:|---------------------:|---------------------------:|
| phase2_as1_aug_top500_id55blend_a0p1_labels_as1 | track1_activity/submissions/phase2_as1_aug_top500_id55blend_a0p1_labels_as1.csv |   0.00000 |                  -0.40657 |                    0.00000 |        1.00000 |                  0.89033 |               0.20687 |              0.61600 |              2.87596 |                    0.05005 |
| phase2_as1_aug_top500_id55blend_a0p2_labels_as1 | track1_activity/submissions/phase2_as1_aug_top500_id55blend_a0p2_labels_as1.csv |   0.00000 |                  -0.40657 |                    0.00000 |        1.00000 |                  0.88996 |               0.21323 |              0.61600 |              2.87596 |                    0.01894 |
| phase2_as1_aug_top500_id55blend_a0p3_labels_as1 | track1_activity/submissions/phase2_as1_aug_top500_id55blend_a0p3_labels_as1.csv |   0.00000 |                  -0.40657 |                    0.00000 |        1.00000 |                  0.88944 |               0.21959 |              0.61600 |              2.87596 |                   -0.01216 |
| phase2_as1_aug_top500_id55blend_a0p4_labels_as1 | track1_activity/submissions/phase2_as1_aug_top500_id55blend_a0p4_labels_as1.csv |   0.00000 |                  -0.40657 |                    0.00000 |        1.00000 |                  0.88878 |               0.22595 |              0.61600 |              2.87596 |                   -0.04326 |
| phase2_as1_aug_top500_id55blend_a0p5_labels_as1 | track1_activity/submissions/phase2_as1_aug_top500_id55blend_a0p5_labels_as1.csv |   0.00000 |                  -0.40657 |                    0.00000 |        1.00000 |                  0.88798 |               0.23232 |              0.61600 |              2.87596 |                   -0.07437 |

## AS2 shift slices

| slice                               |   n |    frac |   mean_shift |   mean_abs_shift |   p90_abs_shift |   max_abs_shift |   n_abs_gt_005 |   n_abs_gt_010 | candidate                                       |
|:------------------------------------|----:|--------:|-------------:|-----------------:|----------------:|----------------:|---------------:|---------------:|:------------------------------------------------|
| all_test                            | 513 | 1.00000 |     -0.02855 |          0.20687 |         0.61600 |         2.87596 |            224 |            202 | phase2_as1_aug_top500_id55blend_a0p1_labels_as1 |
| AS1                                 | 253 | 0.49318 |     -0.05159 |          0.40657 |         0.90129 |         2.87596 |            221 |            202 | phase2_as1_aug_top500_id55blend_a0p1_labels_as1 |
| AS2                                 | 260 | 0.50682 |     -0.00612 |          0.01255 |         0.02510 |         0.06299 |              3 |              0 | phase2_as1_aug_top500_id55blend_a0p1_labels_as1 |
| AS2_overall_risk_ge_0p80            |  26 | 0.05068 |     -0.00687 |          0.00880 |         0.01570 |         0.02957 |              0 |              0 | phase2_as1_aug_top500_id55blend_a0p1_labels_as1 |
| AS2_tag_potent_neighbor_low_support |  87 | 0.16959 |     -0.00465 |          0.01054 |         0.01925 |         0.06060 |              1 |              0 | phase2_as1_aug_top500_id55blend_a0p1_labels_as1 |
| AS2_tag_high_lf_saturated           |  37 | 0.07212 |     -0.00708 |          0.01001 |         0.01894 |         0.02135 |              0 |              0 | phase2_as1_aug_top500_id55blend_a0p1_labels_as1 |
| AS2_tag_high_lf_but_not_high_pred   |  22 | 0.04288 |     -0.00641 |          0.00965 |         0.01541 |         0.02957 |              0 |              0 | phase2_as1_aug_top500_id55blend_a0p1_labels_as1 |
| AS2_tag_member_disagreement         |  24 | 0.04678 |      0.00122 |          0.02300 |         0.04650 |         0.06060 |              1 |              0 | phase2_as1_aug_top500_id55blend_a0p1_labels_as1 |
| all_test                            | 513 | 1.00000 |     -0.03165 |          0.21323 |         0.61600 |         2.87596 |            248 |            205 | phase2_as1_aug_top500_id55blend_a0p2_labels_as1 |
| AS1                                 | 253 | 0.49318 |     -0.05159 |          0.40657 |         0.90129 |         2.87596 |            221 |            202 | phase2_as1_aug_top500_id55blend_a0p2_labels_as1 |
| AS2                                 | 260 | 0.50682 |     -0.01225 |          0.02510 |         0.05021 |         0.12599 |             27 |              3 | phase2_as1_aug_top500_id55blend_a0p2_labels_as1 |
| AS2_overall_risk_ge_0p80            |  26 | 0.05068 |     -0.01373 |          0.01760 |         0.03141 |         0.05913 |              1 |              0 | phase2_as1_aug_top500_id55blend_a0p2_labels_as1 |
| AS2_tag_potent_neighbor_low_support |  87 | 0.16959 |     -0.00930 |          0.02109 |         0.03850 |         0.12120 |              6 |              1 | phase2_as1_aug_top500_id55blend_a0p2_labels_as1 |
| AS2_tag_high_lf_saturated           |  37 | 0.07212 |     -0.01416 |          0.02001 |         0.03787 |         0.04269 |              0 |              0 | phase2_as1_aug_top500_id55blend_a0p2_labels_as1 |
| AS2_tag_high_lf_but_not_high_pred   |  22 | 0.04288 |     -0.01282 |          0.01930 |         0.03082 |         0.05913 |              1 |              0 | phase2_as1_aug_top500_id55blend_a0p2_labels_as1 |
| AS2_tag_member_disagreement         |  24 | 0.04678 |      0.00244 |          0.04601 |         0.09301 |         0.12120 |             11 |              1 | phase2_as1_aug_top500_id55blend_a0p2_labels_as1 |
| all_test                            | 513 | 1.00000 |     -0.03476 |          0.21959 |         0.61600 |         2.87596 |            281 |            219 | phase2_as1_aug_top500_id55blend_a0p3_labels_as1 |
| AS1                                 | 253 | 0.49318 |     -0.05159 |          0.40657 |         0.90129 |         2.87596 |            221 |            202 | phase2_as1_aug_top500_id55blend_a0p3_labels_as1 |
| AS2                                 | 260 | 0.50682 |     -0.01837 |          0.03765 |         0.07531 |         0.18898 |             60 |             17 | phase2_as1_aug_top500_id55blend_a0p3_labels_as1 |
| AS2_overall_risk_ge_0p80            |  26 | 0.05068 |     -0.02060 |          0.02640 |         0.04711 |         0.08870 |              2 |              0 | phase2_as1_aug_top500_id55blend_a0p3_labels_as1 |
| AS2_tag_potent_neighbor_low_support |  87 | 0.16959 |     -0.01395 |          0.03163 |         0.05775 |         0.18181 |             12 |              4 | phase2_as1_aug_top500_id55blend_a0p3_labels_as1 |
| AS2_tag_high_lf_saturated           |  37 | 0.07212 |     -0.02125 |          0.03002 |         0.05681 |         0.06404 |              6 |              0 | phase2_as1_aug_top500_id55blend_a0p3_labels_as1 |
| AS2_tag_high_lf_but_not_high_pred   |  22 | 0.04288 |     -0.01923 |          0.02895 |         0.04623 |         0.08870 |              2 |              0 | phase2_as1_aug_top500_id55blend_a0p3_labels_as1 |
| AS2_tag_member_disagreement         |  24 | 0.04678 |      0.00366 |          0.06901 |         0.13951 |         0.18181 |             13 |              9 | phase2_as1_aug_top500_id55blend_a0p3_labels_as1 |
| all_test                            | 513 | 1.00000 |     -0.03786 |          0.22595 |         0.61600 |         2.87596 |            324 |            229 | phase2_as1_aug_top500_id55blend_a0p4_labels_as1 |
| AS1                                 | 253 | 0.49318 |     -0.05159 |          0.40657 |         0.90129 |         2.87596 |            221 |            202 | phase2_as1_aug_top500_id55blend_a0p4_labels_as1 |
| AS2                                 | 260 | 0.50682 |     -0.02450 |          0.05021 |         0.10042 |         0.25197 |            103 |             27 | phase2_as1_aug_top500_id55blend_a0p4_labels_as1 |
| AS2_overall_risk_ge_0p80            |  26 | 0.05068 |     -0.02747 |          0.03519 |         0.06281 |         0.11827 |              6 |              1 | phase2_as1_aug_top500_id55blend_a0p4_labels_as1 |
| AS2_tag_potent_neighbor_low_support |  87 | 0.16959 |     -0.01860 |          0.04217 |         0.07700 |         0.24241 |             24 |              6 | phase2_as1_aug_top500_id55blend_a0p4_labels_as1 |
| AS2_tag_high_lf_saturated           |  37 | 0.07212 |     -0.02833 |          0.04003 |         0.07575 |         0.08538 |             13 |              0 | phase2_as1_aug_top500_id55blend_a0p4_labels_as1 |
| AS2_tag_high_lf_but_not_high_pred   |  22 | 0.04288 |     -0.02563 |          0.03859 |         0.06164 |         0.11827 |              7 |              1 | phase2_as1_aug_top500_id55blend_a0p4_labels_as1 |
| AS2_tag_member_disagreement         |  24 | 0.04678 |      0.00488 |          0.09202 |         0.18601 |         0.24241 |             15 |             11 | phase2_as1_aug_top500_id55blend_a0p4_labels_as1 |
| all_test                            | 513 | 1.00000 |     -0.04096 |          0.23232 |         0.61600 |         2.87596 |            347 |            236 | phase2_as1_aug_top500_id55blend_a0p5_labels_as1 |
| AS1                                 | 253 | 0.49318 |     -0.05159 |          0.40657 |         0.90129 |         2.87596 |            221 |            202 | phase2_as1_aug_top500_id55blend_a0p5_labels_as1 |
| AS2                                 | 260 | 0.50682 |     -0.03062 |          0.06276 |         0.12552 |         0.31497 |            126 |             34 | phase2_as1_aug_top500_id55blend_a0p5_labels_as1 |
| AS2_overall_risk_ge_0p80            |  26 | 0.05068 |     -0.03433 |          0.04399 |         0.07851 |         0.14783 |              6 |              1 | phase2_as1_aug_top500_id55blend_a0p5_labels_as1 |
| AS2_tag_potent_neighbor_low_support |  87 | 0.16959 |     -0.02325 |          0.05272 |         0.09625 |         0.30301 |             33 |              8 | phase2_as1_aug_top500_id55blend_a0p5_labels_as1 |
| AS2_tag_high_lf_saturated           |  37 | 0.07212 |     -0.03541 |          0.05003 |         0.09468 |         0.10673 |             16 |              1 | phase2_as1_aug_top500_id55blend_a0p5_labels_as1 |
| AS2_tag_high_lf_but_not_high_pred   |  22 | 0.04288 |     -0.03204 |          0.04824 |         0.07706 |         0.14783 |              8 |              2 | phase2_as1_aug_top500_id55blend_a0p5_labels_as1 |
| AS2_tag_member_disagreement         |  24 | 0.04678 |      0.00611 |          0.11502 |         0.23252 |         0.30301 |             17 |             11 | phase2_as1_aug_top500_id55blend_a0p5_labels_as1 |

## AS1 true-bin replay

| candidate                                       | true_bin   |   n |     mae |   bias_pred_minus_true |   spearman |   pred_mean |   pred_std |   delta_mae_vs_anchor |
|:------------------------------------------------|:-----------|----:|--------:|-----------------------:|-----------:|------------:|-----------:|----------------------:|
| phase2_as1_aug_top500_id55blend_a0p1_labels_as1 | lt3        |  24 | 0.00000 |                0.00000 |    1.00000 |     2.31917 |    0.38394 |              -1.14378 |
| phase2_as1_aug_top500_id55blend_a0p1_labels_as1 | 3to4       |  31 | 0.00000 |                0.00000 |    1.00000 |     3.56452 |    0.26117 |              -0.50362 |
| phase2_as1_aug_top500_id55blend_a0p1_labels_as1 | 4to5       |  86 | 0.00000 |                0.00000 |    1.00000 |     4.64401 |    0.27786 |              -0.34145 |
| phase2_as1_aug_top500_id55blend_a0p1_labels_as1 | 5to6       | 102 | 0.00000 |                0.00000 |    1.00000 |     5.41755 |    0.25165 |              -0.24438 |
| phase2_as1_aug_top500_id55blend_a0p1_labels_as1 | gte6       |  10 | 0.00000 |                0.00000 |    1.00000 |     6.18850 |    0.21894 |              -0.55071 |
| phase2_as1_aug_top500_id55blend_a0p2_labels_as1 | lt3        |  24 | 0.00000 |                0.00000 |    1.00000 |     2.31917 |    0.38394 |              -1.14378 |
| phase2_as1_aug_top500_id55blend_a0p2_labels_as1 | 3to4       |  31 | 0.00000 |                0.00000 |    1.00000 |     3.56452 |    0.26117 |              -0.50362 |
| phase2_as1_aug_top500_id55blend_a0p2_labels_as1 | 4to5       |  86 | 0.00000 |                0.00000 |    1.00000 |     4.64401 |    0.27786 |              -0.34145 |
| phase2_as1_aug_top500_id55blend_a0p2_labels_as1 | 5to6       | 102 | 0.00000 |                0.00000 |    1.00000 |     5.41755 |    0.25165 |              -0.24438 |
| phase2_as1_aug_top500_id55blend_a0p2_labels_as1 | gte6       |  10 | 0.00000 |                0.00000 |    1.00000 |     6.18850 |    0.21894 |              -0.55071 |
| phase2_as1_aug_top500_id55blend_a0p3_labels_as1 | lt3        |  24 | 0.00000 |                0.00000 |    1.00000 |     2.31917 |    0.38394 |              -1.14378 |
| phase2_as1_aug_top500_id55blend_a0p3_labels_as1 | 3to4       |  31 | 0.00000 |                0.00000 |    1.00000 |     3.56452 |    0.26117 |              -0.50362 |
| phase2_as1_aug_top500_id55blend_a0p3_labels_as1 | 4to5       |  86 | 0.00000 |                0.00000 |    1.00000 |     4.64401 |    0.27786 |              -0.34145 |
| phase2_as1_aug_top500_id55blend_a0p3_labels_as1 | 5to6       | 102 | 0.00000 |                0.00000 |    1.00000 |     5.41755 |    0.25165 |              -0.24438 |
| phase2_as1_aug_top500_id55blend_a0p3_labels_as1 | gte6       |  10 | 0.00000 |                0.00000 |    1.00000 |     6.18850 |    0.21894 |              -0.55071 |
| phase2_as1_aug_top500_id55blend_a0p4_labels_as1 | lt3        |  24 | 0.00000 |                0.00000 |    1.00000 |     2.31917 |    0.38394 |              -1.14378 |
| phase2_as1_aug_top500_id55blend_a0p4_labels_as1 | 3to4       |  31 | 0.00000 |                0.00000 |    1.00000 |     3.56452 |    0.26117 |              -0.50362 |
| phase2_as1_aug_top500_id55blend_a0p4_labels_as1 | 4to5       |  86 | 0.00000 |                0.00000 |    1.00000 |     4.64401 |    0.27786 |              -0.34145 |
| phase2_as1_aug_top500_id55blend_a0p4_labels_as1 | 5to6       | 102 | 0.00000 |                0.00000 |    1.00000 |     5.41755 |    0.25165 |              -0.24438 |
| phase2_as1_aug_top500_id55blend_a0p4_labels_as1 | gte6       |  10 | 0.00000 |                0.00000 |    1.00000 |     6.18850 |    0.21894 |              -0.55071 |
| phase2_as1_aug_top500_id55blend_a0p5_labels_as1 | lt3        |  24 | 0.00000 |                0.00000 |    1.00000 |     2.31917 |    0.38394 |              -1.14378 |
| phase2_as1_aug_top500_id55blend_a0p5_labels_as1 | 3to4       |  31 | 0.00000 |                0.00000 |    1.00000 |     3.56452 |    0.26117 |              -0.50362 |
| phase2_as1_aug_top500_id55blend_a0p5_labels_as1 | 4to5       |  86 | 0.00000 |                0.00000 |    1.00000 |     4.64401 |    0.27786 |              -0.34145 |
| phase2_as1_aug_top500_id55blend_a0p5_labels_as1 | 5to6       | 102 | 0.00000 |                0.00000 |    1.00000 |     5.41755 |    0.25165 |              -0.24438 |
| phase2_as1_aug_top500_id55blend_a0p5_labels_as1 | gte6       |  10 | 0.00000 |                0.00000 |    1.00000 |     6.18850 |    0.21894 |              -0.55071 |

# Phase 2 validation-slice OOF audit

This report checks whether AS2 risk-map style slices are actually hard on
labeled train OOF predictions. It uses current `ens_caruana_bag20` OOF as
the train-side proxy, not the exact id55 post-hoc CSV gate.

## Short read

- True `>=6` train compounds are the hardest OOF slice: MAE 1.0593,
  bias -1.0593. This confirms high-tail
  compression as a validation stress test.
- True `<3` train compounds are also hard: MAE 0.6489,
  bias 0.5600. Low-tail overprediction
  remains a real failure shape.
- Member-disagreement top10% is hard: MAE 0.6662
  vs rest 0.3609. This is the strongest
  general unlabeled risk flag in this pass.
- Potent-neighbor/low-support is very small on train (n=9)
  but high-error: MAE 0.7107. AS2 has many more
  compounds with this tag, so it deserves manual review.
- High-LF/high-prediction saturation is not hard on train OOF by itself
  (MAE 0.2760); it should not be read as an automatic
  upward-shift instruction.
- AS2 map tags should be treated as slice definitions first, not as
  prediction-shift instructions.
- Compare any future calibrator against these slices before spending a
  final submission.

## Hardest train OOF slices

| slice                           |   n |   frac |    mae |   rest_mae |   delta_mae_vs_rest |   bias_pred_minus_true |   rest_bias_pred_minus_true |   true_mean |   pred_mean |   lf_mean |   member_std_mean |   nn_potent_tanimoto_mean |   support_pred_bin_ge_0.50_mean |
|:--------------------------------|----:|-------:|-------:|-----------:|--------------------:|-----------------------:|----------------------------:|------------:|------------:|----------:|------------------:|--------------------------:|--------------------------------:|
| true_gte6                       |  67 | 0.0162 | 1.0593 |     0.3805 |              0.6788 |                -1.0593 |                      0.0313 |      6.2927 |      5.2335 |    1.2788 |            0.2600 |                    0.2600 |                          0.0597 |
| tag_potent_neighbor_low_support |   9 | 0.0022 | 0.7107 |     0.3907 |              0.3200 |                -0.5116 |                      0.0148 |      5.3689 |      4.8573 |    1.5908 |            0.3881 |                    0.6570 |                          0.0000 |
| true_lt3                        | 695 | 0.1679 | 0.6489 |     0.3395 |              0.3094 |                 0.5600 |                     -0.0965 |      2.3109 |      2.8710 |    0.1117 |            0.2076 |                    0.2053 |                          0.4129 |
| tag_member_disagreement_top10   | 414 | 0.1000 | 0.6662 |     0.3609 |              0.3053 |                 0.0509 |                      0.0096 |      3.8439 |      3.8948 |    0.5271 |            0.4783 |                    0.2222 |                          0.0845 |
| tag_mid_ambiguity_top10         | 414 | 0.1000 | 0.5780 |     0.3707 |              0.2072 |                -0.0049 |                      0.0158 |      4.0474 |      4.0425 |    0.5916 |            0.4059 |                    0.2579 |                          0.0145 |
| tag_count_ge2                   | 705 | 0.1703 | 0.4089 |     0.3879 |              0.0210 |                -0.0607 |                      0.0289 |      4.9086 |      4.8480 |    1.1276 |            0.2822 |                    0.2633 |                          0.0652 |
| overall_risk_top10              | 414 | 0.1000 | 0.3750 |     0.3933 |             -0.0182 |                -0.0841 |                      0.0246 |      5.0444 |      4.9603 |    1.1682 |            0.2469 |                    0.2911 |                          0.0338 |
| tag_high_lf_but_not_high_pred   | 517 | 0.1249 | 0.2964 |     0.4050 |             -0.1086 |                -0.0419 |                      0.0216 |      5.0077 |      4.9658 |    1.3171 |            0.1934 |                    0.2363 |                          0.1006 |

## All slice summary

| slice                           |    n |   frac |    mae |   rest_mae |   delta_mae_vs_rest |   bias_pred_minus_true |   rest_bias_pred_minus_true |   true_mean |   pred_mean |   lf_mean |   member_std_mean |   nn_potent_tanimoto_mean |   support_pred_bin_ge_0.50_mean |
|:--------------------------------|-----:|-------:|-------:|-----------:|--------------------:|-----------------------:|----------------------------:|------------:|------------:|----------:|------------------:|--------------------------:|--------------------------------:|
| all_train_oof                   | 4140 | 1.0000 | 0.3914 |   nan      |            nan      |                 0.0137 |                    nan      |      4.3208 |      4.3345 |    0.7652 |            0.2054 |                    0.2287 |                          0.1357 |
| true_lt3                        |  695 | 0.1679 | 0.6489 |     0.3395 |              0.3094 |                 0.5600 |                     -0.0965 |      2.3109 |      2.8710 |    0.1117 |            0.2076 |                    0.2053 |                          0.4129 |
| true_gte6                       |   67 | 0.0162 | 1.0593 |     0.3805 |              0.6788 |                -1.0593 |                      0.0313 |      6.2927 |      5.2335 |    1.2788 |            0.2600 |                    0.2600 |                          0.0597 |
| tag_high_tail_top10             |  414 | 0.1000 | 0.2885 |     0.4029 |             -0.1144 |                -0.0968 |                      0.0260 |      5.4361 |      5.3393 |    1.4589 |            0.1827 |                    0.2799 |                          0.0870 |
| tag_low_tail_top10              |  414 | 0.1000 | 0.2775 |     0.4041 |             -0.1267 |                -0.0853 |                      0.0247 |      5.3668 |      5.2815 |    1.1339 |            0.1502 |                    0.2752 |                          0.0121 |
| tag_mid_ambiguity_top10         |  414 | 0.1000 | 0.5780 |     0.3707 |              0.2072 |                -0.0049 |                      0.0158 |      4.0474 |      4.0425 |    0.5916 |            0.4059 |                    0.2579 |                          0.0145 |
| tag_potent_neighbor_low_support |    9 | 0.0022 | 0.7107 |     0.3907 |              0.3200 |                -0.5116 |                      0.0148 |      5.3689 |      4.8573 |    1.5908 |            0.3881 |                    0.6570 |                          0.0000 |
| tag_member_disagreement_top10   |  414 | 0.1000 | 0.6662 |     0.3609 |              0.3053 |                 0.0509 |                      0.0096 |      3.8439 |      3.8948 |    0.5271 |            0.4783 |                    0.2222 |                          0.0845 |
| tag_high_lf_saturated           |  518 | 0.1251 | 0.2760 |     0.4080 |             -0.1320 |                -0.0832 |                      0.0275 |      5.4517 |      5.3685 |    1.4366 |            0.1510 |                    0.2518 |                          0.0772 |
| tag_high_lf_but_not_high_pred   |  517 | 0.1249 | 0.2964 |     0.4050 |             -0.1086 |                -0.0419 |                      0.0216 |      5.0077 |      4.9658 |    1.3171 |            0.1934 |                    0.2363 |                          0.1006 |
| tag_count_ge2                   |  705 | 0.1703 | 0.4089 |     0.3879 |              0.0210 |                -0.0607 |                      0.0289 |      4.9086 |      4.8480 |    1.1276 |            0.2822 |                    0.2633 |                          0.0652 |
| overall_risk_top10              |  414 | 0.1000 | 0.3750 |     0.3933 |             -0.0182 |                -0.0841 |                      0.0246 |      5.0444 |      4.9603 |    1.1682 |            0.2469 |                    0.2911 |                          0.0338 |

## True-bin OOF summary

| true_bin   |    n |    mae |   bias_pred_minus_true |   true_mean |   pred_mean |   lf_mean |   high_tail_risk_mean |   low_tail_risk_mean |   mid_ambiguity_mean |
|:-----------|-----:|-------:|-----------------------:|------------:|------------:|----------:|----------------------:|---------------------:|---------------------:|
| 3to4       |  548 | 0.4574 |                -0.0042 |      3.5288 |      3.5246 |    0.3231 |                0.3384 |               0.3973 |               0.5639 |
| 4to5       | 1575 | 0.2866 |                 0.0492 |      4.6008 |      4.6500 |    0.8755 |                0.5360 |               0.5146 |               0.5096 |
| 5to6       | 1247 | 0.3150 |                -0.2723 |      5.3403 |      5.0680 |    1.1608 |                0.6716 |               0.6179 |               0.4859 |
| gte6       |   66 | 1.0676 |                -1.0676 |      6.2972 |      5.2296 |    1.2763 |                0.7288 |               0.6472 |               0.5943 |
| lt3        |  704 | 0.6467 |                 0.5561 |      2.3198 |      2.8759 |    0.1136 |                0.2205 |               0.3250 |               0.4455 |

## AS2 map tag counts for comparison

| as2_tag                         |   as2_n |   as2_frac |
|:--------------------------------|--------:|-----------:|
| tag_potent_neighbor_low_support |      87 |     0.3346 |
| tag_high_lf_saturated           |      37 |     0.1423 |
| tag_member_disagreement         |      24 |     0.0923 |
| tag_high_lf_but_not_high_pred   |      22 |     0.0846 |
| overall_risk_score_ge_0.80      |      26 |     0.1000 |

## Generated files

- `track1_activity/analysis/phase2_validation_slices/outputs/train_oof_validation_slices.csv`
- `track1_activity/analysis/phase2_validation_slices/outputs/slice_summary.csv`
- `track1_activity/analysis/phase2_validation_slices/outputs/true_bin_summary.csv`
- `track1_activity/analysis/phase2_validation_slices/outputs/as2_tag_count_reference.csv`

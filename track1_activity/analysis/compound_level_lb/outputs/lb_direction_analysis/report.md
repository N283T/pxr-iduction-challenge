# LB Direction Analysis

## Scope

Internal analysis of public LB feedback already recorded locally. No external
data and no model training are used here.

## LB-Known Directions vs id48

|   id | submission_name                                            |   lb_mae |   lb_delta_vs_id48 |   projection_on_id50_direction |   mean_abs_shift_vs_id48 |   cosine_vs_id50_direction |
|-----:|:-----------------------------------------------------------|---------:|-------------------:|-------------------------------:|-------------------------:|---------------------------:|
|   48 | ens_meta_axis_a343                                         | 0.407400 |           0.000000 |                       0.000000 |                 0.000000 |                 nan        |
|   43 | ens_hybrid_meta_baseline_5050                              | 0.407484 |           0.000083 |                      -0.077137 |                 0.006371 |                  -0.257091 |
|   31 | ens_caruana_bag20_calibrated_importance                    | 0.407847 |           0.000447 |                       0.168522 |                 0.013919 |                   0.257091 |
|   44 | ens_caruana_bag20_anchor_residual                          | 0.409021 |           0.001620 |                       0.109989 |                 0.024290 |                   0.104439 |
|   46 | ens_region_v2_blend_a8                                     | 0.409159 |           0.001759 |                      -0.215745 |                 0.049195 |                  -0.096805 |
|   50 | ens_internal_decor_cap101_bf50_b40_i1_l20                  | 0.409244 |           0.001843 |                       1.000000 |                 0.023666 |                   1.000000 |
|   45 | ens_caruana_bag20_admet_ai_no_log2fc_calibrated_importance | 0.409363 |           0.001962 |                       0.427962 |                 0.032262 |                   0.284975 |
|   36 | ens_caruana_bag20_calibrated_best                          | 0.414587 |           0.007186 |                       2.305449 |                 0.080562 |                   0.694262 |
|   20 | ens_caruana_bag20_calibrated_blend                         | 0.416837 |           0.009436 |                       0.925918 |                 0.069679 |                   0.303772 |
|   49 | ens_resid_lowd_core_ridge_a10p0_s50_hybrid                 | 0.419688 |           0.012287 |                       0.412336 |                 0.066948 |                   0.149619 |
|   25 | ens_caruana_bag20                                          | 0.429079 |           0.021678 |                       1.941264 |                 0.078578 |                   0.632358 |
|   10 | ens_vanilla                                                | 0.477206 |           0.069806 |                       2.354243 |                 0.087139 |                   0.669932 |
|   12 | tabpfn_2d_full_boltz_umap                                  | 0.490199 |           0.082798 |                       4.686358 |                 0.238091 |                   0.420191 |
|    9 | ens_l2_a0.1                                                | 0.495833 |           0.088432 |                       1.846308 |                 0.079920 |                   0.582644 |
|    1 | ens_l2_a0.05                                               | 0.501755 |           0.094354 |                       1.968855 |                 0.078955 |                   0.636591 |
|    6 | lgbm_mordred_jazzy_umap                                    | 0.573664 |           0.166264 |                       9.781750 |                 0.345659 |                   0.652041 |
|    4 | lgbm_mordred_jazzy_analog                                  | 0.580039 |           0.172639 |                      10.133340 |                 0.349790 |                   0.656855 |

## Reverse id50 Candidates

|    gamma | path                                                           |   mean_pred |   std_pred |   min_pred |   max_pred |   mean_shift_vs_id48 |   mean_abs_shift_vs_id48 |   p90_abs_shift_vs_id48 |   max_abs_shift_vs_id48 |   pearson_vs_id48 |
|---------:|:---------------------------------------------------------------|------------:|-----------:|-----------:|-----------:|---------------------:|-------------------------:|------------------------:|------------------------:|------------------:|
| 0.050000 | track1_activity/submissions/ens_meta_axis_reverse_id50_g05.csv |    4.796315 |   0.762159 |   2.479870 |   6.054220 |             0.000962 |                 0.001183 |                0.002265 |                0.003966 |          0.999999 |
| 0.100000 | track1_activity/submissions/ens_meta_axis_reverse_id50_g10.csv |    4.797277 |   0.762404 |   2.479536 |   6.054937 |             0.001925 |                 0.002367 |                0.004531 |                0.007932 |          0.999996 |
| 0.150000 | track1_activity/submissions/ens_meta_axis_reverse_id50_g15.csv |    4.798240 |   0.762650 |   2.479203 |   6.055655 |             0.002887 |                 0.003550 |                0.006796 |                0.011898 |          0.999992 |
| 0.200000 | track1_activity/submissions/ens_meta_axis_reverse_id50_g20.csv |    4.799202 |   0.762898 |   2.478870 |   6.056372 |             0.003849 |                 0.004733 |                0.009061 |                0.015864 |          0.999986 |

## Tiny Direction Model Diagnostics

|         n |   ridge_alpha |   loo_mae |   loo_corr |   linear_projection_coef |   linear_projection_intercept |
|----------:|--------------:|----------:|-----------:|-------------------------:|------------------------------:|
| 16.000000 |      3.162278 |  0.019625 |   0.885003 |                 0.016879 |                      0.005942 |

This model is intentionally treated as descriptive only; n is too small
for reliable optimization.

## Read

id50 moved in a direction that recovered from the residual failure but
regressed vs id48. The next low-risk A candidate is therefore a small
extrapolation away from id50 rather than a stronger blend toward it.

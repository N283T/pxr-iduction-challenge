# LB ID Direction Compare: `id57_high_activity_lift_rank2`

| candidate                                                         | base                                                                  |   candidate_mean_shift |   candidate_mean_abs_shift |   candidate_p90_abs_shift |   candidate_max_abs_shift |   n_direct_ids |   max_direct_pearson |   max_harmful_pair_pearson |   max_useful_pair_pearson |
|:------------------------------------------------------------------|:----------------------------------------------------------------------|-----------------------:|---------------------------:|--------------------------:|--------------------------:|---------------:|---------------------:|---------------------------:|--------------------------:|
| track1_activity/submissions/ens_id57_high_activity_lift_rank2.csv | track1_activity/submissions/ens_id51_top500_potent46_t40_soft_g50.csv |               0.007616 |                   0.007616 |                  0.024373 |                  0.030000 |             51 |             0.272645 |                   0.429759 |                  0.664044 |

## Most Similar Direct IDs

|   id | submission_name                                            |   lb_mae |   lb_delta_mae_vs_base |   mean_abs_delta_vs_base |   pearson_delta |   projection_on_id_delta |
|-----:|:-----------------------------------------------------------|---------:|-----------------------:|-------------------------:|----------------:|-------------------------:|
|   58 | ens_id55_combo_gate_rank1                                  | 0.407520 |               0.000131 |                 0.012071 |        0.272645 |                 0.304177 |
|   21 | ens_caruana_bag20_calibrated_importance                    | 0.415401 |               0.008012 |                 0.055839 |        0.174136 |                 0.017927 |
|   19 | ens_caruana_bag20_calibrated_importance                    | 0.415401 |               0.008012 |                 0.055839 |        0.174136 |                 0.017927 |
|   29 | ens_caruana_bag20_calibrated_importance                    | 0.418948 |               0.011559 |                 0.055839 |        0.174136 |                 0.017927 |
|   30 | ens_caruana_bag20_calibrated_importance                    | 0.408442 |               0.001054 |                 0.055839 |        0.174136 |                 0.017927 |
|   56 | ens_swap_optuna_t10_top500_calibrated_importance           | 0.413460 |               0.006071 |                 0.055839 |        0.174136 |                 0.017927 |
|   42 | ens_caruana_bag20_calibrated_importance                    | 0.409074 |               0.001686 |                 0.055839 |        0.174136 |                 0.017927 |
|   38 | ens_caruana_bag20_calibrated_importance                    | 0.410951 |               0.003563 |                 0.055839 |        0.174136 |                 0.017927 |
|   41 | ens_caruana_bag20_calibrated_importance                    | 0.413705 |               0.006316 |                 0.055839 |        0.174136 |                 0.017927 |
|   39 | ens_caruana_bag20_calibrated_importance                    | 0.408033 |               0.000644 |                 0.055839 |        0.174136 |                 0.017927 |
|   40 | ens_caruana_bag20_calibrated_importance                    | 0.411041 |               0.003653 |                 0.055839 |        0.174136 |                 0.017927 |
|   23 | ens_caruana_bag20_calibrated_importance                    | 0.414865 |               0.007476 |                 0.055839 |        0.174136 |                 0.017927 |
|   31 | ens_caruana_bag20_calibrated_importance                    | 0.407847 |               0.000458 |                 0.055839 |        0.174136 |                 0.017927 |
|   32 | ens_caruana_bag20_calibrated_importance                    | 0.407847 |               0.000458 |                 0.055839 |        0.174136 |                 0.017927 |
|   28 | ens_caruana_bag20_calibrated_importance                    | 0.414865 |               0.007476 |                 0.055839 |        0.174136 |                 0.017927 |
|   22 | ens_caruana_bag20_calibrated_importance                    | 0.415212 |               0.007824 |                 0.055839 |        0.174136 |                 0.017927 |
|   24 | ens_caruana_bag20_calibrated_importance                    | 0.420019 |               0.012630 |                 0.055839 |        0.174136 |                 0.017927 |
|   45 | ens_caruana_bag20_admet_ai_no_log2fc_calibrated_importance | 0.409363 |               0.001974 |                 0.036200 |        0.112630 |                -0.007864 |
|    3 | ens_vanilla                                                | 0.498862 |               0.091473 |                 0.120373 |        0.108060 |                -0.013061 |
|   10 | ens_vanilla                                                | 0.477206 |               0.069818 |                 0.120373 |        0.108060 |                -0.013061 |

## Similar Helpful Historical Directions

|   from_id |   to_id | from_name         | to_name                           |   lb_delta_mae |   pearson_delta |   candidate_projection |
|----------:|--------:|:------------------|:----------------------------------|---------------:|----------------:|-----------------------:|
|        13 |      16 | ens_caruana_bag20 | ens_caruana_bag20_calibrated_best |      -0.005648 |        0.664044 |               0.202141 |
|        13 |      17 | ens_caruana_bag20 | ens_caruana_bag20_calibrated_best |      -0.009597 |        0.664044 |               0.202141 |
|        13 |      18 | ens_caruana_bag20 | ens_caruana_bag20_calibrated_best |      -0.017008 |        0.664044 |               0.202141 |
|        13 |      26 | ens_caruana_bag20 | ens_caruana_bag20_calibrated_best |      -0.017078 |        0.664044 |               0.202141 |
|        13 |      27 | ens_caruana_bag20 | ens_caruana_bag20_calibrated_best |      -0.018859 |        0.664044 |               0.202141 |
|        13 |      36 | ens_caruana_bag20 | ens_caruana_bag20_calibrated_best |      -0.026840 |        0.664044 |               0.202141 |
|        14 |      16 | ens_caruana_bag20 | ens_caruana_bag20_calibrated_best |      -0.007259 |        0.664044 |               0.202141 |
|        14 |      17 | ens_caruana_bag20 | ens_caruana_bag20_calibrated_best |      -0.011208 |        0.664044 |               0.202141 |
|        14 |      18 | ens_caruana_bag20 | ens_caruana_bag20_calibrated_best |      -0.018619 |        0.664044 |               0.202141 |
|        14 |      26 | ens_caruana_bag20 | ens_caruana_bag20_calibrated_best |      -0.018688 |        0.664044 |               0.202141 |
|        14 |      27 | ens_caruana_bag20 | ens_caruana_bag20_calibrated_best |      -0.020470 |        0.664044 |               0.202141 |
|        14 |      36 | ens_caruana_bag20 | ens_caruana_bag20_calibrated_best |      -0.028451 |        0.664044 |               0.202141 |
|        15 |      16 | ens_caruana_bag20 | ens_caruana_bag20_calibrated_best |      -0.006530 |        0.664044 |               0.202141 |
|        15 |      17 | ens_caruana_bag20 | ens_caruana_bag20_calibrated_best |      -0.010479 |        0.664044 |               0.202141 |
|        15 |      18 | ens_caruana_bag20 | ens_caruana_bag20_calibrated_best |      -0.017890 |        0.664044 |               0.202141 |
|        15 |      26 | ens_caruana_bag20 | ens_caruana_bag20_calibrated_best |      -0.017959 |        0.664044 |               0.202141 |
|        15 |      27 | ens_caruana_bag20 | ens_caruana_bag20_calibrated_best |      -0.019741 |        0.664044 |               0.202141 |
|        15 |      36 | ens_caruana_bag20 | ens_caruana_bag20_calibrated_best |      -0.027722 |        0.664044 |               0.202141 |
|        25 |      26 | ens_caruana_bag20 | ens_caruana_bag20_calibrated_best |      -0.004730 |        0.664044 |               0.202141 |
|        25 |      27 | ens_caruana_bag20 | ens_caruana_bag20_calibrated_best |      -0.006511 |        0.664044 |               0.202141 |

## Similar Harmful Historical Directions

|   from_id |   to_id | from_name                                 | to_name                                                    |   lb_delta_mae |   pearson_delta |   candidate_projection |
|----------:|--------:|:------------------------------------------|:-----------------------------------------------------------|---------------:|----------------:|-----------------------:|
|        20 |      24 | ens_caruana_bag20_calibrated_blend        | ens_caruana_bag20_calibrated_importance                    |       0.003182 |        0.429759 |               0.042184 |
|        20 |      29 | ens_caruana_bag20_calibrated_blend        | ens_caruana_bag20_calibrated_importance                    |       0.002111 |        0.429759 |               0.042184 |
|        55 |      58 | ens_id51_top500_potent46_t40_soft_g35     | ens_id55_combo_gate_rank1                                  |       0.000440 |        0.374446 |               0.347103 |
|        44 |      45 | ens_caruana_bag20_anchor_residual         | ens_caruana_bag20_admet_ai_no_log2fc_calibrated_importance |       0.000342 |        0.343817 |               0.050552 |
|        50 |      56 | ens_internal_decor_cap101_bf50_b40_i1_l20 | ens_swap_optuna_t10_top500_calibrated_importance           |       0.004216 |        0.338422 |               0.082511 |
|        43 |      58 | ens_hybrid_meta_baseline_5050             | ens_id55_combo_gate_rank1                                  |       0.000036 |        0.335018 |               0.124758 |
|        47 |      58 | ens_hybrid_meta_baseline_5050             | ens_id55_combo_gate_rank1                                  |       0.000036 |        0.335018 |               0.124758 |
|        44 |      56 | ens_caruana_bag20_anchor_residual         | ens_swap_optuna_t10_top500_calibrated_importance           |       0.004439 |        0.333797 |               0.052281 |
|        43 |      56 | ens_hybrid_meta_baseline_5050             | ens_swap_optuna_t10_top500_calibrated_importance           |       0.005976 |        0.321679 |               0.044333 |
|        47 |      56 | ens_hybrid_meta_baseline_5050             | ens_swap_optuna_t10_top500_calibrated_importance           |       0.005976 |        0.321679 |               0.044333 |
|        48 |      56 | ens_meta_axis_a343                        | ens_swap_optuna_t10_top500_calibrated_importance           |       0.006059 |        0.311858 |               0.048762 |
|        20 |      26 | ens_caruana_bag20_calibrated_blend        | ens_caruana_bag20_calibrated_best                          |       0.007513 |        0.308368 |               0.008976 |
|        20 |      27 | ens_caruana_bag20_calibrated_blend        | ens_caruana_bag20_calibrated_best                          |       0.005731 |        0.308368 |               0.008976 |
|        54 |      56 | ens_id51_plus_potent_noaux_a30_delta_g050 | ens_swap_optuna_t10_top500_calibrated_importance           |       0.003878 |        0.307844 |               0.041578 |
|        51 |      58 | ens_meta_axis_reverse_id50_g10            | ens_id55_combo_gate_rank1                                  |       0.000194 |        0.307615 |               0.135371 |
|        51 |      56 | ens_meta_axis_reverse_id50_g10            | ens_swap_optuna_t10_top500_calibrated_importance           |       0.006134 |        0.307341 |               0.044047 |
|        48 |      58 | ens_meta_axis_a343                        | ens_id55_combo_gate_rank1                                  |       0.000120 |        0.306297 |               0.138571 |
|        50 |      53 | ens_internal_decor_cap101_bf50_b40_i1_l20 | ens_repooled_trunk_core_only_calibrated_importance         |       0.001321 |        0.284232 |               0.095084 |
|        44 |      53 | ens_caruana_bag20_anchor_residual         | ens_repooled_trunk_core_only_calibrated_importance         |       0.001544 |        0.283753 |               0.026154 |
|        43 |      45 | ens_hybrid_meta_baseline_5050             | ens_caruana_bag20_admet_ai_no_log2fc_calibrated_importance |       0.001879 |        0.281302 |               0.031560 |

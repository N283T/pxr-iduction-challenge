# LB ID Direction Compare: `id57_bold_lf_calib_rank5`

| candidate                                                    | base                                                                  |   candidate_mean_shift |   candidate_mean_abs_shift |   candidate_p90_abs_shift |   candidate_max_abs_shift |   n_direct_ids |   max_direct_pearson |   max_harmful_pair_pearson |   max_useful_pair_pearson |
|:-------------------------------------------------------------|:----------------------------------------------------------------------|-----------------------:|---------------------------:|--------------------------:|--------------------------:|---------------:|---------------------:|---------------------------:|--------------------------:|
| track1_activity/submissions/ens_id57_bold_lf_calib_rank5.csv | track1_activity/submissions/ens_id51_top500_potent46_t40_soft_g50.csv |               0.016824 |                   0.016824 |                  0.058290 |                  0.080000 |             51 |             0.251492 |                   0.396322 |                  0.653072 |

## Most Similar Direct IDs

|   id | submission_name                                            |   lb_mae |   lb_delta_mae_vs_base |   mean_abs_delta_vs_base |   pearson_delta |   projection_on_id_delta |
|-----:|:-----------------------------------------------------------|---------:|-----------------------:|-------------------------:|----------------:|-------------------------:|
|   58 | ens_id55_combo_gate_rank1                                  | 0.407520 |               0.000131 |                 0.012071 |        0.251492 |                 0.683587 |
|   21 | ens_caruana_bag20_calibrated_importance                    | 0.415401 |               0.008012 |                 0.055839 |        0.136419 |                 0.033911 |
|   19 | ens_caruana_bag20_calibrated_importance                    | 0.415401 |               0.008012 |                 0.055839 |        0.136419 |                 0.033911 |
|   29 | ens_caruana_bag20_calibrated_importance                    | 0.418948 |               0.011559 |                 0.055839 |        0.136419 |                 0.033911 |
|   30 | ens_caruana_bag20_calibrated_importance                    | 0.408442 |               0.001054 |                 0.055839 |        0.136419 |                 0.033911 |
|   56 | ens_swap_optuna_t10_top500_calibrated_importance           | 0.413460 |               0.006071 |                 0.055839 |        0.136419 |                 0.033911 |
|   42 | ens_caruana_bag20_calibrated_importance                    | 0.409074 |               0.001686 |                 0.055839 |        0.136419 |                 0.033911 |
|   38 | ens_caruana_bag20_calibrated_importance                    | 0.410951 |               0.003563 |                 0.055839 |        0.136419 |                 0.033911 |
|   41 | ens_caruana_bag20_calibrated_importance                    | 0.413705 |               0.006316 |                 0.055839 |        0.136419 |                 0.033911 |
|   39 | ens_caruana_bag20_calibrated_importance                    | 0.408033 |               0.000644 |                 0.055839 |        0.136419 |                 0.033911 |
|   40 | ens_caruana_bag20_calibrated_importance                    | 0.411041 |               0.003653 |                 0.055839 |        0.136419 |                 0.033911 |
|   23 | ens_caruana_bag20_calibrated_importance                    | 0.414865 |               0.007476 |                 0.055839 |        0.136419 |                 0.033911 |
|   31 | ens_caruana_bag20_calibrated_importance                    | 0.407847 |               0.000458 |                 0.055839 |        0.136419 |                 0.033911 |
|   32 | ens_caruana_bag20_calibrated_importance                    | 0.407847 |               0.000458 |                 0.055839 |        0.136419 |                 0.033911 |
|   28 | ens_caruana_bag20_calibrated_importance                    | 0.414865 |               0.007476 |                 0.055839 |        0.136419 |                 0.033911 |
|   22 | ens_caruana_bag20_calibrated_importance                    | 0.415212 |               0.007824 |                 0.055839 |        0.136419 |                 0.033911 |
|   24 | ens_caruana_bag20_calibrated_importance                    | 0.420019 |               0.012630 |                 0.055839 |        0.136419 |                 0.033911 |
|   45 | ens_caruana_bag20_admet_ai_no_log2fc_calibrated_importance | 0.409363 |               0.001974 |                 0.036200 |        0.098670 |                -0.017934 |
|   52 | ens_repooled_trunk_swap_core_calibrated_importance         | 0.408710 |               0.001322 |                 0.030475 |        0.082179 |                -0.051058 |
|    3 | ens_vanilla                                                | 0.498862 |               0.091473 |                 0.120373 |        0.072242 |                -0.032257 |

## Similar Helpful Historical Directions

|   from_id |   to_id | from_name         | to_name                           |   lb_delta_mae |   pearson_delta |   candidate_projection |
|----------:|--------:|:------------------|:----------------------------------|---------------:|----------------:|-----------------------:|
|        13 |      16 | ens_caruana_bag20 | ens_caruana_bag20_calibrated_best |      -0.005648 |        0.653072 |               0.485846 |
|        13 |      17 | ens_caruana_bag20 | ens_caruana_bag20_calibrated_best |      -0.009597 |        0.653072 |               0.485846 |
|        13 |      18 | ens_caruana_bag20 | ens_caruana_bag20_calibrated_best |      -0.017008 |        0.653072 |               0.485846 |
|        13 |      26 | ens_caruana_bag20 | ens_caruana_bag20_calibrated_best |      -0.017078 |        0.653072 |               0.485846 |
|        13 |      27 | ens_caruana_bag20 | ens_caruana_bag20_calibrated_best |      -0.018859 |        0.653072 |               0.485846 |
|        13 |      36 | ens_caruana_bag20 | ens_caruana_bag20_calibrated_best |      -0.026840 |        0.653072 |               0.485846 |
|        14 |      16 | ens_caruana_bag20 | ens_caruana_bag20_calibrated_best |      -0.007259 |        0.653072 |               0.485846 |
|        14 |      17 | ens_caruana_bag20 | ens_caruana_bag20_calibrated_best |      -0.011208 |        0.653072 |               0.485846 |
|        14 |      18 | ens_caruana_bag20 | ens_caruana_bag20_calibrated_best |      -0.018619 |        0.653072 |               0.485846 |
|        14 |      26 | ens_caruana_bag20 | ens_caruana_bag20_calibrated_best |      -0.018688 |        0.653072 |               0.485846 |
|        14 |      27 | ens_caruana_bag20 | ens_caruana_bag20_calibrated_best |      -0.020470 |        0.653072 |               0.485846 |
|        14 |      36 | ens_caruana_bag20 | ens_caruana_bag20_calibrated_best |      -0.028451 |        0.653072 |               0.485846 |
|        15 |      16 | ens_caruana_bag20 | ens_caruana_bag20_calibrated_best |      -0.006530 |        0.653072 |               0.485846 |
|        15 |      17 | ens_caruana_bag20 | ens_caruana_bag20_calibrated_best |      -0.010479 |        0.653072 |               0.485846 |
|        15 |      18 | ens_caruana_bag20 | ens_caruana_bag20_calibrated_best |      -0.017890 |        0.653072 |               0.485846 |
|        15 |      26 | ens_caruana_bag20 | ens_caruana_bag20_calibrated_best |      -0.017959 |        0.653072 |               0.485846 |
|        15 |      27 | ens_caruana_bag20 | ens_caruana_bag20_calibrated_best |      -0.019741 |        0.653072 |               0.485846 |
|        15 |      36 | ens_caruana_bag20 | ens_caruana_bag20_calibrated_best |      -0.027722 |        0.653072 |               0.485846 |
|        25 |      26 | ens_caruana_bag20 | ens_caruana_bag20_calibrated_best |      -0.004730 |        0.653072 |               0.485846 |
|        25 |      27 | ens_caruana_bag20 | ens_caruana_bag20_calibrated_best |      -0.006511 |        0.653072 |               0.485846 |

## Similar Harmful Historical Directions

|   from_id |   to_id | from_name                                 | to_name                                                    |   lb_delta_mae |   pearson_delta |   candidate_projection |
|----------:|--------:|:------------------------------------------|:-----------------------------------------------------------|---------------:|----------------:|-----------------------:|
|        20 |      24 | ens_caruana_bag20_calibrated_blend        | ens_caruana_bag20_calibrated_importance                    |       0.003182 |        0.396322 |               0.096057 |
|        20 |      29 | ens_caruana_bag20_calibrated_blend        | ens_caruana_bag20_calibrated_importance                    |       0.002111 |        0.396322 |               0.096057 |
|        55 |      58 | ens_id51_top500_potent46_t40_soft_g35     | ens_id55_combo_gate_rank1                                  |       0.000440 |        0.359770 |               0.798698 |
|        43 |      58 | ens_hybrid_meta_baseline_5050             | ens_id55_combo_gate_rank1                                  |       0.000036 |        0.330785 |               0.295225 |
|        47 |      58 | ens_hybrid_meta_baseline_5050             | ens_id55_combo_gate_rank1                                  |       0.000036 |        0.330785 |               0.295225 |
|        44 |      45 | ens_caruana_bag20_anchor_residual         | ens_caruana_bag20_admet_ai_no_log2fc_calibrated_importance |       0.000342 |        0.313699 |               0.117417 |
|        51 |      58 | ens_meta_axis_reverse_id50_g10            | ens_id55_combo_gate_rank1                                  |       0.000194 |        0.307340 |               0.321574 |
|        48 |      58 | ens_meta_axis_a343                        | ens_id55_combo_gate_rank1                                  |       0.000120 |        0.304537 |               0.326620 |
|        50 |      56 | ens_internal_decor_cap101_bf50_b40_i1_l20 | ens_swap_optuna_t10_top500_calibrated_importance           |       0.004216 |        0.288660 |               0.178272 |
|        43 |      56 | ens_hybrid_meta_baseline_5050             | ens_swap_optuna_t10_top500_calibrated_importance           |       0.005976 |        0.282085 |               0.096842 |
|        47 |      56 | ens_hybrid_meta_baseline_5050             | ens_swap_optuna_t10_top500_calibrated_importance           |       0.005976 |        0.282085 |               0.096842 |
|        44 |      56 | ens_caruana_bag20_anchor_residual         | ens_swap_optuna_t10_top500_calibrated_importance           |       0.004439 |        0.279654 |               0.109017 |
|        20 |      26 | ens_caruana_bag20_calibrated_blend        | ens_caruana_bag20_calibrated_best                          |       0.007513 |        0.274231 |               0.020003 |
|        20 |      27 | ens_caruana_bag20_calibrated_blend        | ens_caruana_bag20_calibrated_best                          |       0.005731 |        0.274231 |               0.020003 |
|        43 |      45 | ens_hybrid_meta_baseline_5050             | ens_caruana_bag20_admet_ai_no_log2fc_calibrated_importance |       0.001879 |        0.271445 |               0.080206 |
|        43 |      52 | ens_hybrid_meta_baseline_5050             | ens_repooled_trunk_swap_core_calibrated_importance         |       0.001227 |        0.270897 |               0.078760 |
|        47 |      52 | ens_hybrid_meta_baseline_5050             | ens_repooled_trunk_swap_core_calibrated_importance         |       0.001227 |        0.270897 |               0.078760 |
|        48 |      56 | ens_meta_axis_a343                        | ens_swap_optuna_t10_top500_calibrated_importance           |       0.006059 |        0.270657 |               0.105484 |
|        48 |      52 | ens_meta_axis_a343                        | ens_repooled_trunk_swap_core_calibrated_importance         |       0.001310 |        0.269927 |               0.093540 |
|        51 |      56 | ens_meta_axis_reverse_id50_g10            | ens_swap_optuna_t10_top500_calibrated_importance           |       0.006134 |        0.267237 |               0.095307 |

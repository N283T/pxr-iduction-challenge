# Phase 2 validation matrix

This matrix separates AS1 external validation from the new `train + AS1`
cross-fit OOF. The two numbers answer different questions and should not
be merged into one score prematurely.

## Labeled pool

- train rows: 4140
- AS1 rows: 253
- total labeled rows: 4393

## Phase2 OOF recipe

- feature: `cheme_2d_full_boltz_log2fc_pred_seed10ens`
- model: LightGBM top-500
- folds: 5, UMAP clusters: 100, seed: 42

## Split summary

|   fold |    n_val |   n_val_train_source |   n_val_as1_source |   val_y_mean |   val_y_std |   val_min |   val_max |
|-------:|---------:|---------------------:|-------------------:|-------------:|------------:|----------:|----------:|
| 0.0000 | 873.0000 |             824.0000 |            49.0000 |       4.4836 |      0.9665 |    1.6300 |    6.3450 |
| 1.0000 | 873.0000 |             824.0000 |            49.0000 |       4.4280 |      1.1027 |    1.6100 |    7.5486 |
| 2.0000 | 879.0000 |             827.0000 |            52.0000 |       4.2225 |      1.2121 |    1.6200 |    6.8450 |
| 3.0000 | 886.0000 |             833.0000 |            53.0000 |       4.1079 |      1.1692 |    1.6600 |    6.2300 |
| 4.0000 | 882.0000 |             832.0000 |            50.0000 |       4.4639 |      1.0821 |    1.6200 |    6.8550 |

## Fold metrics

|   fold |   best_iteration |   zero_gain_selected |        n |    mae |   bias_pred_minus_true |   spearman |   pred_mean |   true_mean |
|-------:|-----------------:|---------------------:|---------:|-------:|-----------------------:|-----------:|------------:|------------:|
| 0.0000 |         333.0000 |               0.0000 | 873.0000 | 0.3719 |                 0.0224 |     0.8005 |      4.5060 |      4.4836 |
| 1.0000 |         616.0000 |               0.0000 | 873.0000 | 0.4044 |                 0.0126 |     0.8222 |      4.4406 |      4.4280 |
| 2.0000 |         618.0000 |               0.0000 | 879.0000 | 0.3928 |                 0.0513 |     0.8687 |      4.2738 |      4.2225 |
| 3.0000 |         638.0000 |               0.0000 | 886.0000 | 0.4425 |                 0.0139 |     0.8540 |      4.1218 |      4.1079 |
| 4.0000 |        1131.0000 |               0.0000 | 882.0000 | 0.4277 |                 0.0116 |     0.8180 |      4.4755 |      4.4639 |

## Phase2 OOF slices

| slice        |    n |    mae |   bias_pred_minus_true |   spearman |   pred_mean |   true_mean |
|:-------------|-----:|-------:|-----------------------:|-----------:|------------:|------------:|
| all          | 4393 | 0.4080 |                 0.0224 |     0.8382 |      4.3630 |      4.3406 |
| source_train | 4140 | 0.4060 |                 0.0274 |     0.8378 |      4.3482 |      4.3208 |
| source_as1   |  253 | 0.4410 |                -0.0604 |     0.8314 |      4.6037 |      4.6641 |
| true_lt3     |  719 | 0.6636 |                 0.5451 |     0.0160 |      2.8563 |      2.3112 |
| true_gte6    |   77 | 1.0670 |                -1.0670 |     0.0572 |      5.2122 |      6.2792 |
| bin_lt3      |  728 | 0.6628 |                 0.5424 |     0.0302 |      2.8622 |      2.3197 |
| bin_3to4     |  579 | 0.5171 |                 0.0209 |     0.4707 |      3.5517 |      3.5307 |
| bin_4to5     | 1661 | 0.3061 |                 0.0758 |     0.4057 |      4.6788 |      4.6031 |
| bin_5to6     | 1349 | 0.3116 |                -0.2617 |     0.4392 |      5.0844 |      5.3461 |
| bin_gte6     |   76 | 1.0728 |                -1.0728 |     0.0601 |      5.2101 |      6.2829 |

## AS1 external benchmark, top rows

| candidate                                                               |        n |    mae |   bias_pred_minus_true |   spearman | path                                                                                                    |
|:------------------------------------------------------------------------|---------:|-------:|-----------------------:|-----------:|:--------------------------------------------------------------------------------------------------------|
| ens_id51_top500_potent46_t40_soft_g35                                   | 253.0000 | 0.4066 |                 0.0516 |     0.8488 | track1_activity/submissions/ens_id51_top500_potent46_t40_soft_g35.csv                                   |
| ens_id51_top500_potent46_t40_soft_g50                                   | 253.0000 | 0.4069 |                 0.0519 |     0.8486 | track1_activity/submissions/ens_id51_top500_potent46_t40_soft_g50.csv                                   |
| ens_id55_combo_gate_rank1                                               | 253.0000 | 0.4070 |                 0.0588 |     0.8479 | track1_activity/submissions/ens_id55_combo_gate_rank1.csv                                               |
| ens_id57_high_activity_lift_rank2                                       | 253.0000 | 0.4072 |                 0.0596 |     0.8480 | track1_activity/submissions/ens_id57_high_activity_lift_rank2.csv                                       |
| tabpfn_cheme_2d_full_boltz_log2fc_pred_seed10ens_top500_umap_v3_temp0p7 | 253.0000 | 0.4107 |                -0.0181 |     0.8357 | track1_activity/submissions/tabpfn_cheme_2d_full_boltz_log2fc_pred_seed10ens_top500_umap_v3_temp0p7.csv |
| ens_swap_optuna_t10_top500_calibrated_importance                        | 253.0000 | 0.4131 |                 0.0403 |     0.8424 | track1_activity/submissions/ens_swap_optuna_t10_top500_calibrated_importance.csv                        |
| tabpfn_cheme_2d_full_boltz_log2fc_pred_seed10ens_top500_umap_v3         | 253.0000 | 0.4135 |                -0.0177 |     0.8353 | track1_activity/submissions/tabpfn_cheme_2d_full_boltz_log2fc_pred_seed10ens_top500_umap_v3.csv         |
| tabpfn_cheme_2d_full_boltz_log2fc_pred_seed10ens_top400_umap_v3         | 253.0000 | 0.4136 |                -0.0177 |     0.8355 | track1_activity/submissions/tabpfn_cheme_2d_full_boltz_log2fc_pred_seed10ens_top400_umap_v3.csv         |

## TabPFN member OOF scoreboard

These rows use TabPFN v2.6 on the `train + AS1` cross-fit folds. They are
development metrics for retraining decisions, not direct submission replays.

| member                                                                      |   all_mae |   source_as1_mae |   true_lt3_mae |   true_gte6_mae |   all_spearman |   source_as1_spearman |
|:----------------------------------------------------------------------------|----------:|-----------------:|---------------:|----------------:|---------------:|----------------------:|
| tabpfn_cheme_2d_full_boltz_log2fc_pred_optuna_trial10_seed5ens_top500_umap  |    0.3877 |           0.4505 |         0.6218 |          0.8786 |         0.8553 |                0.8103 |
| tabpfn_cheme_2d_full_boltz_log2fc_pred_seed10ens_top500_umap                |    0.3961 |           0.4242 |         0.6229 |          0.9142 |         0.8498 |                0.8319 |
| tabpfn_cheme_2d_full_boltz_log2fc_pred_optuna_trial10_seed5ens_umap_default |    0.4293 |           0.4621 |         0.7388 |          1.1387 |         0.8157 |                0.8121 |
| tabpfn_chemprop_pretrain_embed_umap_default                                 |    0.4357 |           0.4532 |         0.6939 |          1.0997 |         0.8120 |                0.8226 |
| tabpfn_kermt_pretrain_embed_umap_default                                    |    0.4494 |           0.4731 |         0.7205 |          1.1940 |         0.7958 |                0.7956 |

## Weight comparison scoreboard

| setting                       |   all_mae |   source_as1_mae |   true_lt3_mae |   true_gte6_mae |   all_spearman |   source_as1_spearman |
|:------------------------------|----------:|-----------------:|---------------:|----------------:|---------------:|----------------------:|
| phase2_vanilla_opt            |    0.3916 |           0.4371 |         0.6645 |          0.9877 |         0.8524 |                0.8269 |
| phase2_caruana_bag20          |    0.3995 |           0.4386 |         0.6829 |          1.0493 |         0.8451 |                0.8298 |
| old_ens_caruana_bag20_weights |    0.4024 |           0.4387 |         0.6936 |          1.0662 |         0.8433 |                0.8335 |
| phase2_l2_0p3                 |    0.4113 |           0.4414 |         0.7178 |          1.1049 |         0.8366 |                0.8311 |
| simple_average_old_members    |    0.4230 |           0.4471 |         0.7432 |          1.1443 |         0.8265 |                0.8287 |

## Interpretation

- `AS1 external` remains the fixed LB replacement for already-built test predictions.
- `Phase2 OOF` is the development proxy for models that train on `train + AS1`.
- AS1-only wins should be treated as overfit risk unless Phase2 OOF and train-source slices also hold up.
- The best single member in Phase2 OOF is not automatically the safest Phase2
  choice. The optuna trial10 top-500 member is strongest on all-row OOF, but the
  AS1 replay report previously marked this OOF-optimized top-500/log2fc-heavy
  direction as risky. Use it as a strong axis, not as an unconstrained winner.
- Ensemble weights are useful here as a guardrail against top-500/log2fc
  over-concentration. The goal is not to average in weak members for their own
  sake, but to preserve the top-500 signal while limiting broad damage in AS1
  replay and activity-tail slices.
- A lightweight bin classifier may be worth testing later as a soft gate or
  feature for tail uncertainty, but hard tail switches remain high risk because
  the `>=6` and `<3` labeled tails are small.

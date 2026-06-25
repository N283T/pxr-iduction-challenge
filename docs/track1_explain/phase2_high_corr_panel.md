# Phase 2 high-correlation feature panel

Experiment-only probe: screen broad feature families for columns that
rank `gte6` compounds, then train a CV-safe binary classifier on the
selected high-correlation panel.

## Top100 plus high-corr panel

Combining top100 selected features with the high-correlation panel was tested.
The combination is better than high-corr-only, but it does not beat the original
top100/top200 classifiers.

| config | all AUC | all AP | AS1 AUC | AS1 AP | default precision | default recall | best id55 AS1 MAE | best Phase2 OOF MAE |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| high-corr only p100 | 0.8597 | 0.1384 | 0.8967 | 0.2493 | 0.0775 | 0.6842 | 0.4071 | 0.4079 |
| binary top100 + high-corr | 0.8695 | 0.1530 | 0.8897 | 0.2803 | 0.0789 | 0.7105 | 0.4062 | 0.4078 |
| multiclass top100 + high-corr | 0.8729 | 0.1559 | 0.9008 | 0.2749 | 0.0776 | 0.6974 | 0.4062 | 0.4071 |
| binary top100 | 0.8720 | 0.1304 | 0.8889 | 0.2806 | 0.0857 | 0.7632 | 0.4067 | 0.4075 |
| binary top200 | 0.8919 | 0.1926 | 0.9115 | 0.3798 | 0.0880 | 0.7632 | 0.4065 | 0.4067 |
| multiclass top100 | 0.8976 | 0.2096 | n/a | n/a | 0.1418 | 0.7368 | 0.4058 | 0.4070 |
| multiclass top200 | 0.9016 | 0.2245 | n/a | n/a | 0.1440 | 0.7237 | 0.4062 | 0.4068 |

Current read: top100 + high-corr adds some weak information, but it dilutes the
high-tail ranking. The high-corr panel is better kept as an interpretability
audit rather than concatenated into the classifier.

## Threshold Diagnostic

For diagnostics, threshold cuts are more interpretable than fixed topK. Using
the broad screen output:

| threshold | selected features | families | dominant families |
|:--|--:|--:|:--|
| AP >= 0.10 | 33 | 6 | AttentiveFP 27, tabular bundle 2 |
| AP >= 0.08 | 98 | 11 | AttentiveFP 57, DR latent 8, KERMT 6 |
| oriented AUC >= 0.80 | 20 | 2 | AttentiveFP 18, tabular bundle 2 |
| AP >= 0.10 or AUC >= 0.80 | 38 | 6 | AttentiveFP 32, tabular bundle 2 |
| AP >= 0.10 and top50 precision >= 0.20 | 24 | 2 | AttentiveFP 23, tabular bundle 1 |

The cleanest diagnostic cut is probably `AP >= 0.10 and top50 precision >= 0.20`.
It mostly identifies AttentiveFP dimensions plus the known `log2fc_8p25_pred`
axis. This supports the idea that high-tail detection has a real graph-embedding
signal, but it is not yet precise enough to drive a hard activity lift by itself.

## Latest Run

- Config: `high_corr_panel_tabpfn_pf8_p100_multiclasstop100_v3_ne8_t0p9_balanced`
- Model: `tabpfn`
- Per-family screen count: 8
- Max panel size: 100
- Base topK feature: `cheme_2d_full_boltz_log2fc_pred_seed10ens`
- Base topK count: 100
- Base topK ranker objective: `multiclass`

## Family Screen Summary

| family                                    |   n_screened |   best_ap |   best_auc |   best_top50_precision |   median_top_feature_ap |
|:------------------------------------------|-------------:|----------:|-----------:|-----------------------:|------------------------:|
| cheme_2d_full_boltz_log2fc_pred_seed10ens |           30 |    0.1384 |     0.8636 |                 0.2000 |                  0.0655 |
| attentivefp_pretrain_embed                |           30 |    0.1327 |     0.8324 |                 0.3000 |                  0.1117 |
| chemprop_drlatent_embed                   |           30 |    0.1183 |     0.7722 |                 0.1800 |                  0.0636 |
| chemprop_counter_emax_embed               |           30 |    0.1045 |     0.7426 |                 0.2000 |                  0.0576 |
| chemprop_assay_shape_drlatent_embed       |           30 |    0.1009 |     0.7789 |                 0.2000 |                  0.0582 |
| chemberta_5m_mtr_pretrain_embed           |           30 |    0.1004 |     0.7479 |                 0.1600 |                  0.0531 |
| kermt_pretrain_embed                      |           30 |    0.0984 |     0.7740 |                 0.2200 |                  0.0720 |
| gatedgcn_pretrain_embed                   |           30 |    0.0927 |     0.7663 |                 0.1800 |                  0.0592 |
| chemprop_mtr_embed                        |           30 |    0.0861 |     0.6755 |                 0.1400 |                  0.0402 |
| unimol_v2_log2fc_real_embed               |           30 |    0.0812 |     0.7564 |                 0.1600 |                  0.0584 |
| unimol_v2_pretrain_embed                  |           30 |    0.0812 |     0.7564 |                 0.1600 |                  0.0584 |
| ka_gnn_pretrain_embed                     |           30 |    0.0784 |     0.7553 |                 0.2000 |                  0.0510 |
| chemprop_pretrain_optuna_trial10_embed    |           30 |    0.0783 |     0.7961 |                 0.1800 |                  0.0486 |
| molformer_c3_pretrain_embed               |           30 |    0.0757 |     0.7497 |                 0.1600 |                  0.0551 |
| chemprop_log2fc_htchem_pretrain_embed     |           30 |    0.0736 |     0.7949 |                 0.1800 |                  0.0548 |
| chemprop_pretrain_embed                   |           30 |    0.0653 |     0.7669 |                 0.1600 |                  0.0489 |
| molformer_c3_mtr_embed                    |           30 |    0.0613 |     0.7289 |                 0.1400 |                  0.0409 |

## Top Global Features

| family                                    |   feature_idx | feature_name                                     |   direction |    auc |   oriented_auc |   average_precision |   top25_precision |   top50_precision |   top100_precision |   mean_pos |   mean_neg |
|:------------------------------------------|--------------:|:-------------------------------------------------|------------:|-------:|---------------:|--------------------:|------------------:|------------------:|-------------------:|-----------:|-----------:|
| cheme_2d_full_boltz_log2fc_pred_seed10ens |          2101 | cheme_2d_full_boltz_log2fc_pred_seed10ens__f2101 |           1 | 0.8636 |         0.8636 |              0.1384 |            0.2000 |            0.1600 |             0.1800 |     1.2700 |     0.5633 |
| attentivefp_pretrain_embed                |           335 | attentivefp_pretrain_embed__f0335                |           1 | 0.7637 |         0.7637 |              0.1327 |            0.2400 |            0.2200 |             0.1700 |     0.3805 |     0.0988 |
| attentivefp_pretrain_embed                |           511 | attentivefp_pretrain_embed__f0511                |           1 | 0.7742 |         0.7742 |              0.1275 |            0.4400 |            0.3000 |             0.1700 |     0.5695 |     0.2310 |
| attentivefp_pretrain_embed                |           476 | attentivefp_pretrain_embed__f0476                |           1 | 0.7933 |         0.7933 |              0.1262 |            0.2400 |            0.2400 |             0.1800 |     0.5938 |     0.2280 |
| attentivefp_pretrain_embed                |           506 | attentivefp_pretrain_embed__f0506                |           1 | 0.8106 |         0.8106 |              0.1249 |            0.2800 |            0.2000 |             0.1600 |     0.5415 |     0.2157 |
| attentivefp_pretrain_embed                |           174 | attentivefp_pretrain_embed__f0174                |           1 | 0.7782 |         0.7782 |              0.1203 |            0.2000 |            0.2600 |             0.1800 |     0.3409 |     0.0757 |
| attentivefp_pretrain_embed                |            84 | attentivefp_pretrain_embed__f0084                |           1 | 0.8200 |         0.8200 |              0.1202 |            0.2000 |            0.2200 |             0.1700 |     0.4641 |     0.1010 |
| chemprop_drlatent_embed                   |            37 | chemprop_drlatent_embed__f0037                   |           1 | 0.7178 |         0.7178 |              0.1183 |            0.3200 |            0.1800 |             0.1200 |     0.5257 |    -0.3205 |
| cheme_2d_full_boltz_log2fc_pred_seed10ens |          2103 | pred_htchem                                      |           1 | 0.8257 |         0.8257 |              0.1171 |            0.2400 |            0.2000 |             0.1600 |     5.5039 |     4.4826 |
| attentivefp_pretrain_embed                |           141 | attentivefp_pretrain_embed__f0141                |           1 | 0.8311 |         0.8311 |              0.1157 |            0.1600 |            0.2400 |             0.1600 |     0.5726 |     0.1763 |
| attentivefp_pretrain_embed                |           215 | attentivefp_pretrain_embed__f0215                |           1 | 0.8217 |         0.8217 |              0.1157 |            0.2800 |            0.2400 |             0.1500 |     0.6238 |     0.2172 |
| attentivefp_pretrain_embed                |           271 | attentivefp_pretrain_embed__f0271                |           1 | 0.8213 |         0.8213 |              0.1150 |            0.2800 |            0.2000 |             0.1600 |     0.4106 |     0.1169 |
| attentivefp_pretrain_embed                |           244 | attentivefp_pretrain_embed__f0244                |           1 | 0.8031 |         0.8031 |              0.1131 |            0.2400 |            0.1800 |             0.1300 |     0.6607 |     0.2972 |
| attentivefp_pretrain_embed                |            89 | attentivefp_pretrain_embed__f0089                |           1 | 0.8009 |         0.8009 |              0.1131 |            0.2000 |            0.2000 |             0.1900 |     0.4269 |     0.0946 |
| attentivefp_pretrain_embed                |           380 | attentivefp_pretrain_embed__f0380                |           1 | 0.8149 |         0.8149 |              0.1128 |            0.1600 |            0.2000 |             0.2000 |     0.4622 |     0.1058 |
| attentivefp_pretrain_embed                |           350 | attentivefp_pretrain_embed__f0350                |           1 | 0.7784 |         0.7784 |              0.1127 |            0.4000 |            0.2400 |             0.1600 |     0.6893 |     0.1616 |
| attentivefp_pretrain_embed                |            48 | attentivefp_pretrain_embed__f0048                |           1 | 0.7709 |         0.7709 |              0.1120 |            0.3200 |            0.2000 |             0.1700 |     0.3593 |     0.0907 |
| attentivefp_pretrain_embed                |           131 | attentivefp_pretrain_embed__f0131                |           1 | 0.7633 |         0.7633 |              0.1118 |            0.2400 |            0.2800 |             0.1900 |     0.3748 |     0.1050 |
| attentivefp_pretrain_embed                |            27 | attentivefp_pretrain_embed__f0027                |           1 | 0.7296 |         0.7296 |              0.1116 |            0.2800 |            0.2000 |             0.1700 |     0.4764 |     0.2692 |
| attentivefp_pretrain_embed                |            91 | attentivefp_pretrain_embed__f0091                |           1 | 0.8192 |         0.8192 |              0.1107 |            0.2400 |            0.1600 |             0.1300 |     0.7544 |     0.3862 |
| attentivefp_pretrain_embed                |           420 | attentivefp_pretrain_embed__f0420                |           1 | 0.8074 |         0.8074 |              0.1076 |            0.2000 |            0.2000 |             0.1700 |     0.7882 |     0.4707 |
| attentivefp_pretrain_embed                |           438 | attentivefp_pretrain_embed__f0438                |           1 | 0.7712 |         0.7712 |              0.1059 |            0.2000 |            0.2600 |             0.1800 |     0.4054 |     0.1200 |
| attentivefp_pretrain_embed                |           474 | attentivefp_pretrain_embed__f0474                |           1 | 0.8324 |         0.8324 |              0.1054 |            0.2400 |            0.1400 |             0.1400 |     0.5910 |     0.1681 |
| attentivefp_pretrain_embed                |             6 | attentivefp_pretrain_embed__f0006                |           1 | 0.7876 |         0.7876 |              0.1052 |            0.3200 |            0.2200 |             0.1500 |     0.3113 |     0.0982 |
| attentivefp_pretrain_embed                |            98 | attentivefp_pretrain_embed__f0098                |           1 | 0.7700 |         0.7700 |              0.1051 |            0.1600 |            0.2000 |             0.1800 |     0.4954 |     0.2102 |
| attentivefp_pretrain_embed                |           207 | attentivefp_pretrain_embed__f0207                |           1 | 0.8055 |         0.8055 |              0.1047 |            0.2000 |            0.1800 |             0.1700 |     0.4749 |     0.1112 |
| chemprop_counter_emax_embed               |            79 | chemprop_counter_emax_embed__f0079               |           1 | 0.6119 |         0.6119 |              0.1045 |            0.2800 |            0.1400 |             0.0700 |     0.4834 |    -0.1351 |
| attentivefp_pretrain_embed                |           281 | attentivefp_pretrain_embed__f0281                |           1 | 0.7573 |         0.7573 |              0.1033 |            0.2400 |            0.2600 |             0.2100 |     0.4554 |     0.2403 |
| attentivefp_pretrain_embed                |           510 | attentivefp_pretrain_embed__f0510                |           1 | 0.7736 |         0.7736 |              0.1027 |            0.3600 |            0.2200 |             0.1500 |     0.6331 |     0.2915 |
| attentivefp_pretrain_embed                |           331 | attentivefp_pretrain_embed__f0331                |           1 | 0.8085 |         0.8085 |              0.1025 |            0.2800 |            0.2400 |             0.1500 |     1.0540 |     0.4289 |

## Fold Summary

|   fold |   n_train |    n_val |   n_pos_val |   panel_size |   val_auc |   val_ap |
|-------:|----------:|---------:|------------:|-------------:|----------:|---------:|
| 0.0000 | 3520.0000 | 873.0000 |     11.0000 |     200.0000 |    0.8843 |   0.1185 |
| 1.0000 | 3520.0000 | 873.0000 |     24.0000 |     200.0000 |    0.8659 |   0.2363 |
| 2.0000 | 3514.0000 | 879.0000 |     14.0000 |     200.0000 |    0.8766 |   0.1510 |
| 3.0000 | 3507.0000 | 886.0000 |      6.0000 |     200.0000 |    0.8447 |   0.0344 |
| 4.0000 | 3511.0000 | 882.0000 |     21.0000 |     200.0000 |    0.8774 |   0.2924 |

## Classifier Summary

| slice        |    n |   n_pos |   balanced_accuracy |     f1 |   log_loss |   roc_auc |   average_precision |   score_mean |
|:-------------|-----:|--------:|--------------------:|-------:|-----------:|----------:|--------------------:|-------------:|
| all          | 4393 |      76 |              0.7757 | 0.1397 |     0.3456 |    0.8729 |              0.1559 |       0.2290 |
| source_train | 4140 |      66 |              0.7700 | 0.1306 |     0.3410 |    0.8669 |              0.1584 |       0.2264 |
| source_as1   |  253 |      10 |              0.7930 | 0.2286 |     0.4198 |    0.9008 |              0.2749 |       0.2703 |

## Threshold Class Metrics

| class        |   support |   precision |   recall |     f1 |
|:-------------|----------:|------------:|---------:|-------:|
| not_positive |      4317 |      0.9938 |   0.8541 | 0.9186 |
| positive     |        76 |      0.0776 |   0.6974 | 0.1397 |

## Best High-Shift Gates

|   threshold |   high_shift |   phase2_all_mae |   phase2_all_bias |   phase2_pos_mae |   id55_as1_mae |   id55_as1_bias |   id55_as1_pos_mae |   phase2_flags |   phase2_pos_flags |   as1_flags |   as1_pos_flags |   as2_flags |   as2_mean_abs_shift |
|------------:|-------------:|-----------------:|------------------:|-----------------:|---------------:|----------------:|-------------------:|---------------:|-------------------:|------------:|----------------:|------------:|---------------------:|
|      0.7500 |       0.1000 |           0.4071 |            0.0291 |           1.0188 |         0.4062 |          0.0607 |             0.4807 |       295.0000 |            41.0000 |     23.0000 |          7.0000 |     44.0000 |               0.0169 |
|      0.7500 |       0.0500 |           0.4074 |            0.0257 |           1.0458 |         0.4062 |          0.0561 |             0.5157 |       295.0000 |            41.0000 |     23.0000 |          7.0000 |     44.0000 |               0.0085 |
|      0.7500 |       0.1500 |           0.4071 |            0.0324 |           0.9919 |         0.4064 |          0.0652 |             0.4457 |       295.0000 |            41.0000 |     23.0000 |          7.0000 |     44.0000 |               0.0254 |
|      0.0800 |       0.0500 |           0.4082 |            0.0532 |           1.0247 |         0.4066 |          0.0820 |             0.5007 |      2707.0000 |            73.0000 |    154.0000 |         10.0000 |    209.0000 |               0.0402 |
|      0.2000 |       0.0500 |           0.4074 |            0.0412 |           1.0274 |         0.4067 |          0.0733 |             0.5007 |      1653.0000 |            69.0000 |    110.0000 |         10.0000 |    126.0000 |               0.0242 |
|      0.1000 |       0.0500 |           0.4080 |            0.0501 |           1.0247 |         0.4068 |          0.0797 |             0.5007 |      2434.0000 |            73.0000 |    142.0000 |         10.0000 |    189.0000 |               0.0363 |
|      0.9500 |       0.0500 |           0.4079 |            0.0226 |           1.0682 |         0.4069 |          0.0524 |             0.5457 |        24.0000 |             7.0000 |      4.0000 |          1.0000 |      1.0000 |               0.0002 |
|      0.2500 |       0.0500 |           0.4073 |            0.0383 |           1.0287 |         0.4070 |          0.0708 |             0.5007 |      1395.0000 |            67.0000 |     97.0000 |         10.0000 |    108.0000 |               0.0208 |
|      0.8500 |       0.0500 |           0.4075 |            0.0241 |           1.0544 |         0.4072 |          0.0548 |             0.5257 |       152.0000 |            28.0000 |     16.0000 |          5.0000 |     25.0000 |               0.0048 |
|      0.9500 |       0.1000 |           0.4079 |            0.0229 |           1.0636 |         0.4073 |          0.0532 |             0.5407 |        24.0000 |             7.0000 |      4.0000 |          1.0000 |      1.0000 |               0.0004 |
|      0.1500 |       0.0500 |           0.4077 |            0.0449 |           1.0274 |         0.4073 |          0.0749 |             0.5007 |      1983.0000 |            69.0000 |    118.0000 |         10.0000 |    156.0000 |               0.0300 |
|      0.7500 |       0.2000 |           0.4074 |            0.0358 |           0.9649 |         0.4076 |          0.0698 |             0.4195 |       295.0000 |            41.0000 |     23.0000 |          7.0000 |     44.0000 |               0.0338 |

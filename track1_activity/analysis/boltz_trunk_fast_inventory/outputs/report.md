# Boltz Trunk Fast Inventory

## Coverage

- compounds table rows: 13136
- compound_boltz2 rows: 4653
- compound_boltz2 embedding paths: 4652
- compound_boltz2 preprocessing_failed rows: 1
- compound_boltz2 confidence rows: 4652
- compound_boltz2 affinity rows: 4652
- compound_boltz2 ligand distance rows: 4652
- compound_boltz2_trunk_fast rows: 13134
- compound_boltz2_trunk_fast source paths existing: 13134
- missing from trunk_fast: [1657, 8624]

## Recycling Split

- rcycle=1: 8482
- rcycle=3: 4652

## Stored Vector Dimensions

|   s_prot_dim |   s_lig_dim |   z_if_mean_dim |   z_if_max_dim |
|-------------:|------------:|----------------:|---------------:|
|          384 |         384 |             128 |            128 |

## Sample Raw NPZ Shapes

|   compound_id |   rcycle | readable   | s_shape       | z_shape            |   ligand_tokens | finite   |   size_mb |
|--------------:|---------:|:-----------|:--------------|:-------------------|----------------:|:---------|----------:|
|          4654 |        1 | True       | [1, 455, 384] | [1, 455, 455, 128] |              21 | True     |     97.44 |
|          4655 |        1 | True       | [1, 458, 384] | [1, 458, 458, 128] |              24 | True     |     98.72 |
|          4656 |        1 | True       | [1, 462, 384] | [1, 462, 462, 128] |              28 | True     |    100.43 |
|             1 |        3 | True       | [1, 457, 384] | [1, 457, 457, 128] |              23 | True     |     98.3  |
|             2 |        3 | True       | [1, 462, 384] | [1, 462, 462, 128] |              28 | True     |    100.44 |
|             3 |        3 | True       | [1, 460, 384] | [1, 460, 460, 128] |              26 | True     |     99.57 |

## Existing Boltz-Family Experiments

|   id | name                                                                        |   mae_mean |   rae_mean |   spearman_mean | created_at                       | category          |
|-----:|:----------------------------------------------------------------------------|-----------:|-----------:|----------------:|:---------------------------------|:------------------|
| 1888 | tabpfn_cheme_2d_full_boltz_log2fc_pred_optuna_trial10_seed5ens_umap_default |     0.3959 |     0.4393 |          0.8425 | 2026-04-26 14:23:22.664657+00:00 | descriptor_mix    |
| 1825 | tabpfn_cheme_2d_full_boltz_log2fc_pred_seed15ens_top500_umap                |     0.396  |     0.4395 |          0.8454 | 2026-04-25 08:09:10.839076+00:00 | descriptor_mix    |
| 2236 | tabpfn_cheme_2d_full_boltz_log2fc_pred_seed10ens_admet_ai_top500_umap       |     0.3963 |     0.4399 |          0.8462 | 2026-04-29 08:31:37.085439+00:00 | descriptor_mix    |
| 1609 | tabpfn_cheme_2d_full_boltz_log2fc_pred_seed10ens_top500_umap                |     0.3966 |     0.4401 |          0.8458 | 2026-04-25 07:30:00.516963+00:00 | descriptor_mix    |
| 1552 | tabpfn_cheme_2d_full_boltz_log2fc_pred_seed5ens_top500_umap                 |     0.3987 |     0.4424 |          0.8426 | 2026-04-25 05:44:47.291889+00:00 | descriptor_mix    |
| 1889 | tabpfn_cheme_2d_full_boltz_log2fc_pred_optuna_trial11_seed5ens_umap_default |     0.4008 |     0.4447 |          0.8403 | 2026-04-26 14:27:44.958556+00:00 | descriptor_mix    |
| 1608 | tabpfn_cheme_2d_full_boltz_log2fc_pred_seed10ens_umap_default               |     0.4055 |     0.4501 |          0.838  | 2026-04-25 07:17:08.277311+00:00 | descriptor_mix    |
| 1826 | tabpfn_cheme_2d_full_boltz_log2fc_pred_seed15ens_umap_default               |     0.4058 |     0.4505 |          0.8376 | 2026-04-25 08:14:11.771420+00:00 | descriptor_mix    |
| 1511 | tabpfn_cheme_2d_full_boltz_log2fc_pred_seed5ens_umap_default                |     0.4068 |     0.4514 |          0.8357 | 2026-04-25 05:17:22.872612+00:00 | descriptor_mix    |
| 1102 | tabpfn_oe_cheme_2d_full_boltz_log2fc_pred_top500_umap                       |     0.407  |     0.4511 |          0.8338 | 2026-04-23 02:34:14.083799+00:00 | descriptor_mix    |
|  911 | tabpfn_cheme_2d_full_boltz_log2fc_pred_top500_umap                          |     0.4179 |     0.4634 |          0.8279 | 2026-04-22 11:47:56.158682+00:00 | descriptor_mix    |
|  983 | tabpfn_cheme_2d_full_boltz_log2fc_pred_top300_umap                          |     0.4186 |     0.4643 |          0.8267 | 2026-04-22 12:27:54.001363+00:00 | descriptor_mix    |
|  984 | tabpfn_cheme_2d_full_boltz_log2fc_pred_top200_umap                          |     0.419  |     0.4648 |          0.826  | 2026-04-22 12:33:20.988063+00:00 | descriptor_mix    |
|  985 | tabpfn_cheme_2d_full_boltz_log2fc_pred_top100_umap                          |     0.4192 |     0.465  |          0.8263 | 2026-04-22 12:35:54.751516+00:00 | descriptor_mix    |
| 1291 | tabpfn_cheme_2d_full_boltz_log2fc_pred_ens4_top500_umap                     |     0.4194 |     0.4653 |          0.8253 | 2026-04-23 23:13:18.197785+00:00 | descriptor_mix    |
|  982 | tabpfn_cheme_2d_full_boltz_log2fc_pred_top800_umap                          |     0.4201 |     0.466  |          0.8258 | 2026-04-22 12:20:53.373233+00:00 | descriptor_mix    |
|  728 | tabpfn_cheme_2d_full_boltz_log2fc_pred_umap_default                         |     0.4212 |     0.4674 |          0.8236 | 2026-04-21 06:24:54.085025+00:00 | descriptor_mix    |
|  744 | tabpfn_cheme_2d_full_boltz_log2fc_pred_umap_s43_default                     |     0.4241 |     0.4935 |          0.8085 | 2026-04-21 07:27:19.550200+00:00 | descriptor_mix    |
| 1290 | tabpfn_cheme_2d_full_boltz_log2fc_pred_ens4_umap_default                    |     0.4249 |     0.4718 |          0.8179 | 2026-04-23 23:01:25.002910+00:00 | descriptor_mix    |
| 1510 | tabpfn_cheme_2d_full_boltz_log2fc_emax_pred_umap_default                    |     0.426  |     0.4728 |          0.8168 | 2026-04-25 04:27:40.315527+00:00 | descriptor_mix    |
| 2322 | tabpfn_cheme_2d_full_boltz_log2fc_drlatent_umap_default                     |     0.4273 |     0.4745 |          0.814  | 2026-05-05 07:42:00.264810+00:00 | descriptor_mix    |
| 2321 | lgbm_cheme_2d_full_boltz_log2fc_drlatent_umap_default                       |     0.4302 |     0.4773 |          0.8164 | 2026-05-05 07:36:19.727669+00:00 | descriptor_mix    |
| 2324 | lgbm_cheme_2d_full_boltz_log2fc_contact_umap_default                        |     0.4323 |     0.4797 |          0.8154 | 2026-05-05 07:47:17.166182+00:00 | structure_tabular |
|  909 | tabpfn_cconcat_2d_full_boltz_log2fc_pred_umap_default                       |     0.4331 |     0.4807 |          0.8097 | 2026-04-22 08:45:02.807486+00:00 | descriptor_mix    |
| 2235 | tabpfn_cheme_2d_full_boltz_log2fc_pred_seed10ens_admet_ai_umap              |     0.4352 |     0.4829 |          0.8064 | 2026-04-29 08:22:31.008052+00:00 | descriptor_mix    |
|  910 | tabpfn_cheme_cconcat_2d_full_boltz_log2fc_pred_umap_default                 |     0.4369 |     0.4851 |          0.8061 | 2026-04-22 08:49:20.439305+00:00 | descriptor_mix    |
|  569 | tabpfn_2d_full_boltz_log2fc_pred_umap_default                               |     0.4427 |     0.491  |          0.801  | 2026-04-19 15:29:24.677741+00:00 | descriptor_mix    |
|  988 | tabpfn_cheme_2d_full_boltz_log2fc_pred_pca500_umap                          |     0.4577 |     0.5087 |          0.7905 | 2026-04-22 13:48:33.675071+00:00 | descriptor_mix    |
| 2237 | tabpfn_cheme_2d_full_boltz_admet_ai_top500_umap                             |     0.4631 |     0.5144 |          0.7732 | 2026-04-29 11:21:40.533886+00:00 | descriptor_mix    |
| 2239 | tabpfn_2d_full_boltz_plus_shape_umap                                        |     0.4804 |     0.5339 |          0.7532 | 2026-04-30 13:57:52.743098+00:00 | descriptor_mix    |
|  754 | tabpfn_boltz_raw_plus_pretrain_concat_umap_default                          |     0.4818 |     0.5354 |          0.7631 | 2026-04-22 06:58:13.006046+00:00 | trunk_only        |
|  440 | tabpfn_2d_full_boltz_umap                                                   |     0.4824 |     0.5362 |          0.75   | 2026-04-18 11:02:30.411269+00:00 | descriptor_mix    |
|  751 | tabpfn_boltz_trunk_pretrain_embed_c_concat_umap_default                     |     0.485  |     0.5388 |          0.7612 | 2026-04-22 06:44:26.144959+00:00 | trunk_only        |
|  462 | tabpfn_pooled_boltz_allpairs_umap_default                                   |     0.4859 |     0.5403 |          0.7577 | 2026-04-19 06:54:17.112730+00:00 | trunk_only        |
|  463 | tabpfn_pooled_boltz_umap_default                                            |     0.486  |     0.5403 |          0.7539 | 2026-04-19 06:59:06.049238+00:00 | trunk_only        |
|  438 | tabpfn_2d_full_boltz_umap_default                                           |     0.4874 |     0.5416 |          0.7498 | 2026-04-18 08:28:57.041034+00:00 | descriptor_mix    |
|  908 | tabpfn_pooled_boltz_ab_nozmax_umap_default                                  |     0.4886 |     0.5433 |          0.755  | 2026-04-22 08:27:25.714392+00:00 | trunk_only        |
|  904 | boltz_trunk_pretrain_c_concat_full_ft_umap                                  |     0.489  |     0.5435 |          0.7576 | 2026-04-22 08:14:11.726714+00:00 | trunk_only        |
| 1118 | tabm_cheme_2d_full_boltz_log2fc_pred_umap                                   |     0.4893 |     0.5442 |          0.7759 | 2026-04-23 04:09:11.689426+00:00 | descriptor_mix    |
|  756 | tabpfn_pooled_boltz_ab_zonly_umap_default                                   |     0.4893 |     0.544  |          0.7523 | 2026-04-22 07:06:48.089506+00:00 | trunk_only        |
|  906 | tabpfn_pooled_boltz_ab_zmean_umap_default                                   |     0.4897 |     0.5443 |          0.7516 | 2026-04-22 08:16:15.661131+00:00 | trunk_only        |
|  749 | tabpfn_boltz_trunk_pretrain_embed_c_h512_umap_default                       |     0.4897 |     0.5441 |          0.7565 | 2026-04-22 06:21:29.452147+00:00 | trunk_only        |
|  753 | tabpfn_boltz_trunk_pretrain_embed_c_concat_t8p25_umap_default               |     0.4903 |     0.5447 |          0.7573 | 2026-04-22 06:52:58.524685+00:00 | trunk_only        |
|  903 | boltz_trunk_pretrain_c_concat_head_only_umap                                |     0.4931 |     0.5476 |          0.7554 | 2026-04-22 08:00:47.488908+00:00 | trunk_only        |
|  745 | tabpfn_boltz_trunk_pretrain_embed_c_umap_default                            |     0.4944 |     0.5496 |          0.7518 | 2026-04-22 06:08:23.651133+00:00 | trunk_only        |
|  750 | tabpfn_boltz_trunk_pretrain_embed_c_h1024_umap_default                      |     0.4946 |     0.5498 |          0.7518 | 2026-04-22 06:30:00.825165+00:00 | trunk_only        |
|  752 | tabpfn_boltz_trunk_pretrain_embed_c_cls_umap_default                        |     0.5046 |     0.5608 |          0.7368 | 2026-04-22 06:46:47.549385+00:00 | trunk_only        |
|  987 | tabpfn_cheme_2d_full_boltz_log2fc_pred_pls500_umap                          |     0.5054 |     0.5611 |          0.7556 | 2026-04-22 13:36:46.184250+00:00 | descriptor_mix    |
|  755 | tabpfn_pooled_boltz_ab_sonly_umap_default                                   |     0.5075 |     0.5643 |          0.7335 | 2026-04-22 07:05:36.939359+00:00 | trunk_only        |
|  451 | lgbm_pooled_boltz_umap                                                      |     0.5115 |     0.5691 |          0.7328 | 2026-04-19 05:52:29.756204+00:00 | trunk_only        |
|  748 | tabpfn_boltz_trunk_pretrain_embed_b_first_umap_default                      |     0.5136 |     0.5702 |          0.7296 | 2026-04-22 06:14:57.774084+00:00 | trunk_only        |
|  905 | tabpfn_pooled_boltz_ab_slig_umap_default                                    |     0.5155 |     0.5729 |          0.7247 | 2026-04-22 08:14:52.956891+00:00 | trunk_only        |
|  747 | tabpfn_boltz_trunk_pretrain_embed_b_umap_default                            |     0.5168 |     0.574  |          0.7268 | 2026-04-22 06:10:45.442747+00:00 | trunk_only        |
|  486 | chemprop_multitask_desc30_umap_w0.0469_noboltz_tuned                        |     0.519  |     0.5897 |          0.7235 | 2026-04-19 09:53:17.372431+00:00 | descriptor_mix    |
|  746 | tabpfn_boltz_trunk_pretrain_embed_a_umap_default                            |     0.5227 |     0.5806 |          0.7178 | 2026-04-22 06:09:36.909094+00:00 | trunk_only        |
|  902 | tabpfn_pooled_boltz_ab_sprot_umap_default                                   |     0.5328 |     0.5929 |          0.7012 | 2026-04-22 07:38:53.020797+00:00 | trunk_only        |
|  461 | mlp_pooled_boltz_umap                                                       |     0.5377 |     0.5991 |          0.7138 | 2026-04-19 06:00:39.193730+00:00 | trunk_only        |
|  907 | tabpfn_pooled_boltz_ab_zmax_umap_default                                    |     0.5755 |     0.6396 |          0.6604 | 2026-04-22 08:17:12.825680+00:00 | trunk_only        |
| 1332 | tabpfn_boltz2_tabular_tier0_umap_default                                    |     0.5797 |     0.6439 |          0.6524 | 2026-04-24 02:23:45.174232+00:00 | structure_tabular |
| 1353 | tabpfn_boltz2_mordred3d_umap_default                                        |     0.608  |     0.6761 |          0.6203 | 2026-04-24 02:26:35.897219+00:00 | structure_tabular |
| 2323 | lgbm_boltz2_contact_umap_default                                            |     0.6296 |     0.6985 |          0.6012 | 2026-05-05 07:45:45.592397+00:00 | structure_tabular |

### trunk_only

|   id | name                                                          |   mae_mean |   rae_mean |   spearman_mean | created_at                       | category   |
|-----:|:--------------------------------------------------------------|-----------:|-----------:|----------------:|:---------------------------------|:-----------|
|  754 | tabpfn_boltz_raw_plus_pretrain_concat_umap_default            |     0.4818 |     0.5354 |          0.7631 | 2026-04-22 06:58:13.006046+00:00 | trunk_only |
|  751 | tabpfn_boltz_trunk_pretrain_embed_c_concat_umap_default       |     0.485  |     0.5388 |          0.7612 | 2026-04-22 06:44:26.144959+00:00 | trunk_only |
|  462 | tabpfn_pooled_boltz_allpairs_umap_default                     |     0.4859 |     0.5403 |          0.7577 | 2026-04-19 06:54:17.112730+00:00 | trunk_only |
|  463 | tabpfn_pooled_boltz_umap_default                              |     0.486  |     0.5403 |          0.7539 | 2026-04-19 06:59:06.049238+00:00 | trunk_only |
|  908 | tabpfn_pooled_boltz_ab_nozmax_umap_default                    |     0.4886 |     0.5433 |          0.755  | 2026-04-22 08:27:25.714392+00:00 | trunk_only |
|  904 | boltz_trunk_pretrain_c_concat_full_ft_umap                    |     0.489  |     0.5435 |          0.7576 | 2026-04-22 08:14:11.726714+00:00 | trunk_only |
|  756 | tabpfn_pooled_boltz_ab_zonly_umap_default                     |     0.4893 |     0.544  |          0.7523 | 2026-04-22 07:06:48.089506+00:00 | trunk_only |
|  906 | tabpfn_pooled_boltz_ab_zmean_umap_default                     |     0.4897 |     0.5443 |          0.7516 | 2026-04-22 08:16:15.661131+00:00 | trunk_only |
|  749 | tabpfn_boltz_trunk_pretrain_embed_c_h512_umap_default         |     0.4897 |     0.5441 |          0.7565 | 2026-04-22 06:21:29.452147+00:00 | trunk_only |
|  753 | tabpfn_boltz_trunk_pretrain_embed_c_concat_t8p25_umap_default |     0.4903 |     0.5447 |          0.7573 | 2026-04-22 06:52:58.524685+00:00 | trunk_only |
|  903 | boltz_trunk_pretrain_c_concat_head_only_umap                  |     0.4931 |     0.5476 |          0.7554 | 2026-04-22 08:00:47.488908+00:00 | trunk_only |
|  745 | tabpfn_boltz_trunk_pretrain_embed_c_umap_default              |     0.4944 |     0.5496 |          0.7518 | 2026-04-22 06:08:23.651133+00:00 | trunk_only |
|  750 | tabpfn_boltz_trunk_pretrain_embed_c_h1024_umap_default        |     0.4946 |     0.5498 |          0.7518 | 2026-04-22 06:30:00.825165+00:00 | trunk_only |
|  752 | tabpfn_boltz_trunk_pretrain_embed_c_cls_umap_default          |     0.5046 |     0.5608 |          0.7368 | 2026-04-22 06:46:47.549385+00:00 | trunk_only |
|  755 | tabpfn_pooled_boltz_ab_sonly_umap_default                     |     0.5075 |     0.5643 |          0.7335 | 2026-04-22 07:05:36.939359+00:00 | trunk_only |
|  451 | lgbm_pooled_boltz_umap                                        |     0.5115 |     0.5691 |          0.7328 | 2026-04-19 05:52:29.756204+00:00 | trunk_only |
|  748 | tabpfn_boltz_trunk_pretrain_embed_b_first_umap_default        |     0.5136 |     0.5702 |          0.7296 | 2026-04-22 06:14:57.774084+00:00 | trunk_only |
|  905 | tabpfn_pooled_boltz_ab_slig_umap_default                      |     0.5155 |     0.5729 |          0.7247 | 2026-04-22 08:14:52.956891+00:00 | trunk_only |
|  747 | tabpfn_boltz_trunk_pretrain_embed_b_umap_default              |     0.5168 |     0.574  |          0.7268 | 2026-04-22 06:10:45.442747+00:00 | trunk_only |
|  746 | tabpfn_boltz_trunk_pretrain_embed_a_umap_default              |     0.5227 |     0.5806 |          0.7178 | 2026-04-22 06:09:36.909094+00:00 | trunk_only |

### structure_tabular

|   id | name                                                 |   mae_mean |   rae_mean |   spearman_mean | created_at                       | category          |
|-----:|:-----------------------------------------------------|-----------:|-----------:|----------------:|:---------------------------------|:------------------|
| 2324 | lgbm_cheme_2d_full_boltz_log2fc_contact_umap_default |     0.4323 |     0.4797 |          0.8154 | 2026-05-05 07:47:17.166182+00:00 | structure_tabular |
| 1332 | tabpfn_boltz2_tabular_tier0_umap_default             |     0.5797 |     0.6439 |          0.6524 | 2026-04-24 02:23:45.174232+00:00 | structure_tabular |
| 1353 | tabpfn_boltz2_mordred3d_umap_default                 |     0.608  |     0.6761 |          0.6203 | 2026-04-24 02:26:35.897219+00:00 | structure_tabular |
| 2323 | lgbm_boltz2_contact_umap_default                     |     0.6296 |     0.6985 |          0.6012 | 2026-05-05 07:45:45.592397+00:00 | structure_tabular |

### descriptor_mix

|   id | name                                                                        |   mae_mean |   rae_mean |   spearman_mean | created_at                       | category       |
|-----:|:----------------------------------------------------------------------------|-----------:|-----------:|----------------:|:---------------------------------|:---------------|
| 1888 | tabpfn_cheme_2d_full_boltz_log2fc_pred_optuna_trial10_seed5ens_umap_default |     0.3959 |     0.4393 |          0.8425 | 2026-04-26 14:23:22.664657+00:00 | descriptor_mix |
| 1825 | tabpfn_cheme_2d_full_boltz_log2fc_pred_seed15ens_top500_umap                |     0.396  |     0.4395 |          0.8454 | 2026-04-25 08:09:10.839076+00:00 | descriptor_mix |
| 2236 | tabpfn_cheme_2d_full_boltz_log2fc_pred_seed10ens_admet_ai_top500_umap       |     0.3963 |     0.4399 |          0.8462 | 2026-04-29 08:31:37.085439+00:00 | descriptor_mix |
| 1609 | tabpfn_cheme_2d_full_boltz_log2fc_pred_seed10ens_top500_umap                |     0.3966 |     0.4401 |          0.8458 | 2026-04-25 07:30:00.516963+00:00 | descriptor_mix |
| 1552 | tabpfn_cheme_2d_full_boltz_log2fc_pred_seed5ens_top500_umap                 |     0.3987 |     0.4424 |          0.8426 | 2026-04-25 05:44:47.291889+00:00 | descriptor_mix |
| 1889 | tabpfn_cheme_2d_full_boltz_log2fc_pred_optuna_trial11_seed5ens_umap_default |     0.4008 |     0.4447 |          0.8403 | 2026-04-26 14:27:44.958556+00:00 | descriptor_mix |
| 1608 | tabpfn_cheme_2d_full_boltz_log2fc_pred_seed10ens_umap_default               |     0.4055 |     0.4501 |          0.838  | 2026-04-25 07:17:08.277311+00:00 | descriptor_mix |
| 1826 | tabpfn_cheme_2d_full_boltz_log2fc_pred_seed15ens_umap_default               |     0.4058 |     0.4505 |          0.8376 | 2026-04-25 08:14:11.771420+00:00 | descriptor_mix |
| 1511 | tabpfn_cheme_2d_full_boltz_log2fc_pred_seed5ens_umap_default                |     0.4068 |     0.4514 |          0.8357 | 2026-04-25 05:17:22.872612+00:00 | descriptor_mix |
| 1102 | tabpfn_oe_cheme_2d_full_boltz_log2fc_pred_top500_umap                       |     0.407  |     0.4511 |          0.8338 | 2026-04-23 02:34:14.083799+00:00 | descriptor_mix |
|  911 | tabpfn_cheme_2d_full_boltz_log2fc_pred_top500_umap                          |     0.4179 |     0.4634 |          0.8279 | 2026-04-22 11:47:56.158682+00:00 | descriptor_mix |
|  983 | tabpfn_cheme_2d_full_boltz_log2fc_pred_top300_umap                          |     0.4186 |     0.4643 |          0.8267 | 2026-04-22 12:27:54.001363+00:00 | descriptor_mix |
|  984 | tabpfn_cheme_2d_full_boltz_log2fc_pred_top200_umap                          |     0.419  |     0.4648 |          0.826  | 2026-04-22 12:33:20.988063+00:00 | descriptor_mix |
|  985 | tabpfn_cheme_2d_full_boltz_log2fc_pred_top100_umap                          |     0.4192 |     0.465  |          0.8263 | 2026-04-22 12:35:54.751516+00:00 | descriptor_mix |
| 1291 | tabpfn_cheme_2d_full_boltz_log2fc_pred_ens4_top500_umap                     |     0.4194 |     0.4653 |          0.8253 | 2026-04-23 23:13:18.197785+00:00 | descriptor_mix |
|  982 | tabpfn_cheme_2d_full_boltz_log2fc_pred_top800_umap                          |     0.4201 |     0.466  |          0.8258 | 2026-04-22 12:20:53.373233+00:00 | descriptor_mix |
|  728 | tabpfn_cheme_2d_full_boltz_log2fc_pred_umap_default                         |     0.4212 |     0.4674 |          0.8236 | 2026-04-21 06:24:54.085025+00:00 | descriptor_mix |
|  744 | tabpfn_cheme_2d_full_boltz_log2fc_pred_umap_s43_default                     |     0.4241 |     0.4935 |          0.8085 | 2026-04-21 07:27:19.550200+00:00 | descriptor_mix |
| 1290 | tabpfn_cheme_2d_full_boltz_log2fc_pred_ens4_umap_default                    |     0.4249 |     0.4718 |          0.8179 | 2026-04-23 23:01:25.002910+00:00 | descriptor_mix |
| 1510 | tabpfn_cheme_2d_full_boltz_log2fc_emax_pred_umap_default                    |     0.426  |     0.4728 |          0.8168 | 2026-04-25 04:27:40.315527+00:00 | descriptor_mix |

## Read

- The 13k trunk-fast layer is available for weak-label pretraining and raw `s/z` re-pooling.
- Only the 4652 rcycle=3 full-run rows should be used for pose, confidence, affinity, or contact-gated structure diagnostics.
- Any next Boltz candidate should preserve the existing allpairs reserve member unless replacement evidence is strong.

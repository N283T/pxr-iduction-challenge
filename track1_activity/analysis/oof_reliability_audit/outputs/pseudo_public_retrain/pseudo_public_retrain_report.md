# Pseudo-Public Holdout Retrain Battery

Fast leakage-free LGBM baselines are retrained per split. OOF-stack rows
are diagnostics only because the base predictions come from the existing
canonical OOF pool rather than split-specific retraining.

## Top leakage-free retrains by split

| split                        |   rank_mae_within_kind | model                            | family               |   folds |   n_val_mean |    mae |   mae_std |   spearman |    bias |
|:-----------------------------|-----------------------:|:---------------------------------|:---------------------|--------:|-------------:|-------:|----------:|-----------:|--------:|
| public_adv_top513            |                 1.0000 | rdkit_full_lf_pred_lgbm          | 2d_plus_log2fc_pred  |       1 |     513.0000 | 0.3746 |    0.0000 |     0.8204 | -0.0902 |
| public_adv_top513            |                 2.0000 | mordred_lf_pred_lgbm             | 2d_plus_log2fc_pred  |       1 |     513.0000 | 0.3855 |    0.0000 |     0.8144 | -0.1055 |
| public_adv_top513            |                 3.0000 | rdkit_full_singleconc_jazzy_lgbm | 2d_plus_aux_observed |       1 |     513.0000 | 0.3973 |    0.0000 |     0.8281 | -0.1194 |
| public_adv_top513            |                 4.0000 | mordred_singleconc_jazzy_lgbm    | 2d_plus_aux_observed |       1 |     513.0000 | 0.4002 |    0.0000 |     0.8331 | -0.0895 |
| public_adv_top513            |                 5.0000 | mordred_lgbm                     | 2d                   |       1 |     513.0000 | 0.5170 |    0.0000 |     0.6173 | -0.1529 |
| public_chembl_ext_nn_ge025   |                 1.0000 | rdkit_full_lf_pred_lgbm          | 2d_plus_log2fc_pred  |       1 |     790.0000 | 0.3939 |    0.0000 |     0.8447 |  0.0371 |
| public_chembl_ext_nn_ge025   |                 2.0000 | mordred_lf_pred_lgbm             | 2d_plus_log2fc_pred  |       1 |     790.0000 | 0.3976 |    0.0000 |     0.8399 |  0.0325 |
| public_chembl_ext_nn_ge025   |                 3.0000 | rdkit_full_singleconc_jazzy_lgbm | 2d_plus_aux_observed |       1 |     790.0000 | 0.4163 |    0.0000 |     0.8371 |  0.0294 |
| public_chembl_ext_nn_ge025   |                 4.0000 | mordred_singleconc_jazzy_lgbm    | 2d_plus_aux_observed |       1 |     790.0000 | 0.4201 |    0.0000 |     0.8302 |  0.0293 |
| public_chembl_ext_nn_ge025   |                 5.0000 | mordred_lgbm                     | 2d                   |       1 |     790.0000 | 0.5126 |    0.0000 |     0.7125 |  0.0459 |
| public_hybrid_nolabel_top513 |                 1.0000 | rdkit_full_lf_pred_lgbm          | 2d_plus_log2fc_pred  |       1 |     513.0000 | 0.3300 |    0.0000 |     0.7584 | -0.0809 |
| public_hybrid_nolabel_top513 |                 2.0000 | mordred_lf_pred_lgbm             | 2d_plus_log2fc_pred  |       1 |     513.0000 | 0.3349 |    0.0000 |     0.7537 | -0.0856 |
| public_hybrid_nolabel_top513 |                 3.0000 | rdkit_full_singleconc_jazzy_lgbm | 2d_plus_aux_observed |       1 |     513.0000 | 0.3380 |    0.0000 |     0.7709 | -0.1214 |
| public_hybrid_nolabel_top513 |                 4.0000 | mordred_singleconc_jazzy_lgbm    | 2d_plus_aux_observed |       1 |     513.0000 | 0.3398 |    0.0000 |     0.7730 | -0.1018 |
| public_hybrid_nolabel_top513 |                 5.0000 | mordred_lgbm                     | 2d                   |       1 |     513.0000 | 0.4695 |    0.0000 |     0.4806 | -0.2315 |
| public_hybrid_with_y_top513  |                 1.0000 | rdkit_full_lf_pred_lgbm          | 2d_plus_log2fc_pred  |       1 |     513.0000 | 0.3212 |    0.0000 |     0.6971 | -0.2112 |
| public_hybrid_with_y_top513  |                 2.0000 | mordred_lf_pred_lgbm             | 2d_plus_log2fc_pred  |       1 |     513.0000 | 0.3380 |    0.0000 |     0.6855 | -0.2399 |
| public_hybrid_with_y_top513  |                 3.0000 | mordred_singleconc_jazzy_lgbm    | 2d_plus_aux_observed |       1 |     513.0000 | 0.3418 |    0.0000 |     0.6558 | -0.2513 |
| public_hybrid_with_y_top513  |                 4.0000 | rdkit_full_singleconc_jazzy_lgbm | 2d_plus_aux_observed |       1 |     513.0000 | 0.3431 |    0.0000 |     0.6530 | -0.2520 |
| public_hybrid_with_y_top513  |                 5.0000 | mordred_lgbm                     | 2d                   |       1 |     513.0000 | 0.5285 |    0.0000 |     0.3333 | -0.4611 |
| public_log2fc_top513         |                 1.0000 | rdkit_full_lf_pred_lgbm          | 2d_plus_log2fc_pred  |       1 |     513.0000 | 0.2749 |    0.0000 |     0.6378 | -0.0077 |
| public_log2fc_top513         |                 2.0000 | mordred_lf_pred_lgbm             | 2d_plus_log2fc_pred  |       1 |     513.0000 | 0.2785 |    0.0000 |     0.6316 |  0.0110 |
| public_log2fc_top513         |                 3.0000 | rdkit_full_singleconc_jazzy_lgbm | 2d_plus_aux_observed |       1 |     513.0000 | 0.3565 |    0.0000 |     0.5088 | -0.0889 |
| public_log2fc_top513         |                 4.0000 | mordred_singleconc_jazzy_lgbm    | 2d_plus_aux_observed |       1 |     513.0000 | 0.3569 |    0.0000 |     0.5103 | -0.0635 |
| public_log2fc_top513         |                 5.0000 | mordred_lgbm                     | 2d                   |       1 |     513.0000 | 0.4373 |    0.0000 |     0.3800 | -0.3120 |
| public_testnn_top513         |                 1.0000 | mordred_lf_pred_lgbm             | 2d_plus_log2fc_pred  |       1 |     513.0000 | 0.3597 |    0.0000 |     0.8660 | -0.0616 |
| public_testnn_top513         |                 2.0000 | rdkit_full_lf_pred_lgbm          | 2d_plus_log2fc_pred  |       1 |     513.0000 | 0.3642 |    0.0000 |     0.8662 | -0.0518 |
| public_testnn_top513         |                 3.0000 | rdkit_full_singleconc_jazzy_lgbm | 2d_plus_aux_observed |       1 |     513.0000 | 0.3959 |    0.0000 |     0.8517 | -0.0640 |
| public_testnn_top513         |                 4.0000 | mordred_singleconc_jazzy_lgbm    | 2d_plus_aux_observed |       1 |     513.0000 | 0.3999 |    0.0000 |     0.8503 | -0.0725 |
| public_testnn_top513         |                 5.0000 | mordred_lgbm                     | 2d                   |       1 |     513.0000 | 0.5240 |    0.0000 |     0.6867 | -0.1145 |
| umap_canonical               |                 1.0000 | rdkit_full_lf_pred_lgbm          | 2d_plus_log2fc_pred  |       5 |     828.0000 | 0.3883 |    0.0286 |     0.8510 |  0.0258 |
| umap_canonical               |                 2.0000 | mordred_lf_pred_lgbm             | 2d_plus_log2fc_pred  |       5 |     828.0000 | 0.3903 |    0.0280 |     0.8502 |  0.0203 |
| umap_canonical               |                 3.0000 | rdkit_full_singleconc_jazzy_lgbm | 2d_plus_aux_observed |       5 |     828.0000 | 0.4374 |    0.0276 |     0.8209 |  0.0550 |
| umap_canonical               |                 4.0000 | mordred_singleconc_jazzy_lgbm    | 2d_plus_aux_observed |       5 |     828.0000 | 0.4397 |    0.0292 |     0.8230 |  0.0546 |
| umap_canonical               |                 5.0000 | mordred_lgbm                     | 2d                   |       5 |     828.0000 | 0.5237 |    0.0342 |     0.7084 |  0.0801 |

## OOF-stack diagnostics

| split                        |   rank_mae_within_kind | model                          |   folds |   n_val_mean |    mae |   mae_std |   spearman |    bias |
|:-----------------------------|-----------------------:|:-------------------------------|--------:|-------------:|-------:|----------:|-----------:|--------:|
| public_adv_top513            |                 1.0000 | current_pool_ridge_stack_oof   |       1 |     513.0000 | 0.3720 |    0.0000 |     0.8200 | -0.0949 |
| public_adv_top513            |                 2.0000 | current_pool_caruana_stack_oof |       1 |     513.0000 | 0.3727 |    0.0000 |     0.8180 | -0.0829 |
| public_adv_top513            |                 3.0000 | current_pool_simple_mean_oof   |       1 |     513.0000 | 0.3997 |    0.0000 |     0.7739 | -0.0942 |
| public_chembl_ext_nn_ge025   |                 1.0000 | current_pool_ridge_stack_oof   |       1 |     790.0000 | 0.3909 |    0.0000 |     0.8457 | -0.0048 |
| public_chembl_ext_nn_ge025   |                 2.0000 | current_pool_caruana_stack_oof |       1 |     790.0000 | 0.3960 |    0.0000 |     0.8430 |  0.0101 |
| public_chembl_ext_nn_ge025   |                 3.0000 | current_pool_simple_mean_oof   |       1 |     790.0000 | 0.4235 |    0.0000 |     0.8131 |  0.0171 |
| public_hybrid_nolabel_top513 |                 1.0000 | current_pool_ridge_stack_oof   |       1 |     513.0000 | 0.3255 |    0.0000 |     0.7608 | -0.0634 |
| public_hybrid_nolabel_top513 |                 2.0000 | current_pool_caruana_stack_oof |       1 |     513.0000 | 0.3309 |    0.0000 |     0.7534 | -0.0620 |
| public_hybrid_nolabel_top513 |                 3.0000 | current_pool_simple_mean_oof   |       1 |     513.0000 | 0.3625 |    0.0000 |     0.6874 | -0.0896 |
| public_hybrid_with_y_top513  |                 1.0000 | current_pool_caruana_stack_oof |       1 |     513.0000 | 0.3103 |    0.0000 |     0.6678 | -0.1658 |
| public_hybrid_with_y_top513  |                 2.0000 | current_pool_ridge_stack_oof   |       1 |     513.0000 | 0.3157 |    0.0000 |     0.6752 | -0.1872 |
| public_hybrid_with_y_top513  |                 3.0000 | current_pool_simple_mean_oof   |       1 |     513.0000 | 0.3461 |    0.0000 |     0.5787 | -0.2075 |
| public_log2fc_top513         |                 1.0000 | current_pool_caruana_stack_oof |       1 |     513.0000 | 0.2834 |    0.0000 |     0.6162 | -0.0012 |
| public_log2fc_top513         |                 2.0000 | current_pool_ridge_stack_oof   |       1 |     513.0000 | 0.2848 |    0.0000 |     0.6100 |  0.0271 |
| public_log2fc_top513         |                 3.0000 | current_pool_simple_mean_oof   |       1 |     513.0000 | 0.3084 |    0.0000 |     0.5553 | -0.0667 |
| public_testnn_top513         |                 1.0000 | current_pool_ridge_stack_oof   |       1 |     513.0000 | 0.3600 |    0.0000 |     0.8642 | -0.0625 |
| public_testnn_top513         |                 2.0000 | current_pool_caruana_stack_oof |       1 |     513.0000 | 0.3658 |    0.0000 |     0.8594 | -0.0548 |
| public_testnn_top513         |                 3.0000 | current_pool_simple_mean_oof   |       1 |     513.0000 | 0.4014 |    0.0000 |     0.8268 | -0.0686 |
| umap_canonical               |                 1.0000 | current_pool_ridge_stack_oof   |       5 |     828.0000 | 0.3908 |    0.0276 |     0.8499 |  0.0007 |
| umap_canonical               |                 2.0000 | current_pool_caruana_stack_oof |       5 |     828.0000 | 0.3937 |    0.0266 |     0.8477 |  0.0103 |
| umap_canonical               |                 3.0000 | current_pool_simple_mean_oof   |       5 |     828.0000 | 0.4187 |    0.0278 |     0.8243 |  0.0172 |

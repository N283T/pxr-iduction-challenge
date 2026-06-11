# Phase 2 HTChem top500 SWAP bakeoff

Current top500 member: `tabpfn_cheme_2d_full_boltz_log2fc_pred_seed10ens_top500_umap`

HTChem top500 member: `tabpfn_cheme_2d_full_boltz_log2fc_pred_seed10ens_pred_htchem_top500_umap_v2_6`

## Caruana Bag20 Summary

| variant               |   n_members |     mae |     rae |   spearman |   delta_mae |   delta_spearman |   old_top500_weight |   new_top500_weight |   chemprop_family_share |
|:----------------------|------------:|--------:|--------:|-----------:|------------:|-----------------:|--------------------:|--------------------:|------------------------:|
| baseline              |           9 | 0.39576 | 0.43493 |    0.84665 |     0.00000 |          0.00000 |             0.33748 |             0.00000 |                 0.77650 |
| swap_top500_to_htchem |           9 | 0.39662 | 0.43588 |    0.84520 |     0.00087 |         -0.00145 |             0.00000 |             0.34660 |                 0.78447 |
| add_htchem_top500     |          10 | 0.39218 | 0.43100 |    0.85006 |    -0.00357 |          0.00341 |             0.30155 |             0.19350 |                 0.89437 |

## Residual Correlation

| member                                                                      |   pearson_vs_new_oof |
|:----------------------------------------------------------------------------|---------------------:|
| tabpfn_cheme_2d_full_boltz_log2fc_pred_seed10ens_top500_umap                |              0.99749 |
| tabpfn_cheme_2d_full_boltz_log2fc_pred_optuna_trial10_seed5ens_umap_default |              0.98611 |
| tabpfn_chemprop_pretrain_embed_umap_default                                 |              0.96580 |
| tabpfn_kermt_pretrain_embed_umap_default                                    |              0.94541 |
| tabpfn_gatedgcn_pretrain_embed_umap_default                                 |              0.93178 |
| tabpfn_attentivefp_pretrain_embed_umap_default                              |              0.92632 |
| tabpfn_molformer_c3_pretrain_embed_umap                                     |              0.92570 |
| tabpfn_pooled_boltz_allpairs_umap_default                                   |              0.90746 |
| tabpfn_pooled_boltz_umap_default                                            |              0.90460 |

## Weights

| variant               | member                                                                        |   weight | is_old_top500   | is_new_top500   |
|:----------------------|:------------------------------------------------------------------------------|---------:|:----------------|:----------------|
| baseline              | tabpfn_cheme_2d_full_boltz_log2fc_pred_seed10ens_top500_umap                  |  0.33748 | True            | False           |
| baseline              | tabpfn_cheme_2d_full_boltz_log2fc_pred_optuna_trial10_seed5ens_umap_default   |  0.32738 | False           | False           |
| baseline              | tabpfn_chemprop_pretrain_embed_umap_default                                   |  0.11165 | False           | False           |
| baseline              | tabpfn_kermt_pretrain_embed_umap_default                                      |  0.07553 | False           | False           |
| baseline              | tabpfn_pooled_boltz_allpairs_umap_default                                     |  0.04039 | False           | False           |
| baseline              | tabpfn_molformer_c3_pretrain_embed_umap                                       |  0.03845 | False           | False           |
| baseline              | tabpfn_pooled_boltz_umap_default                                              |  0.03165 | False           | False           |
| baseline              | tabpfn_gatedgcn_pretrain_embed_umap_default                                   |  0.02184 | False           | False           |
| baseline              | tabpfn_attentivefp_pretrain_embed_umap_default                                |  0.01563 | False           | False           |
| swap_top500_to_htchem | tabpfn_cheme_2d_full_boltz_log2fc_pred_seed10ens_pred_htchem_top500_umap_v2_6 |  0.34660 | False           | True            |
| swap_top500_to_htchem | tabpfn_cheme_2d_full_boltz_log2fc_pred_optuna_trial10_seed5ens_umap_default   |  0.32650 | False           | False           |
| swap_top500_to_htchem | tabpfn_chemprop_pretrain_embed_umap_default                                   |  0.11136 | False           | False           |
| swap_top500_to_htchem | tabpfn_kermt_pretrain_embed_umap_default                                      |  0.07544 | False           | False           |
| swap_top500_to_htchem | tabpfn_pooled_boltz_allpairs_umap_default                                     |  0.03767 | False           | False           |
| swap_top500_to_htchem | tabpfn_molformer_c3_pretrain_embed_umap                                       |  0.03680 | False           | False           |
| swap_top500_to_htchem | tabpfn_pooled_boltz_umap_default                                              |  0.02816 | False           | False           |
| swap_top500_to_htchem | tabpfn_gatedgcn_pretrain_embed_umap_default                                   |  0.02184 | False           | False           |
| swap_top500_to_htchem | tabpfn_attentivefp_pretrain_embed_umap_default                                |  0.01563 | False           | False           |
| add_htchem_top500     | tabpfn_cheme_2d_full_boltz_log2fc_pred_optuna_trial10_seed5ens_umap_default   |  0.35767 | False           | False           |
| add_htchem_top500     | tabpfn_cheme_2d_full_boltz_log2fc_pred_seed10ens_top500_umap                  |  0.30155 | True            | False           |
| add_htchem_top500     | tabpfn_cheme_2d_full_boltz_log2fc_pred_seed10ens_pred_htchem_top500_umap_v2_6 |  0.19350 | False           | True            |
| add_htchem_top500     | tabpfn_chemprop_pretrain_embed_umap_default                                   |  0.04165 | False           | False           |
| add_htchem_top500     | tabpfn_kermt_pretrain_embed_umap_default                                      |  0.04146 | False           | False           |
| add_htchem_top500     | tabpfn_pooled_boltz_umap_default                                              |  0.02097 | False           | False           |
| add_htchem_top500     | tabpfn_pooled_boltz_allpairs_umap_default                                     |  0.01883 | False           | False           |
| add_htchem_top500     | tabpfn_molformer_c3_pretrain_embed_umap                                       |  0.01641 | False           | False           |
| add_htchem_top500     | tabpfn_gatedgcn_pretrain_embed_umap_default                                   |  0.00466 | False           | False           |
| add_htchem_top500     | tabpfn_attentivefp_pretrain_embed_umap_default                                |  0.00330 | False           | False           |

## Read

SWAP is preferred over ADD if it improves OOF without increasing correlated family share. ADD is diagnostic only because two highly related top500 members can concentrate the same family axis.

# Phase 2 ensemble weight analysis

Old `ens_caruana_bag20` weights applied to the Phase 2 `train + AS1`
OOF matrix, compared with re-optimized Phase 2 weights.

## Old weights

| member                                                                      |   weight |
|:----------------------------------------------------------------------------|---------:|
| tabpfn_cheme_2d_full_boltz_log2fc_pred_optuna_trial10_seed5ens_top500_umap  |   0.3092 |
| tabpfn_cheme_2d_full_boltz_log2fc_pred_optuna_trial10_seed5ens_umap_default |   0.2879 |
| tabpfn_chemprop_pretrain_embed_umap_default                                 |   0.1515 |
| tabpfn_kermt_pretrain_embed_umap_default                                    |   0.1107 |
| tabpfn_pooled_boltz_umap_default                                            |   0.0456 |
| tabpfn_molformer_c3_pretrain_embed_umap                                     |   0.0403 |
| tabpfn_pooled_boltz_allpairs_umap_default                                   |   0.0350 |
| tabpfn_gatedgcn_pretrain_embed_umap_default                                 |   0.0175 |
| tabpfn_attentivefp_pretrain_embed_umap_default                              |   0.0024 |

## Summary

| setting                       | slice   |    n |    mae |   bias_pred_minus_true |   spearman |   pred_mean |   true_mean |
|:------------------------------|:--------|-----:|-------:|-----------------------:|-----------:|------------:|------------:|
| phase2_vanilla_opt            | all     | 4393 | 0.3916 |                 0.0161 |     0.8524 |      4.3567 |      4.3406 |
| phase2_caruana_bag20          | all     | 4393 | 0.3995 |                 0.0166 |     0.8451 |      4.3572 |      4.3406 |
| old_ens_caruana_bag20_weights | all     | 4393 | 0.4024 |                 0.0173 |     0.8433 |      4.3578 |      4.3406 |
| phase2_l2_0p3                 | all     | 4393 | 0.4113 |                 0.0184 |     0.8366 |      4.3590 |      4.3406 |
| simple_average_old_members    | all     | 4393 | 0.4230 |                 0.0195 |     0.8265 |      4.3601 |      4.3406 |

## Slice scoreboard

| setting                       |   all_mae |   source_as1_mae |   true_lt3_mae |   true_gte6_mae |   all_spearman |   source_as1_spearman |
|:------------------------------|----------:|-----------------:|---------------:|----------------:|---------------:|----------------------:|
| phase2_vanilla_opt            |    0.3916 |           0.4371 |         0.6645 |          0.9877 |         0.8524 |                0.8269 |
| phase2_caruana_bag20          |    0.3995 |           0.4386 |         0.6829 |          1.0493 |         0.8451 |                0.8298 |
| old_ens_caruana_bag20_weights |    0.4024 |           0.4387 |         0.6936 |          1.0662 |         0.8433 |                0.8335 |
| phase2_l2_0p3                 |    0.4113 |           0.4414 |         0.7178 |          1.1049 |         0.8366 |                0.8311 |
| simple_average_old_members    |    0.4230 |           0.4471 |         0.7432 |          1.1443 |         0.8265 |                0.8287 |

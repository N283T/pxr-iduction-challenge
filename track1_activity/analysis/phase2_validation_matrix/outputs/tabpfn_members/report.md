# Phase 2 TabPFN member OOF

These are `train + AS1` cross-fit OOF results for current TabPFN
ensemble members. TabPFN uses v2.6 by default here.

## All labeled rows

| member                                                                      | feature                                                 |   top_k |    n |    mae |   bias_pred_minus_true |   spearman |
|:----------------------------------------------------------------------------|:--------------------------------------------------------|--------:|-----:|-------:|-----------------------:|-----------:|
| tabpfn_cheme_2d_full_boltz_log2fc_pred_optuna_trial10_seed5ens_top500_umap  | cheme_2d_full_boltz_log2fc_pred_optuna_trial10_seed5ens |     500 | 4393 | 0.3877 |                 0.0118 |     0.8553 |
| tabpfn_cheme_2d_full_boltz_log2fc_pred_seed10ens_top500_umap                | cheme_2d_full_boltz_log2fc_pred_seed10ens               |     500 | 4393 | 0.3961 |                 0.0092 |     0.8498 |
| tabpfn_cheme_2d_full_boltz_log2fc_pred_optuna_trial10_seed5ens_umap_default | cheme_2d_full_boltz_log2fc_pred_optuna_trial10_seed5ens |       0 | 4393 | 0.4293 |                 0.0197 |     0.8157 |
| tabpfn_chemprop_pretrain_embed_umap_default                                 | chemprop_pretrain_embed                                 |       0 | 4393 | 0.4357 |                 0.0183 |     0.8120 |
| tabpfn_kermt_pretrain_embed_umap_default                                    | kermt_pretrain_embed                                    |       0 | 4393 | 0.4494 |                 0.0207 |     0.7958 |
| tabpfn_gatedgcn_pretrain_embed_umap_default                                 | gatedgcn_pretrain_embed                                 |       0 | 4393 | 0.4754 |                 0.0314 |     0.7708 |
| tabpfn_molformer_c3_pretrain_embed_umap                                     | molformer_c3_pretrain_embed                             |       0 | 4393 | 0.4818 |                 0.0035 |     0.7648 |
| tabpfn_attentivefp_pretrain_embed_umap_default                              | attentivefp_pretrain_embed                              |       0 | 4393 | 0.4840 |                 0.0174 |     0.7685 |
| tabpfn_pooled_boltz_allpairs_umap_default                                   | pooled_boltz_allpairs                                   |       0 | 4393 | 0.4857 |                 0.0254 |     0.7644 |
| tabpfn_pooled_boltz_umap_default                                            | pooled_boltz                                            |       0 | 4393 | 0.4878 |                 0.0272 |     0.7580 |

## Slice scoreboard

| member                                                                      | feature                                                 |   top_k |   all_mae |   source_as1_mae |   true_lt3_mae |   true_gte6_mae |   all_spearman |   source_as1_spearman |
|:----------------------------------------------------------------------------|:--------------------------------------------------------|--------:|----------:|-----------------:|---------------:|----------------:|---------------:|----------------------:|
| tabpfn_cheme_2d_full_boltz_log2fc_pred_optuna_trial10_seed5ens_top500_umap  | cheme_2d_full_boltz_log2fc_pred_optuna_trial10_seed5ens |     500 |    0.3877 |           0.4505 |         0.6218 |          0.8786 |         0.8553 |                0.8103 |
| tabpfn_cheme_2d_full_boltz_log2fc_pred_seed10ens_top500_umap                | cheme_2d_full_boltz_log2fc_pred_seed10ens               |     500 |    0.3961 |           0.4242 |         0.6229 |          0.9142 |         0.8498 |                0.8319 |
| tabpfn_cheme_2d_full_boltz_log2fc_pred_optuna_trial10_seed5ens_umap_default | cheme_2d_full_boltz_log2fc_pred_optuna_trial10_seed5ens |       0 |    0.4293 |           0.4621 |         0.7388 |          1.1387 |         0.8157 |                0.8121 |
| tabpfn_chemprop_pretrain_embed_umap_default                                 | chemprop_pretrain_embed                                 |       0 |    0.4357 |           0.4532 |         0.6939 |          1.0997 |         0.8120 |                0.8226 |
| tabpfn_kermt_pretrain_embed_umap_default                                    | kermt_pretrain_embed                                    |       0 |    0.4494 |           0.4731 |         0.7205 |          1.1940 |         0.7958 |                0.7956 |
| tabpfn_gatedgcn_pretrain_embed_umap_default                                 | gatedgcn_pretrain_embed                                 |       0 |    0.4754 |           0.5008 |         0.7914 |          1.2378 |         0.7708 |                0.7392 |
| tabpfn_molformer_c3_pretrain_embed_umap                                     | molformer_c3_pretrain_embed                             |       0 |    0.4818 |           0.5123 |         0.7687 |          1.3003 |         0.7648 |                0.7677 |
| tabpfn_attentivefp_pretrain_embed_umap_default                              | attentivefp_pretrain_embed                              |       0 |    0.4840 |           0.4975 |         0.7964 |          1.1902 |         0.7685 |                0.7800 |
| tabpfn_pooled_boltz_allpairs_umap_default                                   | pooled_boltz_allpairs                                   |       0 |    0.4857 |           0.4849 |         0.8209 |          1.1242 |         0.7644 |                0.7650 |
| tabpfn_pooled_boltz_umap_default                                            | pooled_boltz                                            |       0 |    0.4878 |           0.4974 |         0.8324 |          1.1350 |         0.7580 |                0.7566 |

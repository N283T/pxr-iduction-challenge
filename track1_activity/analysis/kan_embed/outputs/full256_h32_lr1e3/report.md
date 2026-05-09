# KAN on Frozen Embedding Report

Run name: `full256_h32_lr1e3`
Experiment: `kan_chemprop_pretrain_embed_full256_h32_lr1e3_umap`
Embedding: `/home/nagaet/pxr-iduction-challenge/data/chemprop_pretrain_embed.parquet`

## OOF Metrics

Coverage: `4140 / 4140`
MAE: `0.467550`
RAE: `0.513831`
Spearman: `0.790523`
Residual r vs ens_caruana_bag20: `0.886337`

## Fold Metrics

|   fold |   epochs_run |   best_val_mae |   loss_start |   loss_end |   input_dim |   pca_variance |   target_mean |   target_std |   val_MAE |   val_RAE |   val_R2 |   val_Spearman_R |   val_Kendall_Tau |
|-------:|-------------:|---------------:|-------------:|-----------:|------------:|---------------:|--------------:|-------------:|----------:|----------:|---------:|-----------------:|------------------:|
|      0 |           45 |       0.506765 |     0.815185 |  0.108484  |         256 |            nan |       4.30623 |      1.11576 |  0.506765 |  0.55906  | 0.635952 |         0.756642 |          0.565466 |
|      1 |           41 |       0.466088 |     0.841919 |  0.119986  |         256 |            nan |       4.34846 |      1.10393 |  0.466088 |  0.469174 | 0.714369 |         0.829654 |          0.632026 |
|      2 |           36 |       0.470025 |     0.857868 |  0.139297  |         256 |            nan |       4.3286  |      1.14126 |  0.470025 |  0.553418 | 0.618771 |         0.795676 |          0.602243 |
|      3 |           49 |       0.421015 |     0.775316 |  0.0831361 |         256 |            nan |       4.28664 |      1.14544 |  0.421015 |  0.540015 | 0.640704 |         0.749497 |          0.570328 |
|      4 |           49 |       0.473126 |     0.810501 |  0.0772436 |         256 |            nan |       4.33425 |      1.09895 |  0.473126 |  0.471066 | 0.693933 |         0.804326 |          0.609218 |

## Decision Gate

Use for deeper KAN/GNN-KAN only if it reaches MAE <= 0.47 or is clearly decorrelated with acceptable Spearman.

## Final Read

KAN is competitive enough for a Caruana ADD/SWAP bakeoff before deciding.

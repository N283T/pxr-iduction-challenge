# KAN on Frozen Embedding Report

Run name: `ka_gnn_pretrain_full_h32_lr1e3`
Experiment: `kan_ka_gnn_pretrain_embed_ka_gnn_pretrain_full_h32_lr1e3_umap`
Embedding: `data/ka_gnn_pretrain_embed.parquet`

## OOF Metrics

Coverage: `4140 / 4140`
MAE: `0.537695`
RAE: `0.590919`
Spearman: `0.720651`
Residual r vs ens_caruana_bag20: `0.817944`

## Fold Metrics

|   fold |   epochs_run |   best_val_mae |   loss_start |   loss_end |   input_dim |   pca_variance |   target_mean |   target_std |   val_MAE |   val_RAE |   val_R2 |   val_Spearman_R |   val_Kendall_Tau |
|-------:|-------------:|---------------:|-------------:|-----------:|------------:|---------------:|--------------:|-------------:|----------:|----------:|---------:|-----------------:|------------------:|
|      0 |           60 |       0.579944 |     0.725276 |   0.2007   |         128 |            nan |       4.30623 |      1.11576 |  0.579944 |  0.639791 | 0.522127 |         0.688149 |          0.498212 |
|      1 |           74 |       0.531692 |     0.846983 |   0.172951 |         128 |            nan |       4.34846 |      1.10393 |  0.531692 |  0.535212 | 0.639057 |         0.759594 |          0.561997 |
|      2 |           40 |       0.523119 |     0.873284 |   0.269116 |         128 |            nan |       4.3286  |      1.14126 |  0.523119 |  0.615932 | 0.544815 |         0.71996  |          0.527068 |
|      3 |           54 |       0.499919 |     0.817895 |   0.247605 |         128 |            nan |       4.28664 |      1.14544 |  0.499919 |  0.64122  | 0.538751 |         0.673312 |          0.491391 |
|      4 |           46 |       0.553148 |     0.8896   |   0.246302 |         128 |            nan |       4.33425 |      1.09895 |  0.553148 |  0.55074  | 0.598565 |         0.750341 |          0.552561 |

## Decision Gate

Use for deeper KAN/GNN-KAN only if it reaches MAE <= 0.47 or is clearly decorrelated with acceptable Spearman.

## Final Read

KAN is below the current ChemProp embedding + TabPFN sibling; do not submit directly.

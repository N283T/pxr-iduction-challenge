# KAN on Frozen Embedding Report

Run name: `pca128_h16_lr1e3`
Experiment: `kan_chemprop_pretrain_embed_pca128_h16_lr1e3_umap`
Embedding: `/home/nagaet/pxr-iduction-challenge/data/chemprop_pretrain_embed.parquet`

## OOF Metrics

Coverage: `4140 / 4140`
MAE: `0.542143`
RAE: `0.595807`
Spearman: `0.741265`
Residual r vs ens_caruana_bag20: `0.783388`

## Fold Metrics

|   fold |   epochs_run |   best_val_mae |   loss_start |   loss_end |   input_dim |   pca_variance |   target_mean |   target_std |   val_MAE |   val_RAE |   val_R2 |   val_Spearman_R |   val_Kendall_Tau |
|-------:|-------------:|---------------:|-------------:|-----------:|------------:|---------------:|--------------:|-------------:|----------:|----------:|---------:|-----------------:|------------------:|
|      0 |           64 |       0.55962  |     0.974131 |   0.141958 |         128 |       0.977275 |       4.30623 |      1.11576 |  0.55962  |  0.617369 | 0.561982 |         0.704687 |          0.51637  |
|      1 |           66 |       0.539618 |     1.0141   |   0.140888 |         128 |       0.976709 |       4.34846 |      1.10393 |  0.539618 |  0.543191 | 0.633912 |         0.779907 |          0.578236 |
|      2 |           60 |       0.530372 |     0.989205 |   0.149923 |         128 |       0.97772  |       4.3286  |      1.14126 |  0.530372 |  0.624472 | 0.551769 |         0.742956 |          0.546838 |
|      3 |           64 |       0.493661 |     0.982939 |   0.152234 |         128 |       0.976708 |       4.28664 |      1.14544 |  0.493661 |  0.633194 | 0.526311 |         0.698816 |          0.517235 |
|      4 |           71 |       0.586599 |     0.997011 |   0.128937 |         128 |       0.976538 |       4.33425 |      1.09895 |  0.586599 |  0.584045 | 0.580691 |         0.757424 |          0.54982  |

## Decision Gate

Use for deeper KAN/GNN-KAN only if it reaches MAE <= 0.47 or is clearly decorrelated with acceptable Spearman.

## Final Read

KAN is below the current ChemProp embedding + TabPFN sibling; do not submit directly.

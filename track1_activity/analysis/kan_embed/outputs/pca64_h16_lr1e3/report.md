# KAN on Frozen Embedding Report

Run name: `pca64_h16_lr1e3`
Experiment: `kan_chemprop_pretrain_embed_pca64_h16_lr1e3_umap`
Embedding: `/home/nagaet/pxr-iduction-challenge/data/chemprop_pretrain_embed.parquet`

## OOF Metrics

Coverage: `4140 / 4140`
MAE: `0.518814`
RAE: `0.570170`
Spearman: `0.754327`
Residual r vs ens_caruana_bag20: `0.825908`

## Fold Metrics

|   fold |   epochs_run |   best_val_mae |   loss_start |   loss_end |   input_dim |   pca_variance |   target_mean |   target_std |   val_MAE |   val_RAE |   val_R2 |   val_Spearman_R |   val_Kendall_Tau |
|-------:|-------------:|---------------:|-------------:|-----------:|------------:|---------------:|--------------:|-------------:|----------:|----------:|---------:|-----------------:|------------------:|
|      0 |           80 |       0.545435 |     1.0023   |   0.226001 |          64 |       0.913291 |       4.30623 |      1.11576 |  0.545435 |  0.60172  | 0.578881 |         0.720847 |          0.531816 |
|      1 |           81 |       0.502716 |     0.990193 |   0.234115 |          64 |       0.91135  |       4.34846 |      1.10393 |  0.502716 |  0.506044 | 0.672583 |         0.804319 |          0.601643 |
|      2 |           91 |       0.50666  |     1.00429  |   0.222685 |          64 |       0.91486  |       4.3286  |      1.14126 |  0.50666  |  0.596552 | 0.56189  |         0.767199 |          0.572357 |
|      3 |           77 |       0.466455 |     0.995749 |   0.226167 |          64 |       0.91167  |       4.28664 |      1.14544 |  0.466455 |  0.598298 | 0.553401 |         0.704888 |          0.527422 |
|      4 |           85 |       0.571874 |     1.00199  |   0.233951 |          64 |       0.910897 |       4.33425 |      1.09895 |  0.571874 |  0.569385 | 0.599066 |         0.756032 |          0.549067 |

## Decision Gate

Use for deeper KAN/GNN-KAN only if it reaches MAE <= 0.47 or is clearly decorrelated with acceptable Spearman.

## Final Read

KAN is below the current ChemProp embedding + TabPFN sibling; do not submit directly.

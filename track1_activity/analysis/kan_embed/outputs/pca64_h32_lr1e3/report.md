# KAN on Frozen Embedding Report

Run name: `pca64_h32_lr1e3`
Experiment: `kan_chemprop_pretrain_embed_pca64_h32_lr1e3_umap`
Embedding: `/home/nagaet/pxr-iduction-challenge/data/chemprop_pretrain_embed.parquet`

## OOF Metrics

Coverage: `4140 / 4140`
MAE: `0.518163`
RAE: `0.569453`
Spearman: `0.757641`
Residual r vs ens_caruana_bag20: `0.824072`

## Fold Metrics

|   fold |   epochs_run |   best_val_mae |   loss_start |   loss_end |   input_dim |   pca_variance |   target_mean |   target_std |   val_MAE |   val_RAE |   val_R2 |   val_Spearman_R |   val_Kendall_Tau |
|-------:|-------------:|---------------:|-------------:|-----------:|------------:|---------------:|--------------:|-------------:|----------:|----------:|---------:|-----------------:|------------------:|
|      0 |           64 |       0.541895 |     0.979633 |   0.170332 |          64 |       0.913291 |       4.30623 |      1.11576 |  0.541895 |  0.597815 | 0.579838 |         0.724125 |          0.533001 |
|      1 |           76 |       0.514217 |     0.99301  |   0.157701 |          64 |       0.91135  |       4.34846 |      1.10393 |  0.514218 |  0.517622 | 0.661042 |         0.796311 |          0.594359 |
|      2 |           74 |       0.504029 |     1.01423  |   0.157381 |          64 |       0.91486  |       4.3286  |      1.14126 |  0.504029 |  0.593454 | 0.582248 |         0.769735 |          0.574114 |
|      3 |           65 |       0.462607 |     1.00824  |   0.183758 |          64 |       0.91167  |       4.28664 |      1.14544 |  0.462607 |  0.593363 | 0.579367 |         0.714448 |          0.534641 |
|      4 |           79 |       0.567098 |     1.01377  |   0.157393 |          64 |       0.910897 |       4.33425 |      1.09895 |  0.567098 |  0.564629 | 0.608673 |         0.760601 |          0.552144 |

## Decision Gate

Use for deeper KAN/GNN-KAN only if it reaches MAE <= 0.47 or is clearly decorrelated with acceptable Spearman.

## Final Read

KAN is below the current ChemProp embedding + TabPFN sibling; do not submit directly.

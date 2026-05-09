# KAN on Frozen Embedding Report

Run name: `pca64_h16_lbfgs`
Experiment: `kan_chemprop_pretrain_embed_pca64_h16_lbfgs_umap`
Embedding: `/home/nagaet/pxr-iduction-challenge/data/chemprop_pretrain_embed.parquet`

## OOF Metrics

Coverage: `4140 / 4140`
MAE: `0.654099`
RAE: `0.718845`
Spearman: `0.648741`
Residual r vs ens_caruana_bag20: `0.681690`

## Fold Metrics

|   fold |   epochs_run |   best_val_mae |   loss_start |   loss_end |   input_dim |   pca_variance |   target_mean |   target_std |   val_MAE |   val_RAE |   val_R2 |   val_Spearman_R |   val_Kendall_Tau |
|-------:|-------------:|---------------:|-------------:|-----------:|------------:|---------------:|--------------:|-------------:|----------:|----------:|---------:|-----------------:|------------------:|
|      0 |           31 |       0.536147 |     1.01592  |   0.22223  |          64 |       0.913291 |       4.30623 |      1.11576 |  0.536147 |  0.591474 | 0.582638 |         0.730407 |          0.534883 |
|      1 |           14 |       0.792061 |     0.993926 |   0.704405 |          64 |       0.91135  |       4.34846 |      1.10393 |  0.792061 |  0.797305 | 0.270747 |         0.652699 |          0.462255 |
|      2 |           14 |       0.715446 |     1.01703  |   0.795136 |          64 |       0.91486  |       4.3286  |      1.14126 |  0.715446 |  0.842382 | 0.246419 |         0.613809 |          0.43213  |
|      3 |           30 |       0.450515 |     1.0053   |   0.2145   |          64 |       0.91167  |       4.28664 |      1.14544 |  0.450515 |  0.577853 | 0.587315 |         0.707202 |          0.527092 |
|      4 |           16 |       0.773309 |     1.03317  |   0.63832  |          64 |       0.910897 |       4.33425 |      1.09895 |  0.773309 |  0.769943 | 0.219724 |         0.661973 |          0.469204 |

## Decision Gate

Use for deeper KAN/GNN-KAN only if it reaches MAE <= 0.47 or is clearly decorrelated with acceptable Spearman.

## Final Read

KAN is below the current ChemProp embedding + TabPFN sibling; do not submit directly.

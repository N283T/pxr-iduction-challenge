# KAN on Frozen Embedding Report

Run name: `smoke_pca16_h4`
Experiment: `kan_chemprop_pretrain_embed_smoke_pca16_h4_umap`
Embedding: `/home/nagaet/pxr-iduction-challenge/data/chemprop_pretrain_embed.parquet`

## OOF Metrics

Coverage: `1661 / 4140`
MAE: `0.963266`
RAE: `1.010322`
Spearman: `-0.139428`
Residual r vs ens_caruana_bag20: `nan`

## Fold Metrics

|   fold |   epochs_run |   best_val_mae |   loss_start |   loss_end |   input_dim |   pca_variance |   target_mean |   target_std |   val_MAE |   val_RAE |     val_R2 |   val_Spearman_R |   val_Kendall_Tau |
|-------:|-------------:|---------------:|-------------:|-----------:|------------:|---------------:|--------------:|-------------:|----------:|----------:|-----------:|-----------------:|------------------:|
|      0 |            2 |       0.960007 |      1.0306  |    1.03273 |          16 |       0.713727 |       4.30623 |      1.11576 |  0.960008 |  1.05907  | -0.0437231 |      -0.263567   |       -0.179355   |
|      1 |            2 |       0.966528 |      1.00881 |    1.00596 |          16 |       0.712679 |       4.34846 |      1.10393 |  0.966528 |  0.972928 | -0.0120383 |       0.00794332 |        0.00591582 |

## Decision Gate

Use for deeper KAN/GNN-KAN only if it reaches MAE <= 0.47 or is clearly decorrelated with acceptable Spearman.

## Final Read

KAN is below the current ChemProp embedding + TabPFN sibling; do not submit directly.

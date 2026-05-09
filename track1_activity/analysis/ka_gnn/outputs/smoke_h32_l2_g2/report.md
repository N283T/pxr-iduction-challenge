# KA-GNN Report

Run name: `smoke_h32_l2_g2`
Experiment: `ka_gnn_smoke_h32_l2_g2_umap`
Reference implementation: `https://github.com/LongLee220/KA-GNN`

## OOF Metrics

Coverage: `831 / 4140`
MAE: `0.813224`
RAE: `0.897144`
Spearman: `0.388441`
Residual r vs ens_caruana_bag20: `nan`

## Fold Metrics

|   fold |   epochs_run |   best_val_mae |   loss_start |   loss_end |   target_mean |   target_std |   node_dim |   edge_dim |   val_MAE |   val_RAE |   val_R2 |   val_Spearman_R |   val_Kendall_Tau |
|-------:|-------------:|---------------:|-------------:|-----------:|--------------:|-------------:|-----------:|-----------:|----------:|----------:|---------:|-----------------:|------------------:|
|      0 |            2 |       0.813224 |      1.06704 |   0.819151 |       4.30623 |      1.11576 |          9 |          3 |  0.813224 |  0.897143 | 0.174495 |         0.388441 |          0.268824 |

## Decision Gate

Continue only if MAE <= 0.49 and residual correlation/candidate correlations show a new axis.

## Final Read

Do not submit directly; compare Caruana ADD and correlations first.

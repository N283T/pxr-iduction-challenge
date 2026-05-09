# KA-GNN Report

Run name: `fourier_h64_l2_g2_meanaggr_lr5e4`
Experiment: `ka_gnn_fourier_h64_l2_g2_meanaggr_lr5e4_umap`
Reference implementation: `https://github.com/LongLee220/KA-GNN`

## OOF Metrics

Coverage: `4140 / 4140`
MAE: `0.582867`
RAE: `0.640563`
Spearman: `0.664112`
Residual r vs ens_caruana_bag20: `0.771480`

## Fold Metrics

|   fold |   epochs_run |   best_val_mae |   loss_start |   loss_end |   target_mean |   target_std |   node_dim |   edge_dim |   val_MAE |   val_RAE |   val_R2 |   val_Spearman_R |   val_Kendall_Tau |
|-------:|-------------:|---------------:|-------------:|-----------:|--------------:|-------------:|-----------:|-----------:|----------:|----------:|---------:|-----------------:|------------------:|
|      0 |           54 |       0.644272 |     0.949149 |   0.469699 |       4.30623 |      1.11576 |          9 |          3 |  0.644272 |  0.710756 | 0.402748 |         0.593304 |          0.419033 |
|      1 |          140 |       0.575396 |     0.926435 |   0.413659 |       4.34846 |      1.10393 |          9 |          3 |  0.575396 |  0.579206 | 0.577385 |         0.712368 |          0.515741 |
|      2 |          140 |       0.545236 |     1.00517  |   0.386393 |       4.3286  |      1.14126 |          9 |          3 |  0.545236 |  0.641972 | 0.466486 |         0.675949 |          0.490377 |
|      3 |          119 |       0.537434 |     0.984964 |   0.408344 |       4.28664 |      1.14544 |          9 |          3 |  0.537433 |  0.689339 | 0.398549 |         0.610542 |          0.438114 |
|      4 |           93 |       0.611156 |     0.967602 |   0.431517 |       4.33425 |      1.09895 |          9 |          3 |  0.611156 |  0.608496 | 0.535913 |         0.713083 |          0.514481 |

## Decision Gate

Continue only if MAE <= 0.49 and residual correlation/candidate correlations show a new axis.

## Final Read

Do not submit directly; compare Caruana ADD and correlations first.

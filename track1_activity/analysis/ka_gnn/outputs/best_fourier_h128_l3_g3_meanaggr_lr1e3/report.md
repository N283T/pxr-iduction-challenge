# KA-GNN Report

Run name: `best_fourier_h128_l3_g3_meanaggr_lr1e3`
Experiment: `ka_gnn_best_fourier_h128_l3_g3_meanaggr_lr1e3_umap`
Reference implementation: `https://github.com/LongLee220/KA-GNN`

## OOF Metrics

Coverage: `4140 / 4140`
MAE: `0.569189`
RAE: `0.625530`
Spearman: `0.659184`
Residual r vs ens_caruana_bag20: `0.785554`

## Fold Metrics

|   fold |   epochs_run |   best_val_mae |   loss_start |   loss_end |   target_mean |   target_std |   node_dim |   edge_dim |   val_MAE |   val_RAE |   val_R2 |   val_Spearman_R |   val_Kendall_Tau |
|-------:|-------------:|---------------:|-------------:|-----------:|--------------:|-------------:|-----------:|-----------:|----------:|----------:|---------:|-----------------:|------------------:|
|      0 |           41 |       0.631718 |      1.15703 |   0.371422 |       4.30623 |      1.11576 |          9 |          3 |  0.631718 |  0.696907 | 0.404312 |         0.601568 |          0.424656 |
|      1 |           69 |       0.54079  |      1.22025 |   0.272217 |       4.34846 |      1.10393 |          9 |          3 |  0.54079  |  0.54437  | 0.613074 |         0.728687 |          0.531899 |
|      2 |           53 |       0.542127 |      1.16591 |   0.32687  |       4.3286  |      1.14126 |          9 |          3 |  0.542127 |  0.638312 | 0.488936 |         0.673112 |          0.485725 |
|      3 |           53 |       0.539141 |      1.11272 |   0.338287 |       4.28664 |      1.14544 |          9 |          3 |  0.539141 |  0.691529 | 0.390674 |         0.607374 |          0.434912 |
|      4 |           27 |       0.591572 |      1.33413 |   0.446533 |       4.33425 |      1.09895 |          9 |          3 |  0.591572 |  0.588996 | 0.557412 |         0.692002 |          0.497411 |

## Decision Gate

Continue only if MAE <= 0.49 and residual correlation/candidate correlations show a new axis.

## Final Read

Do not submit directly and do not add to the pool. KA-GNN is decorrelated, but too weak for Caruana to use.

## Caruana ADD Diagnostic

One-off ADD bakeoff against the current 9-member `ENSEMBLE_MODELS` assigned this recorded KA-GNN member `0.0000` Caruana weight. The extended-pool OOF improvement was a pool-size/reallocation artifact among existing top members, not KA-GNN contribution.

OOF Pearson correlations vs key members for the recorded run:

- `tabpfn_chemprop_pretrain_embed_umap_default`: `0.8433`
- `tabpfn_gatedgcn_pretrain_embed_umap_default`: `0.8705`
- `tabpfn_kermt_pretrain_embed_umap_default`: `0.8534`
- `kan_chemprop_pretrain_embed_best_full256_h32_lr1e3_umap`: `0.8238`

Interpretation: KA-GNN is more decorrelated than expected, but direct pEC50 accuracy is too weak for the pool optimizer to use it.

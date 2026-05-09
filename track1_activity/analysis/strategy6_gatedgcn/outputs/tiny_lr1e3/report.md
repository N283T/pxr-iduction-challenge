# Buterez Strategy 6 GatedGCN Report

Run name: `tiny_lr1e3`
Experiment: `gatedgcn_strategy6_adaptive_readout_tiny_lr1e3_umap`
Pretrain checkpoint: `/home/nagaet/pxr-iduction-challenge/track1_activity/checkpoints/gatedgcn_pretrain/pretrain.pt`

## OOF Metrics

MAE: `0.484016`
RAE: `0.531927`
Spearman: `0.755892`

## Fold Metrics

|   fold |   epochs_run |   best_val_mae |   loss_start |   loss_end |   pretrain_best_val_loss |   val_MAE |   val_RAE |   val_R2 |   val_Spearman_R |   val_Kendall_Tau |
|-------:|-------------:|---------------:|-------------:|-----------:|-------------------------:|----------:|----------:|---------:|-----------------:|------------------:|
|      0 |           55 |       0.517836 |      2.46233 |   0.417919 |                  0.74776 |  0.517836 |  0.571274 | 0.589662 |         0.726215 |          0.540994 |
|      1 |           48 |       0.473497 |      2.58963 |   0.498825 |                  0.74776 |  0.473497 |  0.476632 | 0.703031 |         0.80077  |          0.600636 |
|      2 |           26 |       0.494134 |      3.93468 |   0.544726 |                  0.74776 |  0.494134 |  0.581804 | 0.58071  |         0.733541 |          0.545898 |
|      3 |           56 |       0.437787 |      2.63262 |   0.46946  |                  0.74776 |  0.437787 |  0.561527 | 0.599621 |         0.731874 |          0.552412 |
|      4 |           37 |       0.496099 |      2.85275 |   0.527345 |                  0.74776 |  0.496099 |  0.493939 | 0.67028  |         0.7802   |          0.582286 |

## Decision Gate

Consider Caruana ADD only if MAE <= 0.48 or residual correlation is clearly low without major Spearman collapse.

## Final Read

This smaller Strategy 6 readout is the best run in the first sweep: OOF MAE about 0.484 and Spearman about 0.756. It narrowly misses the 0.48 single-model gate and is still much weaker than the current strongest pool members. Keep as weak-positive evidence for Strategy 6; do not submit directly.

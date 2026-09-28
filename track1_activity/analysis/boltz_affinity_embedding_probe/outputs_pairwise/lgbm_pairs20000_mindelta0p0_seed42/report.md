# Boltz Affinity Embedding Pairwise AS1 Probe

Pair models fit on sampled train_activity pairs; AS1 is held out and evaluated over all AS1 pairs.

## Main Readout

| feature              |   n_features |   n_pairs |   class_accuracy |   class_auc |   reg_sign_accuracy |   delta_mae |   delta_spearman |   val_class_auc |   val_delta_mae |
|:---------------------|-------------:|----------:|-----------------:|------------:|--------------------:|------------:|-----------------:|----------------:|----------------:|
| boltz_affinity_gmean |          384 |     21494 |           0.8376 |      0.9193 |              0.8309 |      1.0137 |           0.7463 |          0.8751 |          0.7307 |

## Threshold Sweep

| feature              |   n_features |   class_accuracy@0 |   class_accuracy@0.25 |   class_accuracy@0.5 |   class_accuracy@1 |   class_auc@0 |   class_auc@0.25 |   class_auc@0.5 |   class_auc@1 |   delta_mae@0 |   delta_mae@0.25 |   delta_mae@0.5 |   delta_mae@1 |   reg_sign_accuracy@0 |   reg_sign_accuracy@0.25 |   reg_sign_accuracy@0.5 |   reg_sign_accuracy@1 |
|:---------------------|-------------:|-------------------:|----------------------:|---------------------:|-------------------:|--------------:|-----------------:|----------------:|--------------:|--------------:|-----------------:|----------------:|--------------:|----------------------:|-------------------------:|------------------------:|----------------------:|
| boltz_affinity_gmean |          384 |             0.7519 |                0.7988 |               0.8376 |             0.9018 |        0.8372 |           0.8852 |          0.9193 |        0.9627 |        0.8256 |           0.9101 |          1.0137 |        1.2493 |                0.7470 |                   0.7929 |                  0.8309 |                0.8945 |

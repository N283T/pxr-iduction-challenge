# Boltz affinity scalar delta probe

## AS1 scalar correlation with pEC50

| split   | feature                       |   n |   pearson |   spearman |
|:--------|:------------------------------|----:|----------:|-----------:|
| as1     | affinity_probability_binary   | 253 |    0.5377 |     0.6012 |
| as1     | affinity_probability_binary_2 | 253 |    0.5083 |     0.5519 |
| as1     | affinity_probability_binary_1 | 253 |    0.4757 |     0.5363 |
| as1     | ensemble_diff_prob            | 253 |    0.1876 |     0.2608 |
| as1     | ensemble_diff_affinity        | 253 |    0.0499 |    -0.0116 |
| as1     | affinity_pred_value_2         | 253 |   -0.5006 |    -0.5249 |
| as1     | affinity_pred_value_1         | 253 |   -0.4975 |    -0.5760 |
| as1     | affinity_pred_value           | 253 |   -0.5523 |    -0.6235 |

## AS1 pairwise delta metrics, |delta pEC50| >= 0.5

| feature                       |   abs_delta_threshold |   n_pairs |    auc |   auc_best_orientation |   sign_accuracy |   sign_accuracy_best_orientation |   pearson |   spearman |   affine_slope_to_delta_pec50 |   raw_delta_mae |   affine_delta_mae |
|:------------------------------|----------------------:|----------:|-------:|-----------------------:|----------------:|---------------------------------:|----------:|-----------:|------------------------------:|----------------:|-------------------:|
| affinity_pred_value           |                0.5000 |     21491 | 0.1353 |                 0.8647 |          0.2110 |                           0.7890 |   -0.6174 |    -0.6372 |                       -1.3045 |          2.0667 |             1.0900 |
| affinity_probability_binary   |                0.5000 |     21491 | 0.8579 |                 0.8579 |          0.7767 |                           0.7767 |    0.6118 |     0.6272 |                        4.6720 |          1.3989 |             1.1023 |
| affinity_probability_binary_2 |                0.5000 |     21491 | 0.8341 |                 0.8341 |          0.7546 |                           0.7546 |    0.5898 |     0.6053 |                        5.2721 |          1.4276 |             1.1463 |
| affinity_pred_value_2         |                0.5000 |     21491 | 0.1781 |                 0.8219 |          0.2539 |                           0.7461 |   -0.5669 |    -0.5800 |                       -1.0498 |          2.0915 |             1.1436 |
| affinity_pred_value_1         |                0.5000 |     21491 | 0.1578 |                 0.8422 |          0.2412 |                           0.7588 |   -0.5643 |    -0.5795 |                       -1.1445 |          2.0639 |             1.1529 |
| affinity_probability_binary_1 |                0.5000 |     21491 | 0.8222 |                 0.8222 |          0.7478 |                           0.7478 |    0.5450 |     0.5527 |                        3.1520 |          1.3706 |             1.1672 |
| ensemble_diff_prob            |                0.5000 |     21491 | 0.6483 |                 0.6483 |          0.6098 |                           0.6098 |    0.2217 |     0.2271 |                        1.7395 |          1.4811 |             1.4432 |
| ensemble_diff_affinity        |                0.5000 |     21491 | 0.5103 |                 0.5103 |          0.5044 |                           0.5044 |    0.0669 |     0.0668 |                        0.1606 |          1.5696 |             1.5337 |

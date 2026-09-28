# Boltz Affinity Embedding AS1 Replay

Phase 1 style replay: models fit on original train labels only; AS1 is held out as test.

## Overall

| feature                     | tabpfn_version   |   n_features |   n |    mae |   bias_pred_minus_true |   spearman |   pearson |   pred_std |
|:----------------------------|:-----------------|-------------:|----:|-------:|-----------------------:|-----------:|----------:|-----------:|
| pooled_boltz_allpairs       | v3               |         1024 | 253 | 0.4997 |                 0.0071 |     0.7715 |    0.7268 |     0.6466 |
| pooled_boltz                | v3               |         1024 | 253 | 0.5218 |                -0.0013 |     0.7445 |    0.7162 |     0.6058 |
| boltz_affinity_g1g2_scalars | v3               |          774 | 253 | 0.5384 |                -0.1055 |     0.7421 |    0.7156 |     0.6758 |
| boltz_affinity_gmean        | v3               |          384 | 253 | 0.5461 |                -0.1202 |     0.7301 |    0.7011 |     0.6895 |
| boltz_affinity_g1g2         | v3               |          768 | 253 | 0.5531 |                -0.1138 |     0.7279 |    0.6946 |     0.6693 |

## Slices

| feature                     | tabpfn_version   |   n_features |    all |   true_3_4 |   true_4_5 |   true_5_6 |   true_gte6 |   true_lt3 |
|:----------------------------|:-----------------|-------------:|-------:|-----------:|-----------:|-----------:|------------:|-----------:|
| boltz_affinity_g1g2         | v3               |          768 | 0.5531 |     0.4960 |     0.3729 |     0.4738 |      1.0089 |     1.4201 |
| boltz_affinity_g1g2_scalars | v3               |          774 | 0.5384 |     0.5160 |     0.3551 |     0.4668 |      0.9699 |     1.3489 |
| boltz_affinity_gmean        | v3               |          384 | 0.5461 |     0.5221 |     0.3604 |     0.4719 |      1.0330 |     1.3553 |
| pooled_boltz                | v3               |         1024 | 0.5218 |     0.5449 |     0.3058 |     0.4146 |      0.8919 |     1.5669 |
| pooled_boltz_allpairs       | v3               |         1024 | 0.4997 |     0.5172 |     0.3091 |     0.3825 |      0.8075 |     1.5300 |

## Prediction Correlations

|                             |   boltz_affinity_g1g2 |   boltz_affinity_g1g2_scalars |   boltz_affinity_gmean |   pooled_boltz |   pooled_boltz_allpairs |
|:----------------------------|----------------------:|------------------------------:|-----------------------:|---------------:|------------------------:|
| boltz_affinity_g1g2         |                1.0000 |                        0.9937 |                 0.9889 |         0.9078 |                  0.9226 |
| boltz_affinity_g1g2_scalars |                0.9937 |                        1.0000 |                 0.9892 |         0.9125 |                  0.9268 |
| boltz_affinity_gmean        |                0.9889 |                        0.9892 |                 1.0000 |         0.9082 |                  0.9242 |
| pooled_boltz                |                0.9078 |                        0.9125 |                 0.9082 |         1.0000 |                  0.9673 |
| pooled_boltz_allpairs       |                0.9226 |                        0.9268 |                 0.9242 |         0.9673 |                  1.0000 |

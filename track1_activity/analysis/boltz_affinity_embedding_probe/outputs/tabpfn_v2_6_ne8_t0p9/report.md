# Boltz Affinity Embedding AS1 Replay

Phase 1 style replay: models fit on original train labels only; AS1 is held out as test.

## Overall

| feature                     | tabpfn_version   |   n_features |   n |    mae |   bias_pred_minus_true |   spearman |   pearson |   pred_std |
|:----------------------------|:-----------------|-------------:|----:|-------:|-----------------------:|-----------:|----------:|-----------:|
| pooled_boltz_allpairs       | v2_6             |         1024 | 253 | 0.4918 |                 0.0377 |     0.7735 |    0.7214 |     0.6797 |
| pooled_boltz                | v2_6             |         1024 | 253 | 0.4946 |                 0.0468 |     0.7641 |    0.7280 |     0.6665 |
| boltz_affinity_g1g2         | v2_6             |          768 | 253 | 0.5284 |                -0.0697 |     0.7499 |    0.7173 |     0.7027 |
| boltz_affinity_gmean        | v2_6             |          384 | 253 | 0.5289 |                -0.0694 |     0.7345 |    0.7095 |     0.6990 |
| boltz_affinity_g1g2_scalars | v2_6             |          774 | 253 | 0.5333 |                -0.0700 |     0.7486 |    0.7118 |     0.7025 |

## Slices

| feature                     | tabpfn_version   |   n_features |    all |   true_3_4 |   true_4_5 |   true_5_6 |   true_gte6 |   true_lt3 |
|:----------------------------|:-----------------|-------------:|-------:|-----------:|-----------:|-----------:|------------:|-----------:|
| boltz_affinity_g1g2         | v2_6             |          768 | 0.5284 |     0.5059 |     0.3766 |     0.4351 |      0.8709 |     1.3557 |
| boltz_affinity_g1g2_scalars | v2_6             |          774 | 0.5333 |     0.5373 |     0.3725 |     0.4392 |      0.8833 |     1.3579 |
| boltz_affinity_gmean        | v2_6             |          384 | 0.5289 |     0.5240 |     0.3568 |     0.4462 |      0.9223 |     1.3396 |
| pooled_boltz                | v2_6             |         1024 | 0.4946 |     0.5474 |     0.3315 |     0.3486 |      0.6945 |     1.5486 |
| pooled_boltz_allpairs       | v2_6             |         1024 | 0.4918 |     0.5238 |     0.3223 |     0.3522 |      0.7325 |     1.5510 |

## Prediction Correlations

|                             |   boltz_affinity_g1g2 |   boltz_affinity_g1g2_scalars |   boltz_affinity_gmean |   pooled_boltz |   pooled_boltz_allpairs |
|:----------------------------|----------------------:|------------------------------:|-----------------------:|---------------:|------------------------:|
| boltz_affinity_g1g2         |                1.0000 |                        0.9969 |                 0.9908 |         0.9163 |                  0.9304 |
| boltz_affinity_g1g2_scalars |                0.9969 |                        1.0000 |                 0.9917 |         0.9127 |                  0.9272 |
| boltz_affinity_gmean        |                0.9908 |                        0.9917 |                 1.0000 |         0.9088 |                  0.9236 |
| pooled_boltz                |                0.9163 |                        0.9127 |                 0.9088 |         1.0000 |                  0.9782 |
| pooled_boltz_allpairs       |                0.9304 |                        0.9272 |                 0.9236 |         0.9782 |                  1.0000 |

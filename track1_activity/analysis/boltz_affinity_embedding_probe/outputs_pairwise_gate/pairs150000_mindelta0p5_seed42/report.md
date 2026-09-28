# Pairwise Score Gate Scan

Pairwise win-score gates replayed against id55 AS1. AS1 labels are used only for gate scan readback.

| feature               | mode      |   quantile |   threshold |   shift |   as1_mae |   delta_mae_vs_anchor |   n_flags_as1 |   n_flags_as2 |   n_true_high_flags_as1 |   n_true_low_flags_as1 |
|:----------------------|:----------|-----------:|------------:|--------:|----------:|----------------------:|--------------:|--------------:|------------------------:|-----------------------:|
| pooled_boltz_allpairs | low_drop  |     0.2000 |      0.2654 | -0.1000 |    0.4056 |               -0.0010 |            57 |            46 |                       0 |                     19 |
| pooled_boltz_allpairs | low_drop  |     0.2000 |      0.2654 | -0.1500 |    0.4056 |               -0.0009 |            57 |            46 |                       0 |                     19 |
| pooled_boltz_allpairs | low_drop  |     0.2000 |      0.2654 | -0.0500 |    0.4058 |               -0.0008 |            57 |            46 |                       0 |                     19 |
| pooled_boltz_allpairs | low_drop  |     0.2500 |      0.3158 | -0.1000 |    0.4058 |               -0.0007 |            68 |            61 |                       0 |                     19 |
| pooled_boltz_allpairs | low_drop  |     0.2500 |      0.3158 | -0.0500 |    0.4058 |               -0.0007 |            68 |            61 |                       0 |                     19 |
| pooled_boltz_allpairs | low_drop  |     0.2000 |      0.2654 | -0.2000 |    0.4062 |               -0.0003 |            57 |            46 |                       0 |                     19 |
| pooled_boltz_allpairs | low_drop  |     0.2500 |      0.3158 | -0.1500 |    0.4064 |               -0.0002 |            68 |            61 |                       0 |                     19 |
| boltz_affinity_g1g2   | low_drop  |     0.2500 |      0.2993 | -0.0500 |    0.4067 |                0.0001 |            68 |            61 |                       0 |                     18 |
| boltz_affinity_g1g2   | high_lift |     0.9500 |      0.8664 |  0.0500 |    0.4068 |                0.0003 |            10 |            16 |                       3 |                      0 |
| pooled_boltz_allpairs | high_lift |     0.9500 |      0.8752 |  0.0500 |    0.4069 |                0.0003 |             8 |            18 |                       3 |                      0 |
| pooled_boltz_allpairs | low_drop  |     0.0500 |      0.1105 | -0.0500 |    0.4069 |                0.0003 |            11 |            15 |                       0 |                      5 |
| pooled_boltz_allpairs | low_drop  |     0.1500 |      0.2050 | -0.0500 |    0.4069 |                0.0004 |            42 |            35 |                       0 |                     15 |
| boltz_affinity_g1g2   | low_drop  |     0.2000 |      0.2573 | -0.0500 |    0.4071 |                0.0006 |            51 |            52 |                       0 |                     16 |
| boltz_affinity_g1g2   | high_lift |     0.9500 |      0.8664 |  0.1000 |    0.4072 |                0.0007 |            10 |            16 |                       3 |                      0 |
| boltz_affinity_g1g2   | high_lift |     0.8500 |      0.7803 |  0.0500 |    0.4072 |                0.0007 |            30 |            47 |                       6 |                      0 |
| pooled_boltz_allpairs | high_lift |     0.9500 |      0.8752 |  0.1000 |    0.4073 |                0.0007 |             8 |            18 |                       3 |                      0 |
| pooled_boltz_allpairs | high_lift |     0.8000 |      0.7214 |  0.0500 |    0.4073 |                0.0008 |            40 |            63 |                       9 |                      0 |
| pooled_boltz_allpairs | high_lift |     0.9000 |      0.8278 |  0.0500 |    0.4074 |                0.0008 |            17 |            35 |                       3 |                      0 |
| pooled_boltz_allpairs | low_drop  |     0.0500 |      0.1105 | -0.1000 |    0.4075 |                0.0009 |            11 |            15 |                       0 |                      5 |
| boltz_affinity_g1g2   | low_drop  |     0.0500 |      0.1083 | -0.0500 |    0.4075 |                0.0009 |            16 |            10 |                       0 |                      6 |

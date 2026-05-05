# Hybrid Delta Candidates

Correction delta from raw potent/no-aux candidates added to `ens_hybrid_meta_baseline_5050.csv`. OOF uses reconstructed hybrid proxy.

| candidate                                |   alpha_potent |   alpha_noaux |   hybrid_base_mae |     mae |    d_mae |   hybrid_base_sp |      sp |    d_sp |   test_mean_abs_delta_vs_hybrid |   test_p90_abs_delta_vs_hybrid |   test_max_abs_delta_vs_hybrid |
|:-----------------------------------------|---------------:|--------------:|------------------:|--------:|---------:|-----------------:|--------:|--------:|--------------------------------:|-------------------------------:|-------------------------------:|
| ens_hybrid_plus_potent_a50_noaux_a50.csv |        0.50000 |       0.50000 |           0.40195 | 0.39468 | -0.00727 |          0.84122 | 0.84724 | 0.00602 |                         0.08207 |                        0.12200 |                        0.25793 |
| ens_hybrid_plus_potent_a30_noaux_a30.csv |        0.30000 |       0.30000 |           0.40195 | 0.39706 | -0.00489 |          0.84122 | 0.84538 | 0.00416 |                         0.04924 |                        0.07320 |                        0.15476 |

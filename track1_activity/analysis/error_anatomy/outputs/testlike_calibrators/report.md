# Test-Like Calibrator Probe

| candidate                                  | fit_mask         |     mae |   spearman |   mean_pred |   mean_residual |   n_fit_total |   delta_mae_vs_raw |   delta_spearman_vs_raw |
|:-------------------------------------------|:-----------------|--------:|-----------:|------------:|----------------:|--------------:|-------------------:|------------------------:|
| affine_fit_all_train_eval_all              | all_train        | 0.39619 |    0.84459 |     4.32063 |         0.00018 |    4140.00000 |           -0.00090 |                -0.00050 |
| affine_fit_all_train_eval_same_mask        | all_train        | 0.39619 |    0.84459 |     4.32063 |         0.00018 |    4140.00000 |           -0.00090 |                -0.00050 |
| raw                                        | none             | 0.39709 |    0.84509 |     4.33423 |        -0.01341 |     nan       |          nan       |               nan       |
| affine_fit_family_gap_top10_eval_all       | family_gap_top10 | 0.40993 |    0.84362 |     4.38373 |        -0.06291 |     414.00000 |            0.01284 |                -0.00147 |
| affine_fit_no_counter_eval_all             | no_counter       | 0.41172 |    0.84444 |     4.23861 |         0.08220 |    1492.00000 |            0.01463 |                -0.00065 |
| affine_fit_no_single_hi_eval_all           | no_single_hi     | 0.41193 |    0.84450 |     4.23425 |         0.08656 |    1766.00000 |            0.01484 |                -0.00059 |
| affine_fit_no_single_lo_eval_all           | no_single_lo     | 0.41320 |    0.84463 |     4.22580 |         0.09501 |    1819.00000 |            0.01611 |                -0.00046 |
| affine_fit_member_std_top10_eval_all       | member_std_top10 | 0.41944 |    0.84382 |     4.37519 |        -0.05438 |     414.00000 |            0.02235 |                -0.00127 |
| affine_fit_no_aux_all_eval_all             | no_aux_all       | 0.43919 |    0.84410 |     4.17446 |         0.14636 |    1294.00000 |            0.04210 |                -0.00099 |
| affine_fit_no_counter_eval_same_mask       | no_counter       | 0.44497 |    0.84197 |     3.58185 |        -0.00135 |    1492.00000 |            0.00092 |                -0.00223 |
| affine_fit_no_aux_all_eval_same_mask       | no_aux_all       | 0.45482 |    0.78977 |     3.37886 |        -0.00206 |    1294.00000 |           -0.00248 |                -0.00364 |
| affine_fit_no_single_hi_eval_same_mask     | no_single_hi     | 0.48515 |    0.81711 |     3.63554 |         0.00090 |    1766.00000 |            0.00215 |                -0.00275 |
| affine_fit_no_single_lo_eval_same_mask     | no_single_lo     | 0.49830 |    0.80829 |     3.61663 |         0.00113 |    1819.00000 |            0.00183 |                -0.00266 |
| affine_fit_family_gap_top10_eval_same_mask | family_gap_top10 | 0.66898 |    0.73567 |     3.92487 |         0.00203 |     414.00000 |            0.00097 |                -0.00546 |
| affine_fit_member_std_top10_eval_same_mask | member_std_top10 | 0.67122 |    0.70358 |     3.84046 |         0.00348 |     414.00000 |           -0.00262 |                -0.00368 |

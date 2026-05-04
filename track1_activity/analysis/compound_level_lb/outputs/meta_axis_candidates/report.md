# Meta-Axis Candidate Sweep

Known LB anchors along `prediction = (1-alpha) * baseline9 + alpha * family_meta`:

|    alpha |   lb_mae |    lb_sp | label            |
|---------:|---------:|---------:|:-----------------|
| 0.000000 | 0.407847 | 0.845446 | id32_baseline9   |
| 0.500000 | 0.407484 | 0.847050 | id43_hybrid      |
| 1.000000 | 0.409074 | 0.847572 | id42_family_meta |

Quadratic MAE fit: `0.00390765 * a^2 + -0.00268030 * a + 0.40784711`
Fitted MAE optimum alpha: `0.343`

Candidate CSVs were written but should be treated as low-upside A/B candidates.
The fitted improvement over id43 is about 0.0001 MAE, below normal LB noise.

|    alpha |   predicted_lb_mae_quadratic |   predicted_lb_sp_quadratic |   mean_pred |   std_pred |   mean_abs_shift_vs_base |   p90_abs_shift_vs_base | path                                                                                           |
|---------:|-----------------------------:|----------------------------:|------------:|-----------:|-------------------------:|------------------------:|:-----------------------------------------------------------------------------------------------|
| 0.250000 |                     0.407421 |                    0.846383 |    4.794603 |   0.763157 |                 0.010145 |                0.022415 | track1_activity/analysis/compound_level_lb/outputs/meta_axis_candidates/ens_meta_axis_a250.csv |
| 0.300000 |                     0.407395 |                    0.846538 |    4.795006 |   0.762486 |                 0.012174 |                0.026898 | track1_activity/analysis/compound_level_lb/outputs/meta_axis_candidates/ens_meta_axis_a300.csv |
| 0.343000 |                     0.407387 |                    0.846663 |    4.795353 |   0.761915 |                 0.013919 |                0.030753 | track1_activity/analysis/compound_level_lb/outputs/meta_axis_candidates/ens_meta_axis_a343.csv |
| 0.350000 |                     0.407388 |                    0.846682 |    4.795409 |   0.761823 |                 0.014204 |                0.031380 | track1_activity/analysis/compound_level_lb/outputs/meta_axis_candidates/ens_meta_axis_a350.csv |
| 0.400000 |                     0.407400 |                    0.846815 |    4.795812 |   0.761168 |                 0.016233 |                0.035863 | track1_activity/analysis/compound_level_lb/outputs/meta_axis_candidates/ens_meta_axis_a400.csv |

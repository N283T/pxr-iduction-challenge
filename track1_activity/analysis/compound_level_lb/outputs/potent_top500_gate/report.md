# Potent46-Gated Top500 CSV Candidates

Candidate formula:

```text
candidate = id51 + gamma * gate(nn_potent46_tanimoto) * (top500 - ens_caruana_bag20)
```

This is a CSV-only probe. It does not retrain or change the ensemble pool.

## Candidate Summary

| candidate                                 |   threshold | gate   |    gamma |   n_gate_nonzero |   n_gate_full |   mean_gate |   mean_shift |   mean_abs_shift |   p90_abs_shift |   max_abs_shift |   corr_vs_anchor | path                                                                  |
|:------------------------------------------|------------:|:-------|---------:|-----------------:|--------------:|------------:|-------------:|-----------------:|----------------:|----------------:|-----------------:|:----------------------------------------------------------------------|
| ens_id51_top500_potent46_t30_hard_g15.csv |    0.300000 | hard   | 0.150000 |              300 |           300 |    0.584795 |     0.000268 |         0.008310 |        0.023393 |        0.068170 |         0.999853 | track1_activity/submissions/ens_id51_top500_potent46_t30_hard_g15.csv |
| ens_id51_top500_potent46_t30_hard_g25.csv |    0.300000 | hard   | 0.250000 |              300 |           300 |    0.584795 |     0.000446 |         0.013850 |        0.038989 |        0.113616 |         0.999594 | track1_activity/submissions/ens_id51_top500_potent46_t30_hard_g25.csv |
| ens_id51_top500_potent46_t30_hard_g35.csv |    0.300000 | hard   | 0.350000 |              300 |           300 |    0.584795 |     0.000625 |         0.019390 |        0.054584 |        0.159063 |         0.999210 | track1_activity/submissions/ens_id51_top500_potent46_t30_hard_g35.csv |
| ens_id51_top500_potent46_t30_hard_g50.csv |    0.300000 | hard   | 0.500000 |              300 |           300 |    0.584795 |     0.000893 |         0.027700 |        0.077978 |        0.227232 |         0.998409 | track1_activity/submissions/ens_id51_top500_potent46_t30_hard_g50.csv |
| ens_id51_top500_potent46_t30_soft_g15.csv |    0.300000 | soft   | 0.150000 |              300 |           244 |    0.542900 |     0.000256 |         0.007797 |        0.022866 |        0.068170 |         0.999860 | track1_activity/submissions/ens_id51_top500_potent46_t30_soft_g15.csv |
| ens_id51_top500_potent46_t30_soft_g25.csv |    0.300000 | soft   | 0.250000 |              300 |           244 |    0.542900 |     0.000427 |         0.012996 |        0.038110 |        0.113616 |         0.999613 | track1_activity/submissions/ens_id51_top500_potent46_t30_soft_g25.csv |
| ens_id51_top500_potent46_t30_soft_g35.csv |    0.300000 | soft   | 0.350000 |              300 |           244 |    0.542900 |     0.000597 |         0.018194 |        0.053354 |        0.159063 |         0.999247 | track1_activity/submissions/ens_id51_top500_potent46_t30_soft_g35.csv |
| ens_id51_top500_potent46_t30_soft_g50.csv |    0.300000 | soft   | 0.500000 |              300 |           244 |    0.542900 |     0.000853 |         0.025991 |        0.076220 |        0.227232 |         0.998482 | track1_activity/submissions/ens_id51_top500_potent46_t30_soft_g50.csv |
| ens_id51_top500_potent46_t35_hard_g15.csv |    0.350000 | hard   | 0.150000 |              284 |           284 |    0.553606 |     0.000185 |         0.007927 |        0.023140 |        0.068170 |         0.999856 | track1_activity/submissions/ens_id51_top500_potent46_t35_hard_g15.csv |
| ens_id51_top500_potent46_t35_hard_g25.csv |    0.350000 | hard   | 0.250000 |              284 |           284 |    0.553606 |     0.000309 |         0.013211 |        0.038567 |        0.113616 |         0.999603 | track1_activity/submissions/ens_id51_top500_potent46_t35_hard_g25.csv |
| ens_id51_top500_potent46_t35_hard_g35.csv |    0.350000 | hard   | 0.350000 |              284 |           284 |    0.553606 |     0.000433 |         0.018496 |        0.053994 |        0.159063 |         0.999228 | track1_activity/submissions/ens_id51_top500_potent46_t35_hard_g35.csv |
| ens_id51_top500_potent46_t35_hard_g50.csv |    0.350000 | hard   | 0.500000 |              284 |           284 |    0.553606 |     0.000618 |         0.026423 |        0.077134 |        0.227232 |         0.998444 | track1_activity/submissions/ens_id51_top500_potent46_t35_hard_g50.csv |
| ens_id51_top500_potent46_t35_soft_g15.csv |    0.350000 | soft   | 0.150000 |              284 |           176 |    0.491618 |     0.000369 |         0.007094 |        0.021625 |        0.068170 |         0.999879 | track1_activity/submissions/ens_id51_top500_potent46_t35_soft_g15.csv |
| ens_id51_top500_potent46_t35_soft_g25.csv |    0.350000 | soft   | 0.250000 |              284 |           176 |    0.491618 |     0.000616 |         0.011823 |        0.036042 |        0.113616 |         0.999665 | track1_activity/submissions/ens_id51_top500_potent46_t35_soft_g25.csv |
| ens_id51_top500_potent46_t35_soft_g35.csv |    0.350000 | soft   | 0.350000 |              284 |           176 |    0.491618 |     0.000862 |         0.016552 |        0.050459 |        0.159063 |         0.999348 | track1_activity/submissions/ens_id51_top500_potent46_t35_soft_g35.csv |
| ens_id51_top500_potent46_t35_soft_g50.csv |    0.350000 | soft   | 0.500000 |              284 |           176 |    0.491618 |     0.001231 |         0.023646 |        0.072084 |        0.227232 |         0.998683 | track1_activity/submissions/ens_id51_top500_potent46_t35_soft_g50.csv |
| ens_id51_top500_potent46_t40_hard_g15.csv |    0.400000 | hard   | 0.150000 |              278 |           278 |    0.541910 |     0.000258 |         0.007808 |        0.023086 |        0.068170 |         0.999857 | track1_activity/submissions/ens_id51_top500_potent46_t40_hard_g15.csv |
| ens_id51_top500_potent46_t40_hard_g25.csv |    0.400000 | hard   | 0.250000 |              278 |           278 |    0.541910 |     0.000430 |         0.013013 |        0.038476 |        0.113616 |         0.999605 | track1_activity/submissions/ens_id51_top500_potent46_t40_hard_g25.csv |
| ens_id51_top500_potent46_t40_hard_g35.csv |    0.400000 | hard   | 0.350000 |              278 |           278 |    0.541910 |     0.000603 |         0.018219 |        0.053867 |        0.159063 |         0.999232 | track1_activity/submissions/ens_id51_top500_potent46_t40_hard_g35.csv |
| ens_id51_top500_potent46_t40_hard_g50.csv |    0.400000 | hard   | 0.500000 |              278 |           278 |    0.541910 |     0.000861 |         0.026027 |        0.076952 |        0.227232 |         0.998450 | track1_activity/submissions/ens_id51_top500_potent46_t40_hard_g50.csv |
| ens_id51_top500_potent46_t40_soft_g15.csv |    0.400000 | soft   | 0.150000 |              278 |            81 |    0.388964 |     0.000460 |         0.005642 |        0.016707 |        0.067850 |         0.999911 | track1_activity/submissions/ens_id51_top500_potent46_t40_soft_g15.csv |
| ens_id51_top500_potent46_t40_soft_g25.csv |    0.400000 | soft   | 0.250000 |              278 |            81 |    0.388964 |     0.000766 |         0.009403 |        0.027844 |        0.113083 |         0.999755 | track1_activity/submissions/ens_id51_top500_potent46_t40_soft_g25.csv |
| ens_id51_top500_potent46_t40_soft_g35.csv |    0.400000 | soft   | 0.350000 |              278 |            81 |    0.388964 |     0.001073 |         0.013165 |        0.038982 |        0.158316 |         0.999523 | track1_activity/submissions/ens_id51_top500_potent46_t40_soft_g35.csv |
| ens_id51_top500_potent46_t40_soft_g50.csv |    0.400000 | soft   | 0.500000 |              278 |            81 |    0.388964 |     0.001533 |         0.018806 |        0.055688 |        0.226165 |         0.999035 | track1_activity/submissions/ens_id51_top500_potent46_t40_soft_g50.csv |

## Recommended first probe

Submitted `ens_id51_top500_potent46_t40_soft_g35.csv` as id=55.
It improved id51 by 0.000246 MAE on LB, with a small Spearman drop.

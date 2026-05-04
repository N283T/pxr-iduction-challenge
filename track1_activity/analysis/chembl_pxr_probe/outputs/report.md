# ChEMBL PXR Activation Probe

Filtered ChEMBL activation EC50 molecules: **267**

## Coverage

| split   |    n |   exact |   nn_ge_0.3 |   nn_ge_0.4 |   nn_max |   nn_median |
|:--------|-----:|--------:|------------:|------------:|---------:|------------:|
| train   | 4140 |      12 |         184 |          40 |   1.0000 |      0.2192 |
| test    |  513 |       0 |          27 |           0 |   0.3939 |      0.2254 |

## OOF Models

| model                     |    mae |      r2 |   spearman |   delta_mae_vs_reference |   residual_r_vs_reference |
|:--------------------------|-------:|--------:|-----------:|-------------------------:|--------------------------:|
| chembl_pxr_lgbm           | 0.8413 | -0.0101 |     0.1612 |                 nan      |                  nan      |
| chembl_pxr_ridge          | 0.8937 |  0.0198 |     0.1373 |                 nan      |                  nan      |
| reference_ensemble_oof    | 0.3910 |  0.7635 |     0.8502 |                 nan      |                  nan      |
| reference_plus_0.01_lgbm  | 0.3911 |  0.7635 |     0.8502 |                   0.0001 |                    0.4933 |
| reference_plus_0.02_lgbm  | 0.3914 |  0.7632 |     0.8502 |                   0.0004 |                    0.4933 |
| reference_plus_0.05_lgbm  | 0.3931 |  0.7616 |     0.8501 |                   0.0021 |                    0.4933 |
| reference_plus_0.10_lgbm  | 0.3981 |  0.7558 |     0.8491 |                   0.0071 |                    0.4933 |
| reference_plus_0.20_lgbm  | 0.4184 |  0.7327 |     0.8445 |                   0.0274 |                    0.4933 |
| reference_plus_0.01_ridge | 0.3914 |  0.7635 |     0.8502 |                   0.0004 |                    0.4913 |
| reference_plus_0.02_ridge | 0.3918 |  0.7632 |     0.8501 |                   0.0008 |                    0.4913 |
| reference_plus_0.05_ridge | 0.3941 |  0.7617 |     0.8499 |                   0.0031 |                    0.4913 |
| reference_plus_0.10_ridge | 0.4003 |  0.7561 |     0.8494 |                   0.0093 |                    0.4913 |
| reference_plus_0.20_ridge | 0.4228 |  0.7338 |     0.8470 |                   0.0318 |                    0.4913 |

## Interpretation

This is a cheap external-data feature probe only. No submission CSV is produced.
The primary pass criterion is a strong OOF gain after blending with the current
ensemble reference; weak standalone or positive delta MAE should be closed.

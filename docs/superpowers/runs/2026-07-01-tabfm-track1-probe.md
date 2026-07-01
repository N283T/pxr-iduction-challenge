# TabFM Track 1 Phase-1 Replay Probe

Date: 2026-07-01

## Summary

Google's TabFM v1.0.0 PyTorch release was tested as a zero-shot / in-context
tabular regressor for Track 1 Activity. The evaluation used a Phase-1-style
replay:

- Fit context: `train_activity` only.
- Prediction target: all 513 `test_activity` compounds.
- Score: the 253 released AS1 labels from `test_activity_phase1_labels`.
- No AS1 labels were used for fitting in these replay runs.

The best observed TabFM setting was:

| Model | AS1 MAE | AS1 Spearman | Bias |
| --- | ---: | ---: | ---: |
| TabFM top300, `n_estimators=4`, `norm=none`, no feature shuffle | 0.4161 | 0.8299 | +0.0096 |

This is credible but did not beat the stronger TabPFN reference runs:

| Reference | AS1 MAE | AS1 Spearman | Notes |
| --- | ---: | ---: | --- |
| `ens_id51_top500_potent46_t40_soft_g35` | 0.4066 | 0.8488 | sanity-check anchor |
| `tabpfn_..._seed10ens_top500_umap_v3_temp0p7` | 0.4107 | 0.8357 | best comparable non-AS1 TabPFN found |
| `tabpfn_..._seed10ens_top500_umap` | 0.4214 | 0.8334 | older top500 TabPFN |
| TabFM best from this probe | 0.4161 | 0.8299 | top300, ne4, no shuffle |

Main takeaways:

- TabFM is usable on this PXR feature matrix, but it is slower and weaker than
  the best TabPFN v3 top500 replay.
- More features did not monotonically help. The useful region was roughly
  top100-top300 by LGBM gain, with top300 best in this sweep.
- TabFM inference ensembling helped a little up to `n_estimators=4`, but
  `n_estimators=8` did not improve MAE.
- The full 2103-column feature matrix failed on the RTX 5080 16 GB VRAM run
  with CUDA OOM at `max_num_rows=4096`.
- The public TabFM ensemble appears to be an inference-time preset around the
  same released checkpoint, not a separate downloadable ensemble checkpoint.

## Setup

TabFM was not vendored into this repository. The local probe used a clone at
`/tmp/tabfm` and a minimal no-dependency install to avoid perturbing the pixi
CUDA/PyTorch stack:

```bash
pixi run python -m pip install --no-deps -e /tmp/tabfm \
  absl-py 'jaxtyping<0.3' 'typeguard<3'
```

The model loaded `google/tabfm-1.0.0-pytorch` regression weights from Hugging
Face. Note that the HF weights are under `tabfm-non-commercial-v1.0`; treat
these results as research probes unless the license is appropriate.

Probe scripts added:

- `track1_activity/scripts/run_tabfm_probe.py`
- `track1_activity/scripts/run_tabfm_topk_sweep.py`

The first AS1 replay implementation briefly had an indexing bug: it assigned
`row_number()` after joining AS1 labels, which produced AS1-local indices
instead of 513-test-set indices. This was fixed by assigning `test_idx` in a
subquery before joining `test_activity_phase1_labels`; anchor checks then
reproduced the known AS1 MAE scale.

## Feature Count Sweep

Base feature set:
`cheme_2d_full_boltz_log2fc_pred_seed10ens` (2103 columns before top-k).

Feature ranking:
single full-train LightGBM gain ranking, then top-k columns passed to TabFM.

Fixed TabFM settings for this table:
`n_estimators=1`, `norm_methods=none`, `feat_shuffle_method=none`,
`max_num_rows=4096`, `device=cuda`.

| top-k | AS1 MAE | AS1 RAE | AS1 Spearman | Bias |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 0.4238 | 0.5307 | 0.8200 | +0.0207 |
| 100 | 0.4222 | 0.5286 | 0.8198 | +0.0185 |
| 128 | 0.4211 | 0.5272 | 0.8223 | +0.0170 |
| 150 | 0.4283 | 0.5363 | 0.8251 | +0.0257 |
| 200 | 0.4248 | 0.5319 | 0.8262 | +0.0170 |
| 256 | 0.4212 | 0.5273 | 0.8288 | +0.0156 |
| 300 | 0.4186 | 0.5241 | 0.8283 | +0.0137 |
| 500 | 0.4248 | 0.5319 | 0.8273 | -0.0128 |
| 2103/full | OOM | OOM | OOM | OOM |

Full-width failure:

```text
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 25.89 GiB.
GPU 0 has a total capacity of 15.92 GiB.
```

The OOM was observed for full 2103 columns at `max_num_rows=4096`.

## Top300 Ensemble Sweep

Top300 was selected for deeper probing because it was the best single-estimator
feature count.

| top-k | n_estimators | norm_methods | feature shuffle | AS1 MAE | AS1 RAE | AS1 Spearman | Bias | Runtime |
| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 300 | 1 | none | none | 0.4186 | 0.5241 | 0.8283 | +0.0137 | 5.2m |
| 300 | 4 | none | none | 0.4161 | 0.5210 | 0.8299 | +0.0096 | 20.7m |
| 300 | 8 | none | none | 0.4164 | 0.5214 | 0.8290 | +0.0091 | 42.2m |
| 300 | 4 | none | random | 0.4177 | 0.5230 | 0.8334 | -0.0024 | 21.1m |
| 300 | 8 | none | random | 0.4185 | 0.5241 | 0.8320 | -0.0019 | 40.7m |
| 300 | 4 | none,power | random | 0.4196 | 0.5254 | 0.8322 | +0.0010 | 20.4m |

Interpretation:

- `n_estimators=4` was the useful point in this setting.
- `n_estimators=8` increased runtime without improving AS1 MAE.
- Random feature shuffling improved Spearman but not MAE.
- Adding `power` normalization hurt MAE relative to `norm=none`.

## Commands

Representative commands:

```bash
pixi run python track1_activity/scripts/run_tabfm_topk_sweep.py \
  --tabfm-repo /tmp/tabfm \
  --top-k-list 100,150,200,300 \
  --resume \
  --n-estimators 1 \
  --norm-methods none \
  --feat-shuffle-method none \
  --max-num-rows 4096

pixi run python track1_activity/scripts/run_tabfm_topk_sweep.py \
  --tabfm-repo /tmp/tabfm \
  --top-k-list 300 \
  --resume \
  --n-estimators 4 \
  --norm-methods none \
  --feat-shuffle-method none \
  --max-num-rows 4096
```

Generated run artifacts live under the ignored directory:

```text
track1_activity/analysis/tabfm_topk_sweep/outputs/
```

## Decision

For this challenge setting, TabFM is interesting but not currently compelling
as a production ensemble member:

- It trails the best comparable TabPFN v3 AS1 replay by about 0.0054 MAE.
- It is substantially slower at useful ensemble settings.
- The best setting found here still trails the id51/id55 anchor by about
  0.0095 MAE on AS1.

Unless a new orthogonality or calibration use case appears, TabPFN remains the
better cost/performance option for this repository.

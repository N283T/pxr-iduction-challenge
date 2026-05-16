# PocketXMol PXR Probe

Date: 2026-05-15

## Goal

Probe whether PocketXMol, a pocket-conditioned generative/docking model, offers
a useful signal beyond ligand-only encoders and existing Boltz-derived features.

This is not directly comparable to ChemProp: ChemProp is ligand-only 2D message
passing, while PocketXMol conditions a 3D ligand graph on a protein pocket during
denoising. The closest existing local feature family is Boltz pose/trunk features.

## Setup

PocketXMol was cloned outside this repository:

```bash
ghq get https://github.com/pengxingang/PocketXMol.git
```

Weights were downloaded from the upstream Zenodo record into the PocketXMol
checkout. The archive size was about 610 MB. A separate Python 3.10 venv was used
to avoid changing the pixi environment.

The PXR inputs intentionally avoid `structures/gator/`. Each compound uses:

- protein: `structures/boltz2/outputs/posebusters_tmp_proteins/<compound>.pdb`
- ligand coordinates: `structures/boltz2/ligands/<compound>.sdf`
- pocket extraction: `ref_ligand_path` with radius 10 A

Some Boltz SDFs cannot be read by PocketXMol's default sanitized RDKit loader.
The wrapper writes temporary SDFs with DB standard-SMILES bond orders and Boltz
coordinates, kekulized for PocketXMol compatibility.

Generated configs, SDF outputs, logs, and summaries are ignored under:

```text
structures/pocketxmol_probe/
```

## Script

Added:

```bash
pixi run python track1_activity/scripts/run_pocketxmol_probe.py --help
```

The script selects compounds, generates one PocketXMol config per compound, runs
the upstream `scripts/sample_use.py`, and collects `cfd_traj`, `cfd_pos`,
`cfd_node`, and `cfd_edge` from `gen_info.csv`.

Hidden-state extraction is intentionally split into two stages:

```bash
pixi run python track1_activity/scripts/run_pocketxmol_embedding_probe.py --help
pixi run python track1_activity/scripts/pool_pocketxmol_hidden.py --help
```

The extraction stage reuses the same PocketXMol preprocessing and prepared SDF
path, then runs one forward pass per requested timestep and saves variable-length
raw hidden arrays per compound:

- `<timestep>_pocket`: pocket encoder hidden states
- `<timestep>_node`: ligand atom hidden states after the denoiser
- `<timestep>_edge`: ligand edge hidden states after the denoiser

The pooling stage is separate, so mean/std/max/min/quantile/delta variants can be
tested without rerunning PocketXMol.

## Runs

Official example smoke:

```bash
/home/nagaet/.cache/codex/pocketxmol/.venv/bin/python scripts/sample_use.py \
  --config_task /home/nagaet/.cache/codex/pocketxmol/dock_smallmol_smoke.yml \
  --config_model configs/sample/pxm.yml \
  --outdir /home/nagaet/.cache/codex/pocketxmol/outputs_official_smoke \
  --device cuda:0 --batch_size 1 --num_workers 0
```

PXR smoke:

```bash
/home/nagaet/.cache/codex/pocketxmol/.venv/bin/python scripts/sample_use.py \
  --config_task /home/nagaet/.cache/codex/pocketxmol/pxr_00018_smoke.yml \
  --config_model configs/sample/pxm.yml \
  --outdir /home/nagaet/.cache/codex/pocketxmol/outputs_pxr_00018_smoke \
  --device cuda:0 --batch_size 1 --num_workers 0
```

Main scalar-confidence probe:

```bash
pixi run python track1_activity/scripts/run_pocketxmol_probe.py \
  --run-name pxm_radius10_steps100_n200_kek \
  --limit 200 \
  --num-steps 100 \
  --num-mols 1 \
  --batch-size 1 \
  --device cuda:0
```

Full scalar-confidence pass:

```bash
pixi run python track1_activity/scripts/run_pocketxmol_probe.py \
  --run-name pxm_radius10_steps100_all_kek \
  --all-compounds \
  --num-steps 100 \
  --num-mols 1 \
  --batch-size 1 \
  --device cuda:0
```

Trajectory-output mini probe:

```bash
pixi run python track1_activity/scripts/run_pocketxmol_probe.py \
  --run-name pxm_radius10_steps100_n40_tensors \
  --limit 40 \
  --num-steps 100 \
  --num-mols 1 \
  --batch-size 1 \
  --device cuda:0 \
  --save-output-tensors
```

Hidden-state raw smoke:

```bash
pixi run python track1_activity/scripts/run_pocketxmol_embedding_probe.py \
  --run-name pxm_hidden_raw_smoke3 \
  --compound-ids 18 35 55 \
  --timesteps 1.0 0.5 0.05 \
  --device cuda:0
```

Pooling examples:

```bash
pixi run python track1_activity/scripts/pool_pocketxmol_hidden.py \
  --raw-manifest structures/pocketxmol_probe/pxm_hidden_raw_smoke3/raw_hidden/raw_hidden_manifest.csv \
  --out-npz structures/pocketxmol_probe/pxm_hidden_raw_smoke3/pooled_mean_std.npz \
  --out-features structures/pocketxmol_probe/pxm_hidden_raw_smoke3/pooled_mean_std_features.txt \
  --stats mean std

pixi run python track1_activity/scripts/pool_pocketxmol_hidden.py \
  --raw-manifest structures/pocketxmol_probe/pxm_hidden_raw_smoke3/raw_hidden/raw_hidden_manifest.csv \
  --out-npz structures/pocketxmol_probe/pxm_hidden_raw_smoke3/pooled_rich_delta.npz \
  --out-features structures/pocketxmol_probe/pxm_hidden_raw_smoke3/pooled_rich_delta_features.txt \
  --stats mean std max min q25 q50 q75 \
  --include-deltas
```

## Results

Runtime:

- 200 compounds completed successfully.
- Mean elapsed time was 8.095 s per compound.
- Total wall-clock for the 200-compound run was 26.98 min.
- Full scalar pass completed 4652 / 4652 compounds successfully.
- Full scalar pass mean elapsed time was 8.542 s per compound, 11.04 GPU-hours
  total before analysis.

200-compound scalar probe composition:

- 50 lowest-activity train compounds
- 50 highest-activity train compounds
- 50 deterministic-random train compounds
- 50 deterministic-random test compounds

Scalar confidence correlations versus pEC50 on the 150 train rows from the
200-compound pilot:

| feature | Pearson | Spearman |
|---|---:|---:|
| `cfd_traj` | -0.0732 | -0.0465 |
| `cfd_pos` | 0.0611 | -0.0077 |
| `cfd_node` | 0.1657 | 0.1323 |
| `cfd_edge` | 0.2367 | 0.1476 |

Correlations on the 50 random train rows:

| feature | Pearson | Spearman |
|---|---:|---:|
| `cfd_traj` | -0.1860 | -0.2424 |
| `cfd_pos` | -0.0443 | -0.0357 |
| `cfd_node` | -0.0087 | -0.1073 |
| `cfd_edge` | 0.2623 | 0.2617 |

Five-fold CV using only the four scalar confidence features was weak:

| subset | model | MAE |
|---|---|---:|
| all 150 train rows | Ridge | 1.8020 |
| all 150 train rows | HistGradientBoosting | 1.6452 |
| all 150 train rows | RandomForest | 1.6971 |
| 50 random train rows | Ridge | 0.6866 |
| 50 random train rows | HistGradientBoosting | 0.7120 |
| 50 random train rows | RandomForest | 0.7672 |

Residual diagnostic against experiment `2451`
(`tabpfn_cheme_2d_full_boltz_log2fc_pred_seed10ens_top500_umap_v3`) was also
weak on the 150 sampled train rows:

| feature | abs-error Pearson | abs-error Spearman |
|---|---:|---:|
| `cfd_traj` | -0.1300 | -0.1334 |
| `cfd_pos` | -0.0741 | -0.1346 |
| `cfd_node` | 0.1017 | 0.0561 |
| `cfd_edge` | 0.0258 | 0.0040 |

The 40-compound trajectory-output probe did not rescue the signal. Spearman
correlations versus pEC50 on 30 train rows were small and unstable; the strongest
trajectory-statistic correlation was `cfd_pos`/`traj_last` around -0.27.

Full-pass scalar confidence correlations versus pEC50 on 4139 train rows:

| feature | Pearson | Spearman |
|---|---:|---:|
| `cfd_traj` | -0.10777 | -0.07085 |
| `cfd_pos` | 0.09951 | 0.07156 |
| `cfd_node` | 0.24852 | 0.17825 |
| `cfd_edge` | 0.36346 | 0.23845 |

Five-fold CV using only the four full-pass scalar confidence features:

| model | MAE | prediction std | Spearman(pred, y) |
|---|---:|---:|---:|
| Ridge | 0.82121 | 0.42542 | 0.26252 |
| HistGradientBoosting | 0.78135 | 0.50350 | 0.27658 |
| RandomForest | 0.77948 | 0.49890 | 0.28161 |
| ExtraTrees | 0.80003 | 0.38180 | 0.27178 |

Residual diagnostics stayed near zero against strong existing OOF models. For
experiment `2451`
(`tabpfn_cheme_2d_full_boltz_log2fc_pred_seed10ens_top500_umap_v3`), OOF MAE on
the 4139 rows was 0.39715 and the strongest absolute-error correlation among
PocketXMol scalar features was only Spearman -0.0915 (`cfd_edge`). Similar checks
against Boltz, ChemProp, KERMT, MolFormer, AttentiveFP, and GatedGCN OOF runs did
not show a useful residual axis.

Hidden-state raw extraction smoke:

- 3 compounds completed.
- Example `00018` raw shapes:
  - `t1p000_pocket`: `(494, 128)`
  - `t1p000_node`: `(23, 320)`
  - `t1p000_edge`: `(506, 96)`
  - the same block structure was saved for `t0p500` and `t0p050`.
- Mean/std pooling produced a `(3, 3264)` matrix.
- Rich pooling with mean/std/max/min/q25/q50/q75 plus timestep deltas produced a
  `(3, 15232)` matrix.

Full hidden-state extraction:

```bash
pixi run python track1_activity/scripts/run_pocketxmol_embedding_probe.py \
  --run-name pxm_hidden_raw_steps3_all \
  --all-compounds \
  --timesteps 1.0 0.5 0.05 \
  --device cuda:0
```

- Raw hidden states completed for 4652 / 4652 compounds with available Boltz
  structures: 4139 train and 513 test.
- One train compound in the DB lacks the required structure coverage and is not
  included in these PocketXMol hidden-state probes.
- Raw hidden artifacts occupy about 3.1 GB under
  `structures/pocketxmol_probe/pxm_hidden_raw_steps3_all/raw_hidden/`.

Boltz-inspired re-pooling variants were generated from the same raw hidden files
without rerunning PocketXMol. The closest reusable ideas from the Boltz trunk
workflow were raw-NPZ re-pooling, per-block statistics, quantiles, and timestep
deltas. PocketXMol region pooling is not available from this first raw dump
because residue/region membership was not saved alongside `h_pocket`.

Five-fold canonical UMAP CV over the 4139 covered train rows:

| pooled feature | model | dims | MAE | Spearman |
|---|---|---:|---:|---:|
| node+edge mean/std + t1.0-minus-t0.05 delta | Ridge | 3328 | 0.58155 | 0.68376 |
| all blocks mean/std/q10/q90 + delta | Ridge | 8704 | 0.58233 | 0.68427 |
| all blocks mean/std | Ridge | 3264 | 0.58234 | 0.68245 |
| rich stats mean/std/max/min/q25/q50/q75 + delta | Ridge | 15232 | 0.58245 | 0.67867 |
| node-only mean/std + delta | Ridge | 2560 | 0.58382 | 0.68145 |
| t0.05-only mean/std | Ridge | 1088 | 0.58425 | 0.68274 |
| t0.5-only mean/std | Ridge | 1088 | 0.60275 | 0.66401 |
| t1.0-only mean/std | Ridge | 1088 | 0.63080 | 0.62492 |
| edge-only mean/std + delta | Ridge | 768 | 0.63755 | 0.62946 |
| pocket-only mean/std + delta | Ridge | 1024 | 0.77612 | 0.33676 |

Fold-safe top-512 feature selection did not improve the best Ridge variants.
PCA-256 Ridge/HGB checks on the top four variants also did not improve the best
linear result; the best PCA result was rich-stats PCA Ridge at MAE 0.58829, and
PCA HGB was worse at MAE 0.61104 or higher.

The best PocketXMol hidden OOF (`node+edge mean/std + delta`, Ridge) is much
stronger than the four scalar confidence scores, but it is still far weaker than
the current top-tier ligand/auxiliary models. Its OOF predictions are also highly
correlated with existing members:

| reference OOF | reference MAE | Pearson vs PocketXMol | Spearman vs PocketXMol |
|---|---:|---:|---:|
| `tabpfn_cheme_2d_full_boltz_log2fc_pred_seed10ens_top500_umap_v3` | 0.39715 | 0.81637 | 0.78275 |
| `tabpfn_pooled_boltz_allpairs_umap_default_v3` | 0.49833 | 0.87408 | 0.84199 |
| `tabpfn_pooled_boltz_umap_default_v3` | 0.50326 | 0.86956 | 0.83245 |
| `tabpfn_molformer_c3_pretrain_embed_umap_default_v3` | 0.48010 | 0.83958 | 0.80965 |
| `tabpfn_kermt_pretrain_embed_umap_default_v3` | 0.44981 | 0.82793 | 0.79415 |

Interpretation: PocketXMol hidden-state pooling is a real feature source and is
not equivalent to the scalar confidence outputs, but this first pass mostly
recovers a familiar ligand/3D axis. It does not currently justify a submission
on its own. If revisited, the next useful extraction change would be to save
pocket residue metadata so Boltz-style region/core-pocket pooling can be tested
without another blind global mean-pooling pass.

## Interpretation

PocketXMol is practical to run locally and is not prohibitively heavy for a
single-sample confidence pass. The useful distinction from ChemProp is real: this
is a pocket-conditioned 3D denoising model, not a ligand-only encoder.

However, the readily exposed confidence outputs are not strong enough to treat
as submission material. The full pass shows weak standalone signal and essentially
no useful residual alignment with the current strongest models.

The remaining potentially interesting axis is hidden-state pooling from the
pocket encoder and denoiser. The extraction path now saves raw variable-length
hidden arrays first, matching the Boltz-style workflow where pooling can be
changed after the expensive forward pass.

## Decision

Do not submit or ensemble scalar PocketXMol confidence features.

Keep the wrapper for future probes. If spending more time here, prioritize a
metadata-aware hidden-state extraction for Boltz-style region/core-pocket pooling
over additional scalar-confidence sampling.

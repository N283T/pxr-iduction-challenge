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

## Results

Runtime:

- 200 compounds completed successfully.
- Mean elapsed time was 8.095 s per compound.
- Total wall-clock for the 200-compound run was 26.98 min.
- A scalar-only full 4652-compound pass is therefore roughly 10-11 GPU-hours at
  this speed, before extra samples/seeds.

200-compound scalar probe composition:

- 50 lowest-activity train compounds
- 50 highest-activity train compounds
- 50 deterministic-random train compounds
- 50 deterministic-random test compounds

Scalar confidence correlations versus pEC50 on the 150 train rows:

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

## Interpretation

PocketXMol is practical to run locally and is not prohibitively heavy for a
single-sample confidence pass. The useful distinction from ChemProp is real: this
is a pocket-conditioned 3D denoising model, not a ligand-only encoder.

However, the readily exposed confidence outputs are not strong enough to treat
as submission material. They may still be useful as diagnostics, but the current
evidence does not justify a full scalar-confidence feature build for ensembling.

The remaining potentially interesting axis is not the public confidence scalar;
it is hidden-state pooling from the pocket encoder or denoiser. That requires a
small upstream-script fork or monkeypatch to return `h_pocket`, `h_node`, and
`h_edge` from `PMAsymDenoiser.forward`.

## Decision

Do not submit or ensemble scalar PocketXMol confidence features yet.

Keep the wrapper for future probes. If spending more time here, prioritize a
hidden-state extraction run over scaling scalar confidence to all compounds.

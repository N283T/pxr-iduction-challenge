# GatorAffinity Axis

Status: closed negative-control protein-ligand axis.

This directory contains the GatorAffinity Phase A/B scripts. The main value now
is historical: it records a protein-ligand direction that was tested and did not
produce a useful Track 1 feature compared with the stronger Boltz-derived axes.

## Scripts

| Script | Purpose |
|---|---|
| `01_build_inputs.py` | Build GatorAffinity inputs from challenge compounds and structures. |
| `02_analyze.py` | Analyze zero-shot GatorAffinity predictions against available labels. |
| `03_build_fold_pkls.py` | Build UMAP-fold train/validation pickle files for fold-level fine-tuning. |
| `04_infer_fold.py` | Run fold inference for a trained GatorAffinity fold. |
| `05_run_all_folds.sh` | Shell wrapper for all fold fine-tuning/inference runs. |
| `06_collate_oof.py` | Collate fold predictions into an OOF experiment record. |
| `07_zeroshot_as_feature.py` | Test Gator zero-shot predictions as extra features for a 2D/Boltz baseline. |

## Cleanup Stance

Keep this as a closed negative-control axis. Reopen only if the external model,
input representation, or protein-ligand target changes materially.

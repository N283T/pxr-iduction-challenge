# PXR Induction Challenge

Entry for the [OpenADMET PXR Blind Challenge](https://huggingface.co/spaces/openadmet/pxr-challenge) (April–July 2026).

## Leaderboard

**7th place** (as of 2026-04-07) out of 31 teams.

| Rank | Team | RAE | R² | Spearman |
|------|------|-----|-----|----------|
| 1 | nova | 0.58 | 0.54 | 0.83 |
| 2 | MirthEngine | 0.59 | 0.55 | 0.81 |
| 3 | Asidsal11 | 0.59 | 0.53 | 0.79 |
| 4 | Radi | 0.60 | 0.52 | 0.82 |
| 5 | N1NC1O | 0.60 | 0.52 | 0.79 |
| 6 | deoxys | 0.62 | 0.49 | 0.80 |
| **7** | **N283T (us)** | **0.62** | **0.53** | **0.80** |

## Overview

Predicting human Pregnane X Receptor (PXR) activity from molecular structure. PXR is a nuclear receptor critical for drug metabolism (ADMET).

### Tracks

| Track | Task | Metric | Test Size |
|-------|------|--------|-----------|
| 1 - Activity | Predict pEC50 | RAE ↓ | 513 compounds |
| 2 - Structure | Predict protein-ligand 3D | LDDT-PLI ↑ | 78 compounds |

## Approach

Following the methodology from [EUOS25 winning solution](https://www.yumizsui.com/blog/euos25-challenge/):

1. **SMILES standardization**: ChEMBL structure pipeline (salt stripping, neutralization)
2. **Diverse single-feature models**: Each model uses one feature type, individually Optuna-tuned
3. **UMAP chemical space split CV**: Morgan FP + UMAP + KMeans clustering for realistic OOF estimates
4. **Gap regularization**: Penalizes train-val gap to improve out-of-distribution generalization
5. **Residual learning**: Physicochemical base model + Mordred residual model
6. **Weighted ensemble**: L2-regularized blend of 23 diverse models (LightGBM + GNN + Transformer)

### Feature Sources

| Type | Features | Source |
|------|----------|--------|
| Descriptors | RDKit 2D (41d), Mordred 2D (1402d) | RDKit, mordredcommunity |
| Fingerprints | Morgan, MACCS, Avalon, AtomPair (167-2048d) | RDKit |
| Embeddings | ChemBERTa (7 variants, 384-768d) | HuggingFace |
| Embeddings | CheMeleon MPNN (300d) | Zenodo pretrained |
| Embeddings | BERT-base-SMILES (768d) | HuggingFace |
| Embeddings | MoLFormer-XL (768d) | IBM/HuggingFace |
| Graph | ChemProp D-MPNN (Optuna-tuned) | chemprop |
| Graph | AttentiveFP (Optuna-tuned) | PyG |
| Graph | CheMeleon fine-tune | chemprop |

### Ensemble

23 models blended with L2-regularized weights (scipy minimize on OOF RAE):

| Model | Weight | Solo RAE |
|-------|--------|----------|
| chemprop_optuna_umap | 26% | 0.572 |
| chemprop_scaffold | 14% | 0.620 |
| residual_physprop+mordred | 13% | 0.580 |
| attentivefp_optuna_umap | 12% | 0.580 |
| lgbm_mordred_umap | 6% | 0.576 |
| lgbm_molformer_xl_umap | 6% | 0.646 |
| + 17 others | 23% | — |

## Setup

### Prerequisites

- [pixi](https://pixi.sh/) (package manager)
- GPU with CUDA (for GNN models and embedding computation)
- WSL2: `CONDA_OVERRIDE_CUDA=13.1` set automatically via shell config

### Install & Run

```bash
pixi install

# Download dataset
pixi run python download_data.py

# Start PostgreSQL + full DB setup
pixi run db-start
pixi run psql -h /tmp -p 5433 -c "CREATE DATABASE pxr_challenge;"
pixi run psql -h /tmp -p 5433 -d pxr_challenge -c "CREATE EXTENSION rdkit;"
pixi run psql -h /tmp -p 5433 -d pxr_challenge -f db/schema.sql
pixi run python db/load_data.py
pixi run psql -h /tmp -p 5433 -d pxr_challenge -f db/add_std_columns.sql
pixi run python db/standardize_compounds.py
pixi run psql -h /tmp -p 5433 -d pxr_challenge -f db/recompute_descriptors.sql
pixi run psql -h /tmp -p 5433 -d pxr_challenge -f db/experiments_schema.sql
pixi run python db/compute_mordred.py
pixi run python db/compute_embeddings.py
pixi run python db/compute_chemeleon.py
```

### Reproducing Results

```bash
# 1. LightGBM single-feature models (CPU, ~30 min each)
pixi run python track1_activity/scripts/run_train.py --model lgbm --feature mordred --split umap
pixi run python track1_activity/scripts/run_train.py --model lgbm --feature chemeleon --split umap
# ... repeat for other features

# 2. Deep learning models (GPU, ~1-4 hours each)
pixi run python track1_activity/scripts/run_chemprop_optuna.py --n-trials 50 --split umap
pixi run python track1_activity/scripts/run_attentivefp_optuna.py --n-trials 30 --split umap
pixi run python track1_activity/scripts/run_residual_learning.py --split umap
# Or run all DL models sequentially:
bash track1_activity/scripts/run_all_models.sh

# 3. Ensemble
pixi run python track1_activity/scripts/run_ensemble.py
```

### Useful Commands

```bash
pixi run db-start / db-stop / db-psql
pixi run db-psql -c "SELECT * FROM experiment_summary;"
```

## Project Structure

```
data/                        # Parquet datasets (gitignored)
db/
  schema.sql                 # Core tables: compounds, activities
  standardize_compounds.py   # ChEMBL pipeline standardization
  compute_mordred.py         # Mordred 2D descriptors → DB (JSONB)
  compute_embeddings.py      # ChemBERTa/BERT/MoLFormer variants → DB
  compute_chemeleon.py       # CheMeleon MPNN fingerprints → DB
  experiments_schema.sql     # Experiment tracking tables
docs/
  track1_eda_report.md       # EDA findings + feature importance + distribution analysis
  literature_qsar_ml.md      # PXR QSAR/ML literature review
  literature_wet_lab.md      # PXR wet-lab/biology literature review
  leaderboard_2026-04-07.csv # Leaderboard snapshot (latest)
track1_activity/
  src/
    data.py                  # DB loading (SQLAlchemy, ORDER BY t.id)
    features.py              # FP_REGISTRY: 12 fingerprint types
    evaluate.py              # Metrics, DB recording, OOF storage
    splits.py                # Murcko scaffold + UMAP split CV
  scripts/
    run_train.py                  # Unified LightGBM/XGBoost/CatBoost training
    run_chemprop_optuna.py        # ChemProp D-MPNN Optuna tuning
    run_attentivefp_optuna.py     # AttentiveFP (PyG) Optuna tuning
    run_molformer_finetune.py     # MoLFormer-XL fine-tuning
    run_residual_learning.py      # Two-stage residual learning
    run_ensemble.py               # Weighted ensemble optimization
    run_all_models.sh             # Sequential DL training pipeline
    api.py                        # OpenADMET API client: fetch leaderboard + submit (gitignored, contains PII)
    archive/                      # Early exploration scripts
  notebooks/                 # marimo notebooks
  submissions/               # Submission CSVs (gitignored)
track2_structure/            # Structure prediction (future)
```

## Data Sources

All data from [openadmet/pxr-challenge-train-test](https://huggingface.co/datasets/openadmet/pxr-challenge-train-test) (Apache 2.0).

| Dataset | Rows | Description |
|---------|------|-------------|
| Train (default) | 4,140 | Dose-response with pEC50, Emax |
| Test (blinded) | 513 | pEC50 to predict |
| Counter-assay | 2,860 | PXR-null control |
| Single-concentration | 21,014 | Single-dose screening |

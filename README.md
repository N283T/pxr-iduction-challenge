# PXR Induction Challenge

Entry for the [OpenADMET PXR Blind Challenge](https://huggingface.co/spaces/openadmet/pxr-challenge) (April–July 2026).

## Overview

Predicting human Pregnane X Receptor (PXR) activity from molecular structure. PXR is a nuclear receptor critical for drug metabolism (ADMET).

### Tracks

| Track | Task | Metric | Test Size |
|-------|------|--------|-----------|
| 1 - Activity | Predict pEC50 | RAE ↓ | 513 compounds |
| 2 - Structure | Predict protein-ligand 3D | LDDT-PLI ↑ | 78 compounds |

## Setup

### Prerequisites

- [pixi](https://pixi.sh/) (package manager)
- GPU with CUDA (optional, for GNN models)

### Install & Run

```bash
# Install dependencies
pixi install

# Download dataset
pixi run python download_data.py

# Start PostgreSQL + load data
pixi run db-start
pixi run psql -h /tmp -p 5433 -c "CREATE DATABASE pxr_challenge;"
pixi run psql -h /tmp -p 5433 -d pxr_challenge -c "CREATE EXTENSION rdkit;"
pixi run psql -h /tmp -p 5433 -d pxr_challenge -f db/schema.sql
pixi run python db/load_data.py
pixi run psql -h /tmp -p 5433 -d pxr_challenge -f db/compute_descriptors.sql
pixi run psql -h /tmp -p 5433 -d pxr_challenge -f db/experiments_schema.sql
```

### Useful Commands

```bash
pixi run db-start          # Start PostgreSQL
pixi run db-stop           # Stop PostgreSQL
pixi run db-psql           # Connect to DB

# Compare experiments
pixi run db-psql -c "SELECT * FROM experiment_summary;"
```

## Current Results

| Experiment | Model | Features | RAE | R² | Spearman |
|-----------|-------|----------|-----|-----|----------|
| baseline_lgbm_descriptors | LightGBM | RDKit 2D descriptors | 0.6235 | 0.5180 | 0.6584 |

## Project Structure

```
data/                        # Parquet datasets (gitignored)
db/                          # Database schema & scripts
docs/                        # Analysis reports
track1_activity/             # Activity prediction (pEC50)
  notebooks/                 # marimo notebooks
  scripts/                   # Model & analysis scripts
  submissions/               # Submission files
track2_structure/            # Structure prediction
```

## Data Sources

All data from [openadmet/pxr-challenge-train-test](https://huggingface.co/datasets/openadmet/pxr-challenge-train-test) (Apache 2.0).

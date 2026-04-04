# PXR Induction Challenge

## Competition

OpenADMET PXR Blind Challenge (April 1 – July 1, 2026)
https://huggingface.co/spaces/openadmet/pxr-challenge

Two tracks:
- **Track 1 (Activity)**: Predict pEC50 for 513 blinded compounds. Primary metric: RAE.
- **Track 2 (Structure)**: Predict protein-ligand 3D structures for 78 compounds. Primary metric: LDDT-PLI.

## Environment

- **Package manager**: pixi (conda-forge)
- **Python**: 3.12
- **Database**: PostgreSQL 18 + RDKit cartridge (port 5433, socket /tmp)
- **GPU**: RTX 5080 (16GB VRAM)

### DB Commands

```bash
pixi run db-start    # Start PostgreSQL
pixi run db-stop     # Stop PostgreSQL
pixi run db-psql     # Connect to pxr_challenge DB
```

### DB Setup (after fresh clone)

```bash
pixi run db-start
pixi run python download_data.py            # Download parquet files
pixi run psql -h /tmp -p 5433 -d pxr_challenge -f db/schema.sql
pixi run python db/load_data.py             # Load data into DB
pixi run psql -h /tmp -p 5433 -d pxr_challenge -f db/add_std_columns.sql
pixi run python db/standardize_compounds.py  # ChEMBL pipeline standardization
pixi run psql -h /tmp -p 5433 -d pxr_challenge -f db/recompute_descriptors.sql
pixi run psql -h /tmp -p 5433 -d pxr_challenge -f db/experiments_schema.sql
```

## Project Structure

```
data/                        # Parquet files (gitignored, re-downloadable)
db/
  schema.sql                 # Core tables: compounds, train/test/counter/single_conc
  add_std_columns.sql        # Add std_smiles/std_mol columns
  standardize_compounds.py   # ChEMBL pipeline standardization
  recompute_descriptors.sql  # Descriptors & fingerprints from std_mol
  experiments_schema.sql     # Experiment tracking tables
  load_data.py               # Data loader script
  pgdata/                    # PostgreSQL data dir (gitignored)
docs/
  track1_eda_report.md       # EDA findings
track1_activity/
  notebooks/                 # marimo notebooks
  scripts/                   # Analysis & model scripts
  submissions/               # CSV/parquet submission files
track2_structure/
  notebooks/
  scripts/
  submissions/
```

## DB Schema

### Core Tables
- `compounds` — SMILES + RDKit mol column (13,136 rows)
- `train_activity` — pEC50 dose-response data (4,140 rows)
- `test_activity` — Blinded test compounds (513 rows)
- `counter_assay` — PXR-null control data (2,860 rows)
- `single_concentration` — Single-dose screening (21,014 rows)

### Computed Tables
- `compound_descriptors` — 38 RDKit 2D descriptors + scaffold + formula + InChIKey
- `compound_fingerprints` — Morgan, FeatMorgan, MACCS, AtomPair, Avalon FPs

### Experiment Tracking
- `experiments` — Model config, hyperparameters, submission path
- `experiment_cv_results` — Per-fold CV metrics
- `experiment_summary` — View: aggregated metrics sorted by RAE

## Conventions

- Notebooks: use **marimo** (not Jupyter)
- All code, comments, commits in **English**
- Experiment results go into `experiments` + `experiment_cv_results` tables
- Submission files stored in `track{1,2}_*/submissions/`
- Compare experiments: `pixi run db-psql -c "SELECT * FROM experiment_summary;"`

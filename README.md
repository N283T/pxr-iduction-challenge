# PXR Induction Challenge

Entry for the [OpenADMET PXR Blind Challenge](https://huggingface.co/spaces/openadmet/pxr-challenge) (April–July 2026).

## Leaderboard

**7th place** (as of 2026-04-21) out of 89+ teams. Research log: [issue #100](https://github.com/N283T/pxr-iduction-challenge/issues/100).

| Rank | Team | MAE ↓ | RAE ↓ | R² ↑ | Spearman ↑ |
|------|------|-------|-------|------|------------|
| 1 | Yan | 0.4142 | 0.5200 | 0.635 | 0.848 |
| 2 | sia | 0.4160 | 0.5225 | 0.640 | 0.847 |
| 3 | Ray_art | 0.4197 | 0.5269 | 0.617 | 0.843 |
| 4 | W1175 | 0.4202 | 0.5276 | 0.621 | 0.844 |
| 5 | nova | 0.4218 | 0.5297 | 0.635 | 0.835 |
| 6 | histidine | 0.4236 | 0.5322 | 0.644 | 0.834 |
| **7** | **N283T (us)** | **0.4358** | **0.5474** | **0.634** | **0.841** |
| 8 | Zoboomafoo83 | 0.4368 | 0.5485 | 0.611 | 0.832 |
| 9 | discoverybytes | 0.4397 | 0.5519 | 0.586 | 0.835 |
| 10 | jordanshivers | 0.4415 | 0.5544 | 0.594 | 0.818 |

Primary metric is **MAE** (lower is better). Top-3 gap: +0.016 MAE (half of what it was before post-hoc calibration).

## Overview

Predicting human Pregnane X Receptor (PXR) activity from molecular structure. PXR is a nuclear receptor critical for drug metabolism and is implicated in clinically relevant drug-drug interactions.

### Tracks

| Track | Task | Metric | Test Size |
|-------|------|--------|-----------|
| 1 - Activity | Predict pEC50 | MAE ↓ (primary), RAE/R²/Spearman | 513 compounds |
| 2 - Structure | Predict protein-ligand 3D | LDDT-PLI ↑ | 78 compounds |

## Approach (Track 1)

1. **SMILES standardization**: ChEMBL structure pipeline (salt strip + neutralize + canonicalize), with post-hoc repair for Boltz-2-incompatible bridged bicyclics.
2. **Diverse, decorrelated pool members**: Ten single-feature or single-architecture models, each individually Optuna-tuned on a canonical 5-fold UMAP-cluster split.
3. **Canonical CV**: Morgan FP + UMAP → KMeans 50 clusters → 5 folds (seed=42). Chosen after a 12+ variant bake-off; see PR #70.
4. **Pretrain + frozen + embed recipe (Buterez 2024 strategy-3)**: Pretrain an encoder on weak labels (single-concentration log2_fc), freeze it, extract 300–768d embeddings for all 13,136 compounds, run TabPFN v7 on those for the pEC50 regression. Three pool members (chemprop / MoLFormer-c3 / predicted-log2fc as TabPFN feature) use this pattern.
5. **Boltz-2 structure features**: Full-pose inference for 4,653 compounds (~4 d on RTX 5080) yields confidence/affinity/pose geometry that feed two pool members (`lgbm_pooled_boltz`, `tabpfn_pooled_boltz`).
6. **Caruana forward-selection ensemble (bagged 20×)**: Discrete count-based weights, structurally robust against the weight-zero-sum failure mode seen with L2/vanilla optimizers when correlated-strong members are added (see issue #82).
7. **Post-hoc regression calibration**: Positive-constrained affine (`slope ≥ 0`) applied to the final ensemble output. Lifted the submission from rank 10 → 7 on 2026-04-21 (PR #101).

### Feature Sources

| Type | Features | Source |
|------|----------|--------|
| Descriptors | RDKit 2D (217d), Mordred 2D (~1460d), Jazzy H-bond | RDKit, mordredcommunity, jazzy |
| Boltz-2 pose | Pose-derived 2D (full + jazzy), 3D descriptors, confidence/affinity/plddt/pae/pde vectors | Boltz-2 outputs |
| Embeddings | ChemBERTa (7 variants), BERT-SMILES, MoLFormer-XL, CheMeleon MPNN (300d) | HuggingFace, Zenodo |
| Pretrained (frozen+embed) | ChemProp / MoLFormer-c3 / GatedGCN, pretrained on log2_fc | In-repo |
| Graphs | ChemProp D-MPNN, AttentiveFP, GatedGCN (Optuna-tuned) | chemprop, PyG |

### Ensemble (10 members, caruana_bag20 + linear_pos calibration)

| Pool member | Caruana weight |
|---|---|
| tabpfn_chemprop_pretrain_embed_umap_default | 0.355 |
| tabpfn_2d_full_boltz_log2fc_pred_umap_default | 0.231 |
| tabpfn_pooled_boltz_allpairs_umap_default | 0.112 |
| tabpfn_pooled_boltz_umap_default | 0.089 |
| tabpfn_molformer_c3_pretrain_embed_umap | 0.078 |
| chemprop_chemeleon_umap | 0.042 |
| chemprop_optuna_umap | 0.038 |
| gatedgcn_pretrain_finetune_frozen_umap | 0.037 |
| lgbm_pooled_boltz_umap | 0.011 |
| attentivefp_optuna_umap | 0.007 |

Ensemble output is then calibrated via `linear_pos` (positive-constrained affine) chosen from 4 candidates (linear, linear_pos, spline_k5, isotonic) by nested CV MAE with a Spearman guardrail.

## Setup

### Prerequisites

- [pixi](https://pixi.sh/) (conda-forge + pypi)
- Python 3.12
- PostgreSQL 18 with the RDKit cartridge
- GPU (RTX 5080 16 GB in our setup; required for GNN/transformer training and Boltz-2)
- WSL2: `CONDA_OVERRIDE_CUDA=13.1` set via shell config

### Install & DB setup

```bash
pixi install
pixi run db-start
pixi run python download_data.py

pixi run psql -h /tmp -p 5433 -c "CREATE DATABASE pxr_challenge;"
pixi run psql -h /tmp -p 5433 -d pxr_challenge -c "CREATE EXTENSION rdkit;"
pixi run psql -h /tmp -p 5433 -d pxr_challenge -f db/schema.sql
pixi run python db/load_data.py
pixi run psql -h /tmp -p 5433 -d pxr_challenge -f db/add_std_columns.sql
pixi run python db/standardize_compounds.py
pixi run python db/fix_bridged_stereo.py --apply
pixi run psql -h /tmp -p 5433 -d pxr_challenge -f db/recompute_descriptors.sql
pixi run psql -h /tmp -p 5433 -d pxr_challenge -f db/experiments_schema.sql
pixi run psql -h /tmp -p 5433 -d pxr_challenge -f db/lb_submissions_schema.sql
pixi run python db/compute_mordred.py
pixi run python db/compute_jazzy.py
pixi run python db/compute_embeddings.py
pixi run python db/compute_chemeleon.py
pixi run psql -h /tmp -p 5433 -d pxr_challenge -f db/compound_descriptors_full_schema.sql
pixi run python db/compute_rdkit_descriptors_full.py
```

Boltz-2 (separate `uv tool` env) is set up per the runbook in `CLAUDE.md`.

### Reproducing the current submission

```bash
# 1. Train / refresh pool members (GPU heavy — see CLAUDE.md for per-script commands)
bash track1_activity/scripts/run_all_models.sh

# 2. Build caruana_bag20 ensemble + write ens_caruana_bag20.csv
pixi run python track1_activity/scripts/run_ensemble.py

# 3. Apply post-hoc calibration + write ens_caruana_bag20_calibrated_best.csv
pixi run python track1_activity/scripts/run_ensemble_calibrate.py

# 4. Submit (observes 4h cooldown)
pixi run python track1_activity/scripts/api.py submit \
    track1_activity/submissions/ens_caruana_bag20_calibrated_best.csv \
    --notes "..."

# 5. Check leaderboard / local submission history
pixi run python track1_activity/scripts/api.py fetch
pixi run python track1_activity/scripts/api.py status
pixi run python track1_activity/scripts/api.py cooldown
```

### Useful commands

```bash
pixi run db-start / db-stop / db-psql
pixi run db-psql -c "SELECT * FROM experiment_summary LIMIT 20;"
```

## Project Structure

```
data/                        # Parquet datasets (gitignored)
db/
  schema.sql                 # Core tables: compounds, train/test/counter/single_conc
  experiments_schema.sql     # Experiment tracking + OOF predictions
  lb_submissions_schema.sql  # Local LB submission history
  standardize_compounds.py   # ChEMBL standardization
  fix_bridged_stereo.py      # Hamming-1 cis-bridgehead repair (Boltz-2 input fixup)
  compute_*.py               # Feature precomputation (mordred, jazzy, embeddings, etc.)
  boltz2_*_schema.sql        # Boltz-2 prediction outputs (pose, posebusters)
docs/
  superpowers/specs/         # Approved feature design docs
  superpowers/plans/         # Implementation plans
  track1_eda_report.md       # EDA + feature importance
  literature_qsar_ml.md      # QSAR/ML literature review
  literature_wet_lab.md      # Wet-lab / biology context
  leaderboard_YYYY-MM-DD.csv # LB snapshots
track1_activity/
  src/
    data.py                  # DB loading (SQLAlchemy, deterministic ORDER BY)
    features.py              # FP_REGISTRY + per-feature loaders
    evaluate.py              # Metrics, DB recording, OOF storage
    splits.py                # UMAP / scaffold CV
    losses.py                # Custom chemprop losses (relative-distance aux)
    peft_backbones.py        # LoRA backbone registry (molformer_xl, molformer_c3)
    peft_methods.py          # LoRA / PEFT method registry
    peft_trainer.py          # Shared PEFT regressor trainer
    pyg_training.py          # PyG graph training helpers
    pseudo_labels.py         # Weak-label prep for counter-assay
  scripts/
    run_train.py                    # Unified LightGBM/XGBoost/CatBoost + TabPFN
    run_ensemble.py                 # Caruana forward-selection ensemble (bagged 20×)
    run_ensemble_calibrate.py       # Post-hoc calibration (linear / linear_pos / spline_k5 / isotonic) + best selection
    run_chemprop_optuna.py          # ChemProp D-MPNN Optuna
    run_chemprop_pretrain.py        # Pretrain chemprop on log2_fc (weak label)
    run_chemprop_embed_extract.py   # Extract [frozen] chemprop encoder embeddings
    run_chemprop_predict_log2fc.py  # Use pretrained encoder to predict log2_fc for test
    run_chemprop_finetune.py        # Frozen-encoder head FT on pEC50
    run_chemprop_chemeleon.py       # CheMeleon foundation finetune
    run_chemprop_multitask.py       # Multitask (pec50 + log2_fc) head
    run_chemprop_multitask_desc.py  # Multitask with descriptor aux (negative result)
    run_chemprop_relative_aux.py    # FMGCL relative-distance aux loss (negative result)
    run_attentivefp_optuna.py       # AttentiveFP (PyG) Optuna
    run_attentivefp_pretrain_finetune.py
    run_gatedgcn_optuna.py          # GatedGCN (PyG) Optuna
    run_gatedgcn_pretrain_finetune.py
    run_gin_optuna.py / run_graphgps_optuna.py
    run_molformer_c3_pretrain.py    # MoLFormer-c3 pretrain on log2_fc
    run_molformer_c3_embed_extract.py
    run_peft_finetune.py            # Generic PEFT/LoRA finetune (negative result on pEC50)
    run_residual_learning.py        # Physprop + Mordred two-stage
    boltz_affhead/                  # Boltz-2 trunk-embedding retarget (issue #74)
    run_all_models.sh               # Sequential DL training pipeline
    api.py                          # OpenADMET LB client (fetch / submit / status / cooldown)
  notebooks/                 # marimo notebooks
  submissions/               # Submission CSVs (gitignored)
track2_structure/
  src/boltz2/                # Input builder + post-processor
  scripts/                   # Boltz-2 full run, recovery, embeddings, posebusters
structures/                  # Boltz-2 artifacts (MSA, inputs, outputs) — gitignored
```

## Data Sources

All data from [openadmet/pxr-challenge-train-test](https://huggingface.co/datasets/openadmet/pxr-challenge-train-test) (Apache 2.0).

| Dataset | Rows | Description |
|---------|------|-------------|
| Train | 4,140 | Dose-response with pEC50 |
| Test (blinded) | 513 | pEC50 to predict (all labels withheld) |
| Counter-assay | 2,860 | PXR-null control |
| Single-concentration | 21,014 | Single-dose screening (log2_fc at 8.25 µM and 33 µM) |

Additional structural context: AlphaFold AF-O75469 LBD + 72 PDB holo structures (Track 2), Boltz-2 pose predictions (Track 1 + 2).

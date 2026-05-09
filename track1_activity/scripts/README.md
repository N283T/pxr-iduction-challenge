# Track 1 Scripts

This directory mixes current pipeline entry points with historical experiment
scripts. Archive or document old experiment axes before moving or deleting
files.

## Current Workflow

The normal Track 1 flow is:

1. Train or register model predictions with `run_train.py` or a focused model
   script.
2. Build the explicit allow-list ensemble with `run_ensemble.py`.
3. Re-run both calibrators:
   - `run_ensemble_calibrate.py`
   - `run_ensemble_calibrate_importance.py`
4. Submit through the local ignored submission client and
   `scheduled_submit.sh`.

## Local-Only Scripts

`api.py` and `submit.py` are intentionally ignored by git. They can contain
personal tokens or account-specific submission state. Do not recreate, commit,
move, or delete them as part of repository cleanup.

## Directory Guide

| Directory | Purpose |
|---|---|
| `boltz_affhead/` | Boltz-2 trunk/affinity/head pooling and ensemble bakeoff experiments. |
| `archive/` | Superseded baseline, scaffold, and ensemble scripts kept for history. |
| `eda_cv_prep/` | CV split, OOF/LB gap, and 3D feature preparation diagnostics. |
| `eda_redo/` | Rebuilt EDA and CHeMBL lookup scripts from the April analysis pass. |
| `multitask_aux/` | Auxiliary target selection diagnostics. |
| `unimol/` | Uni-Mol v2 experiments; active hold / promising revisit candidate despite earlier null pool attempts. |
| `clamp/` | CLAMP raw encoder experiment; closed/null as of PR #159. |
| `gator/` | GatorAffinity experiments; closed negative control for protein-ligand directions. |
| `run_gsl_mpp_lite.py` | Experimental GSL-MPP-inspired molecule-graph residual smoothing probe for Track 1. |
| `run_gsl_mpp_learned.py` | Experimental learned GSL-MPP-style molecule-graph residual model for Track 1. |
| `run_gatedgcn_strategy6.py` | Buterez 2024 Strategy 6 probe: frozen GatedGCN low-fidelity encoder plus adaptive readout. |
| `run_chemprop_strategy6.py` | Buterez 2024 Strategy 6 probe: frozen ChemProp low-fidelity encoder plus adaptive readout. |
| `run_kan_embed.py` | pykan/KAN regressor probe on frozen molecular embeddings. |
| `run_ka_gnn.py` | PyG Fourier KA-GNN probe for direct molecular graph regression. |
| `run_ka_gnn_pretrain.py` | KA-GNN low-fidelity log2_fc pretraining for frozen embedding extraction. |
| `run_ka_gnn_embed_extract.py` | Extract frozen KA-GNN graph embeddings from a pretrain checkpoint. |
| `drugclip/` | DrugCLIP external-data feature extraction scripts. |

When in doubt, archive or document first. Delete only after a separate review
confirms that the script is duplicated, unrecoverable, or no longer meaningful.

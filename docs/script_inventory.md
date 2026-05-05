# Track 1 Script Inventory

This document is a cleanup map for `track1_activity/scripts/`. It is not a
deletion list. The repository contains many one-off experiment scripts that are
still useful for reconstructing issue and PR history, even when their model axis
is no longer competitive.

## Ground Rules

- Do not delete tracked experiment scripts only because the result was null.
  Move them to an archive area first, or document why they are obsolete.
- Do not touch ignored local submission clients such as
  `track1_activity/scripts/api.py` or `track1_activity/scripts/submit.py`.
  These files can contain personal tokens or account-specific state.
- Keep generated artifacts out of git. Rebuild feature parquet, checkpoint,
  submission, and structure artifacts from scripts or the database when needed.
- Prefer updating this inventory before moving scripts so future cleanup changes
  are reviewable.

## Current Entry Points

These are the scripts that should remain easy to find from the top level:

| Path | Role |
|---|---|
| `track1_activity/scripts/run_train.py` | Canonical model runner and feature loader registry for LightGBM/XGBoost/CatBoost/TabPFN experiments. |
| `track1_activity/scripts/run_ensemble.py` | Canonical explicit allow-list ensemble builder. |
| `track1_activity/scripts/run_ensemble_calibrate.py` | Nested-CV post-hoc calibrator for the canonical ensemble. |
| `track1_activity/scripts/run_ensemble_calibrate_importance.py` | Density-ratio importance-weighted affine calibrator; keep rerunning when the ensemble pool changes. |
| `track1_activity/scripts/scheduled_submit.sh` | Cooldown-aware submission wrapper. |
| `track1_activity/scripts/build_log2fc_seed_ensemble.py` | ChemProp log2fc seed-ensemble builder for the current frozen-encoder recipe. |
| `track1_activity/scripts/build_kermt_seed_ensemble.py` | KERMT seed-ensemble builder. |
| `track1_activity/scripts/build_boltz2_contact_features.py` | Boltz-2 pose contact feature builder registered in `run_train.py`. |
| `track1_activity/scripts/build_dose_response_latent.py` | Dose-response latent feature builder registered in `run_train.py`. |
| `track1_activity/scripts/run_chemprop_pretrain.py` | Current ChemProp single-concentration pretrain recipe. |
| `track1_activity/scripts/run_chemprop_predict_log2fc.py` | Log2fc prediction from pretrained ChemProp checkpoints. |
| `track1_activity/scripts/run_chemprop_assay_shape_pretrain.py` | Assay-shape pretrain axis. |
| `track1_activity/scripts/run_chemprop_assay_shape_embed_extract.py` | Embedding extractor for the assay-shape axis. |

`track1_activity/scripts/api.py` is intentionally absent from this list because
it is ignored and local-only.

## Active Subdirectories

| Directory | Status | Notes |
|---|---|---|
| `boltz_affhead/` | Active research axis | Boltz-2 trunk pooling, all-pair pooling, re-pooling, residual-head, and ensemble bakeoff scripts. The most recent work is around `37_trunk_fast_inventory.py` through `40_trunk_residual_head.py`. |
| `multitask_aux/` | Recent diagnostic axis | Multitask auxiliary target selection and TabPFN follow-up. Keep until the assay-shape/dose-response latent direction is fully closed. |
| `eda_cv_prep/` | Historical but useful | Split, OOF/LB gap, 3D feature bakeoff, and pruning diagnostics. Keep as provenance for split and 3D-feature decisions. |
| `eda_redo/` | Historical but useful | April EDA rebuild and CHeMBL lookup scripts. Keep while issue #100 is the research log. |
| `archive/` | Legacy scripts | Early baselines and superseded ensemble scripts. Safe place for old top-level scripts that are no longer entry points. |

## Closed Or Mostly Historical Axes

These directories/scripts should generally stay out of the main workflow unless
the axis is explicitly reopened:

| Path | Reason to keep | Cleanup stance |
|---|---|---|
| `track1_activity/scripts/unimol/` | Documents Uni-Mol v2 null result and feature framework from PR #114/#160. | Keep as a self-contained closed axis. Do not delete unless the corresponding docs are migrated. |
| `track1_activity/scripts/clamp/` | Documents CLAMP raw encoder null result from PR #159. | Keep as a small closed axis. |
| `track1_activity/scripts/gator/` | Documents GatorAffinity Phase A/B attempts and failure mode. | Keep as a closed protein-ligand axis; useful negative control for future Boltz-like ideas. |
| `track1_activity/scripts/drugclip/` | External-data DrugCLIP feature extraction and scoring references. | Keep but avoid using in no-external-data submissions. |
| `track1_activity/scripts/archive/` | Preserves old baselines and superseded ensemble implementations. | Prefer moving obsolete top-level scripts here instead of deleting them. |

## Cleanup Candidates

Start with documentation-only or move-only PRs:

1. Move old top-level EDA scripts into `track1_activity/scripts/archive/eda/`
   after confirming they are not referenced by current docs.
2. Move one-off leaderboard proxy and region diagnostic scripts into
   `track1_activity/analysis/` if they are analysis artifacts rather than model
   entry points.
3. Add short README files to closed-axis directories (`unimol/`, `gator/`,
   `clamp/`, `drugclip/`) with the final result and relevant PR/issue links.
4. Only after the README pass, consider deleting scripts that duplicate an
   archived copy byte-for-byte or are impossible to run because their dependency
   source is gone.

## Do Not Delete Without A Separate Review

- `track1_activity/scripts/run_train.py`
- `track1_activity/scripts/run_ensemble.py`
- `track1_activity/scripts/run_ensemble_calibrate.py`
- `track1_activity/scripts/run_ensemble_calibrate_importance.py`
- `track1_activity/scripts/scheduled_submit.sh`
- `track1_activity/scripts/boltz_affhead/`
- `track1_activity/boltz2/`
- `track1_activity/scripts/api.py` and `track1_activity/scripts/submit.py`
  when present locally, because they are ignored and may contain private state.

# Plan: Pseudo-labeling from single_concentration data

**Parent issue:** #35
**Analysis:** `docs/auxiliary_data_analysis.md`
**Branch:** `feature/pseudo-labeling`
**Baseline:** `lgbm_mordred_umap` (OOF RAE=0.5818)

## Objective

Use the strong correlation (r=0.724) between `single_concentration.log2_fc_estimate` at 8.25e-6 M and `train_activity.pec50` to:

1. Learn a regressor `f: single-conc features -> pEC50`
2. Apply it to the ~8,483 compounds that have single-conc data but no train pEC50
3. Augment the training set with these pseudo-labeled compounds (lower sample weight)
4. Expect ~2-3× effective training set size

## Success criteria

- Mapping CV R² ≥ 0.45 on the train ∩ single-conc overlap
- Augmented LGBM model matches or improves baseline OOF RAE 0.5818
- Best pseudo-weight ≥ one of {0.1, 0.3, 0.5, 1.0} yields ≥ 0.005 OOF RAE improvement
- No leakage: pseudo-labeled compounds are never used as CV validation samples

## File structure

```
track1_activity/
  src/
    pseudo_labels.py            # NEW — pseudo label loading utilities
  scripts/
    build_pseudo_labels.py      # NEW — Task A: fit mapping, generate labels
    run_train_pseudo_sweep.py   # NEW — Task C: sweep + record
data/
  pseudo_labels.parquet         # NEW — generated artifact (gitignored)
docs/plans/
  pseudo-labeling-plan.md       # this file
```

## Task A: Build pseudo labels (`build_pseudo_labels.py`)

**Input**: DB connection

**Steps**:
1. Load `train_activity` joined with `single_concentration` (2,374 compounds with both 8.25e-6 and pEC50 overlap is the floor; use all overlap)
2. For each compound, compute per-compound features from single_concentration:
   - `log2fc_8.25e-6` (the main signal)
   - `log2fc_3.30e-5` (may be NaN for some compounds)
   - `log2fc_stderr_8.25e-6`
   - `cohens_d_8.25e-6`
   - `p_value_8.25e-6`
   - `n_concs` (number of concentrations measured)
3. Merge with `train_activity.pec50` (target)
4. Handle missing values: impute `log2fc_3.30e-5` with median, keep other NaNs as feature (LGBM supports NaN natively)
5. Train LightGBM regressor with 5-fold KFold CV (random shuffle, seed=42)
6. Report RMSE, MAE, R² per fold and OOF
7. Refit on full overlap
8. Apply to compounds that have single_conc data but NOT in train_activity (~8,483 compounds)
9. Compute confidence: `1 / (1 + log2fc_stderr_8.25e-6)` (clip to [0.1, 1.0])
10. Save result to `data/pseudo_labels.parquet`:
    - Columns: `compound_id`, `pseudo_pec50`, `confidence`, `n_concs`
11. Print summary: count, pseudo_pec50 distribution, mean confidence

**CLI**:
```
pixi run python track1_activity/scripts/build_pseudo_labels.py \
    [--out data/pseudo_labels.parquet]
```

**Tests**:
- Script runs end-to-end and produces parquet file
- OOF R² ≥ 0.45 (hard gate)
- Output row count > 5,000 (sanity)
- No NaN in `pseudo_pec50`

**Files created**:
- `track1_activity/scripts/build_pseudo_labels.py`
- `data/pseudo_labels.parquet` (gitignored via existing `data/` rule)

## Task B: Pseudo-label-aware training (`src/pseudo_labels.py` + `run_train.py` patch)

**Objective**: Extend the LGBM training pipeline to support pseudo-labels, with fold-safe augmentation (pseudo samples only appear in training portion of each fold, never in validation).

**Steps**:

1. Create `track1_activity/src/pseudo_labels.py` with:
   - `load_pseudo_labels(path: Path) -> pd.DataFrame` — reads parquet
   - `build_pseudo_feature_matrix(feature_name: str, pseudo_df: pd.DataFrame) -> np.ndarray` — builds same feature matrix used for train (Mordred, fingerprints, embeddings) by calling the existing loader internals for the given compound_ids. This must be feature-agnostic.
   - Helper: `augment_fold(X_tr, y_tr, tr_idx, pseudo_X, pseudo_y, pseudo_w, base_weight=1.0) -> (X_aug, y_aug, w_aug)` — concatenates real train fold with pseudo rows, returns sample weights

2. Modify `track1_activity/scripts/run_train.py`:
   - Add flags `--pseudo PATH` and `--pseudo-weight FLOAT` (default None = disabled)
   - Inside `run()` after loading features, also load pseudo feature matrix if `--pseudo` is set
   - In the CV loop, augment each fold's training portion (never validation)
   - Pass `sample_weight` to LightGBM via `lgb.Dataset(X_tr, label=y_tr, weight=w_tr)`
   - Update `exp_name` to include `_pseudo{w}` suffix when pseudo is used
   - Store pseudo metadata in experiment notes

3. Tests:
   - Dry-run `--pseudo data/pseudo_labels.parquet --pseudo-weight 0.3 --feature mordred --model lgbm --split umap` completes and produces OOF predictions
   - CV fold shapes: validation count unchanged (only real train), training count = real_fold + pseudo_count
   - Experiment recorded in DB

**Constraints**:
- Fold-safety is CRITICAL: a pseudo compound must NEVER end up in a validation fold. Since pseudo compounds have compound_ids disjoint from train_activity, this is guaranteed at the compound_id level, but make it explicit in code with an assertion.
- Pseudo-label feature computation must handle Mordred loading; this is the main feature for the baseline. Fingerprints are nice-to-have for this task (stretch if time allows, otherwise Mordred-only is enough for the sweep).
- Do not support pseudo-labels for feature types that require SMILES-only recomputation if the pseudo compound_ids aren't already precomputed in DB tables. Start with Mordred (always in `compound_mordred`). Error clearly if not supported.

**Files changed**:
- `track1_activity/src/pseudo_labels.py` (new)
- `track1_activity/scripts/run_train.py` (modified)

## Task C: Sweep and record (`run_train_pseudo_sweep.py`)

**Objective**: Run baseline + 4 pseudo-weight variants and report which improves the most.

**Steps**:

1. Create `track1_activity/scripts/run_train_pseudo_sweep.py`:
   - Hardcode best baseline config: `model=lgbm, feature=mordred, split=umap, trials=0` (default params to keep it fast)
   - Run sequence: baseline (no pseudo), then weights {0.1, 0.3, 0.5, 1.0}
   - Each run: call `run_train.py` subprocess or import `run()` directly
   - Collect OOF RAE from each
   - Print comparison table sorted by RAE
   - Save summary to `track1_activity/submissions/pseudo_sweep_summary.csv`
2. Run it end-to-end

**CLI**:
```
pixi run python track1_activity/scripts/run_train_pseudo_sweep.py
```

**Success criteria**:
- All 5 runs complete and record to DB
- Summary table printed
- At least one pseudo-weight matches or improves baseline RAE 0.5818

**Files changed**:
- `track1_activity/scripts/run_train_pseudo_sweep.py` (new)
- DB entries in `experiments`, `experiment_cv_results`, `experiment_oof_predictions`

## Risks

- **Mapping quality**: if CV R² < 0.45, pseudo labels are too noisy to help. Abort Task C and report.
- **Distribution shift**: pseudo-labeled compounds may have a different chemotype distribution than train, especially the ones outside train. Watch for OOF degradation at high pseudo weights.
- **UMAP split stability**: UMAP clustering uses Morgan FPs of train compounds only — unchanged. Pseudo compounds are added only to training portion after splitting.
- **CV leakage**: double-enforced — pseudo compound_ids are disjoint from train (verified in Task A), AND fold augmentation happens only on the training indices.
- **Time budget**: sweeping 5 runs × trials=0 × UMAP split (5 folds) ≈ 5-10 min total if default params. Acceptable.

## Out of scope (do NOT do)

- Multi-task learning with counter_assay (separate issue #36)
- Label denoising via triangulation (separate issue #37)
- Pseudo-labels for feature types beyond Mordred in Task B (keep scope tight)
- Ensemble integration (follow-up if sweep succeeds)
- Hyperparameter re-tuning (use default params for the sweep)

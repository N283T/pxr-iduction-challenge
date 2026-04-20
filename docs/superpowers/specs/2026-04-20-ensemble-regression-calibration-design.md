# Design: Post-hoc Regression Calibration for ens_caruana_bag20

- **Status**: Approved (2026-04-20 evening)
- **Author**: Claude Code session, brainstormed with user
- **Related**: Follows PR #98 MoLFormer-c3 (merged, LB unchanged at rank 10 despite OOF improvement). LB analysis revealed rank metrics (Spearman rank 4, Kendall rank 4, R² rank 5) are top-tier while MAE (rank 10) lags by ~0.032 — classic "good ordering, wrong scale" signature suggesting systematic prediction bias amenable to post-hoc calibration.

## Goal

Apply post-hoc regression calibration to the current `ens_caruana_bag20` ensemble output (10-pool, OOF MAE 0.4309). Fit two calibrators (Linear and Isotonic) on OOF predictions vs. true pEC50, apply to test predictions, produce two new LB submission CSVs. Evaluate calibration gain via nested CV to guard against calibration-induced overfitting. This is an experimental probe (user-flagged "risk OK") before investing in more sophisticated methods (conformal prediction, quantile calibration) pending ChatGPT deep-research findings.

## Background

Current LB position after PR #98 merge:

| Metric | N283T value | Rank | Gap to top-3 |
|---|---|---|---|
| MAE (primary) | 0.4423 | 10 | +0.032 |
| RAE | 0.5556 | 10 | +0.041 |
| R² | 0.6253 | 5 | −0.013 |
| Spearman | 0.8412 | 4 | −0.009 |
| Kendall | 0.6463 | 4 | −0.017 |

Ordering metrics are within 1–2% of top tier; absolute metrics lag ~8%. A monotonic transform (preserves ordering, corrects scale) is the natural intervention.

## Non-goals (deferred)

- Conformal prediction / CV+ / quantile-regression calibration — pending ChatGPT deep-research survey (user running in parallel)
- Calibrating each pool member before ensembling — adds complexity; ensemble-output calibration is simpler and captures the aggregate bias
- Integrating calibrated predictions back into `ENSEMBLE_MODELS` as new members — calibrated output is monotonically transformed, so caruana would trivially pick the uncalibrated original; treat calibration as a post-ensemble head rather than a new member
- Tuning isotonic k-knots via Optuna — fixed defaults first; revisit if nested CV shows overfitting

## Architecture

### Single new script

`track1_activity/scripts/run_ensemble_calibrate.py`

Reads ensemble artifacts from DB + filesystem, fits calibrators on OOF, applies to test predictions, writes two new submission CSVs, records two experiment rows in DB.

### Input artifacts

- **Per-member OOF predictions** from `experiment_oof_predictions` (10 active `ENSEMBLE_MODELS`)
- **Caruana weights** from the latest `ens_caruana_bag20` row's `hyperparameters` JSONB (key: `weights`)
- **True pEC50**: from `train_activity` via existing `data.load_train_smiles_target()`
- **Raw test predictions**: the latest `track1_activity/submissions/ens_caruana_bag20.csv`

### Pipeline

1. Load per-member OOF matrix `X_oof` shape (4140, 10) and member names from DB
2. Load weights `w` from `ens_caruana_bag20` hyperparameters JSONB; align by name
3. Reconstruct ensemble OOF: `y_pred_oof = X_oof @ w`
4. **Sanity check**: `MAE(y_true, y_pred_oof)` should match id=644's recorded value (0.4309 ± 0.001 for float rounding)
5. **Nested CV** (5-fold UMAP, seed=42, matches ENSEMBLE_MODELS training CV):
   - For each outer fold:
     - Fit `LinearRegression().fit(y_pred_oof[tr], y_true[tr])` — 2 params (slope, intercept)
     - Fit `IsotonicRegression(out_of_bounds="clip").fit(y_pred_oof[tr], y_true[tr])` — monotonic spline with clipping at train bounds
     - Apply each calibrator to `y_pred_oof[va]`, measure MAE/RAE/Spearman/Kendall
   - Report honest calibrated metrics (mean ± std across folds)
6. **Final fit** on FULL OOF (no held-out) for both calibrators; apply to raw test predictions
7. Write 2 submission CSVs:
   - `track1_activity/submissions/ens_caruana_bag20_calibrated_linear.csv`
   - `track1_activity/submissions/ens_caruana_bag20_calibrated_isotonic.csv`
8. Record 2 experiments via `record_experiment(... on_conflict_replace=True)`:
   - Names: `ens_caruana_bag20_calibrated_linear`, `ens_caruana_bag20_calibrated_isotonic`
   - `model_type="ensemble_calibrated"`, `feature_set="caruana_bag20_output"`
   - `hyperparameters`: `{method: ..., fitted_params: ...}` (linear: {slope, intercept}; isotonic: {n_knots, y_min, y_max})
   - `fold_metrics`: 5-fold nested CV results per method
   - Save calibrated OOF predictions via `save_oof_predictions()` for audit

### Calibrator details

**Linear** (`sklearn.linear_model.LinearRegression`):
- Simple 2-parameter fit: `y_cal = slope * y_pred + intercept`
- Expected to capture: uniform shift + uniform scale
- Risk: tiny model, essentially no overfit risk even on small folds

**Isotonic** (`sklearn.isotonic.IsotonicRegression(out_of_bounds="clip")`):
- Non-parametric monotonic regression, up to 4140 step segments
- Expected to capture: non-linear monotonic warping (e.g., predictions compressed in middle range, stretched at extremes)
- Risk: with 4140 points of possibly flat OOF values in some regions, the fit may memorize noise. `clip` prevents runaway on out-of-distribution test predictions.

Both are monotonic by construction: **Spearman and Kendall are preserved exactly**; only absolute metrics (MAE/R²) can change.

## Data flow

```
experiment_oof_predictions (10 models × 4140 rows)
                       ↓
              X_oof (4140, 10) matrix
                       ↓ dot with weights w from ens_caruana_bag20.hyperparameters
              y_pred_oof (4140,) raw ensemble OOF
                       ↓ nested 5-fold UMAP CV
         Calibrated OOF: linear, isotonic (4140,) + honest metrics
                       ↓ fit on full OOF, apply to test
         test CSV (linear), test CSV (isotonic)
                       ↓
             DB records (2 experiments) + submission CSVs
```

## Acceptance criteria

1. **Sanity**: reconstructed raw ensemble OOF MAE matches DB's recorded value for `ens_caruana_bag20` (id=644) within 1e-3
2. **Nested CV MAE improvement**: at least one of {linear, isotonic} shows nested-CV MAE ≤ raw OOF MAE (0.4309). Equality or minor regression (Δ ≤ +0.002) is tolerable if the other method improves significantly.
3. **Ordering preserved**: Spearman change |Δ| < 0.005 for both methods (mathematically guaranteed for monotonic calibrators; empirical check only catches bugs)
4. **2 submission CSVs written** with correct headers (SMILES, Molecule Name, pEC50) and 513 rows
5. **2 DB experiment rows recorded** with calibrated OOF predictions (4140 each)
6. `ruff format` and `ruff check` clean on the new script

Failure handling:
- If nested CV shows both methods regress: report to user; skip LB submission. The run still produces the DB artifacts for analysis.
- If raw OOF sanity check fails: weights or OOF matrix is misaligned; debug before proceeding.

## Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Isotonic overfits on 4140-point OOF | Medium | Nested CV catches this; fallback to linear-only |
| Calibrator fits train-space bias but LB test-space is different (covariate shift) | Medium | Inherent risk, acknowledged as "experimental" by user. Nested CV is the best pre-LB estimate. If LB submission regresses, drop calibrated variants. |
| Weights in DB don't match the last ensemble run (e.g., multiple caruana_bag20 rows) | Low | Query `ORDER BY id DESC LIMIT 1` to get the latest; print the weights before use for sanity |
| Member name alignment: OOF matrix columns vs. weight dict keys must match | Medium | Load weights as dict `{name: weight}`; iterate names from `ENSEMBLE_MODELS` in run_ensemble.py to define column order; assert every name has a weight |

## ETA

- Implementation: ~1 h
- Smoke + run: ~5 min (CPU-only workload: linear fit + isotonic fit on 4140 points is seconds)
- LB submission: 1 cooldown slot (4h), user decides which CSV to submit after reviewing nested CV results

## Future work (next PRs)

- Review ChatGPT deep-research findings on regression calibration
- Conformal prediction / CV+ for interval + point correction
- Quantile-regression calibration (quantile-matching to train distribution)
- Per-cluster calibration (different bias per chemical-space region)
- Calibrating each pool member before ensembling (member-level bias correction)

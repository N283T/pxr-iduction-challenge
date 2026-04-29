# Anchor Residual Model

Date: 2026-04-29
Status: Approved (brainstorming compressed per user direction)
Owner: N283T

## Background

Track 1 LB rank 2 (N283T 0.4075 / Sp 0.8470). Codex consult 2026-04-29
proposed three sequential analog-prior approaches:

1. potent-46 proximity-gated calibrator → **null** (PR #151, near affine ≈ global affine)
2. **anchor residual model** ← this spec
3. rank-preserving MAE correction (deferred)

Codex's brief: "train 上で「base_pred 誤差」を目的変数にして、`nearest potent
anchor` 関連特徴だけで residual を学習。強い base model に小さく足す設計なら
壊しにくい。"

## Goal

Add a small residual correction on top of the existing
`ens_caruana_bag20_calibrated_importance` predictions. The residual model
uses anchor (potent-46) features to predict the local error pattern of the
base model, applies a damped correction `final = base + α × residual_hat`.

The proximity calibrator was an affine on `base` only; the residual model
adds **additional features** (NN-Tanimoto, anchor pEC50, gap), giving it
freedom to capture non-linear corrections that an affine cannot. This is
the structural difference that motivates retrying after the proximity null.

Success target: LB MAE Δ ≤ -0.0008 vs id=43 (close gap to sia 0.4066),
Sp Δ ≥ -0.005.

## Definitions

- **base prediction**: `ens_caruana_bag20_calibrated_importance.csv` for
  test, and the corresponding OOF for train (caruana_bag20 OOF reconstructed
  from member predictions × stored weights, then importance affine applied).
- **potent-46**: train compounds with `pec50 ≥ 6 AND pec50 - counter_pec50 ≥ 1.5`
  (definition from `splits.py::analog_aware_split_indices`, 46 compounds expected).
- **nearest potent anchor** for compound q: `argmax_{p ∈ potent-46, p ≠ q} Tanimoto(FP(q), FP(p))`
  (Morgan r=2, 2048 bit; train rows in potent-46 self-exclude).
- **damping α**: scalar multiplier on residual prediction, fixed at 0.5 for v1.

## Anchor features (4 columns per compound)

| name | meaning | range |
|---|---|---|
| `nn_tanimoto` | NN Tanimoto to potent-46 (self-excluded) | [0, 1] |
| `anchor_pec50` | pEC50 of the nearest potent anchor | ~6.0 to ~9.0 |
| `base_pred` | calibrated base prediction (importance affine applied) | ~3 to ~9 |
| `pred_minus_anchor` | `base_pred - anchor_pec50` | typically negative (base lower than potent) |

These 4 features are enough for v1. No descriptor / FP / Mordred features
to keep the residual model simple and avoid duplicating the base model's job.

## Method

### Step 1 — Reconstruct base predictions
- OOF: load caruana_bag20 OOF (member preds × weights from
  `experiments.hyperparameters`), apply importance affine (re-fit inline,
  same recipe as `run_ensemble_calibrate_importance.py`).
- Test: read `ens_caruana_bag20_calibrated_importance.csv`. If absent,
  re-derive by applying the same importance affine to
  `ens_caruana_bag20.csv`.

### Step 2 — Compute anchor features
For each train + test compound:
- Find nearest potent-46 anchor (with self-exclude on train).
- Record `nn_tanimoto`, `anchor_pec50`, `base_pred`, `pred_minus_anchor`.

### Step 3 — OOF residual prediction (5-fold UMAP CV)
- Use canonical UMAP split (Morgan + Jaccard + KMeans k=50, seed 42).
- For each fold f:
  - Train residual model on (k-1) folds: target = `y - base_pred`, features = 4 anchor cols.
  - Predict residual on held-out fold f.
- Concatenate: full-train OOF `residual_hat`.

### Step 4 — Apply damped correction
- `corrected_oof = base_pred_oof + α × residual_hat`  (α=0.5 fixed)
- Compare MAE/Sp:
  - raw OOF (before importance affine)
  - base OOF (importance-calibrated, current production baseline)
  - corrected OOF

### Step 5 — Test prediction
- Fit residual model on **ALL** train (no fold split) — standard practice
  for final test prediction after CV-validated hyperparameters.
- Predict residual for test using the 4 anchor features.
- `test_corrected = base_test + α × residual_hat_test`

### Step 6 — Write submission
- `track1_activity/submissions/ens_caruana_bag20_anchor_residual.csv`

## Residual model

LightGBM regressor with conservative hyperparameters:

```python
LGBMRegressor(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    min_child_samples=30,
    reg_lambda=1.0,
    reg_alpha=0.5,
    subsample=0.8,
    colsample_bytree=0.9,
    random_state=42,
    verbosity=-1,
    n_jobs=-1,
)
```

Rationale: residuals are noise-heavy (caruana_bag20 cross-run variance ±0.003);
small tree depth + min_child_samples=30 prevents memorizing per-fold noise.
Linear could work but residual signal may be non-monotone in `nn_tanimoto`
(correction may kick in only above some threshold).

## Validation

### OOF gate (must pass before LB submit)

Compare against base OOF (importance-calibrated, current production):
- `corrected OOF MAE ≤ base OOF MAE` (no regress; same precision floor as proximity v2)
- `corrected OOF Spearman ≥ base OOF Spearman - 0.005`
- `|residual_hat|` stats sanity:
  - `max(|residual_hat|) ≤ 0.5` pEC50 unit (warn if exceeded; abandon if > 1.0)
  - `q99(|residual_hat|) ≤ 0.3`
  - Implies the residual model is making small corrections, not "replacing" base

If gate fails: report and do not LB-submit.

### LB A/B (mandatory regardless of OOF)

Per `feedback_oof_lb_reverse_amplification`. Submit during a free 4h
cooldown window. Compare against id=43 (current best, 0.4075 / Sp 0.8470).

Decision:
- LB MAE Δ ≤ -0.0005 AND Sp Δ ≥ -0.005: keep
- LB MAE Δ in [-0.0005, +0.0005]: hold (bag noise)
- LB MAE Δ > +0.0005: revert (do not adopt as production)

## Non-goals

- **NOT touching the 9-pool composition** — residual on top of existing.
- **NOT tuning α via OOF for v1** — fixed 0.5. v2 may sweep [0.0, 0.25, 0.5, 0.75, 1.0].
- **NOT using k-NN aggregate features** (k>1) — single NN for v1, k-NN deferred.
- **NOT including descriptor / FP / Mordred features** — anchor features only.
- **NOT changing CV split** — UMAP canonical.
- **NOT a per-stratum design** (proximity calibrator pivot found that asymmetric
  hard-binning collapses to global; residual model uses continuous features
  directly so no binning needed).

## Risks

1. **Self-match leak** in NN-Tanimoto computation — mitigated via explicit
   self-exclude (proven correct in proximity work).
2. **Label leakage via `anchor_pec50`** — using a neighboring train compound's
   pEC50 as a feature is k-NN style, not a leak per se. The neighbor is
   ALWAYS in train (potent-46 ⊂ train), so for OOF computation on train we
   self-exclude. For test, all potent-46 are valid anchors.
3. **OOF → LB reverse amplification** — calibrator/correction family
   changes have documented sign-flips. LB A/B mandatory.
4. **Residual model overfit on noise** — guarded by small LGBM hyperparams +
   OOF validation + |residual_hat| sanity.
5. **Residual model fitted on importance-calibrated base** — the residual is
   defined relative to the calibrated baseline. If we later switch the base
   calibrator, this residual model becomes stale. Acceptable for v1 (one-shot
   experiment); document for future re-fits.

## Deliverables

- `track1_activity/scripts/run_anchor_residual.py` (new, ~350 lines)
- `track1_activity/submissions/ens_caruana_bag20_anchor_residual.csv` (gitignored)
- Console report:
  - potent-46 size
  - 4-feature OOF + test stats (mean / q25 / q75 / max for each)
  - Per-fold residual model OOF MAE (sanity)
  - `|residual_hat|` distribution (mean / q99 / max)
  - base OOF MAE / Sp, corrected OOF MAE / Sp, deltas
  - explicit gate pass/fail summary

## Open questions (revisit if v1 succeeds)

- α sweep with nested CV
- k-NN aggregate features (top-3 mean Tanimoto, top-3 mean anchor pEC50)
- gating: apply correction only when nn_tanimoto ≥ some threshold (continuous gate)
- combine with proposal #3 (rank-preserving MAE correction)

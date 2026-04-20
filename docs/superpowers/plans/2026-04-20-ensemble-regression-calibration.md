# Post-hoc Regression Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fit Linear + Isotonic post-hoc calibrators on `ens_caruana_bag20`'s 4140-point OOF predictions vs. true pEC50, apply to the 513 test predictions, emit two new LB submission CSVs + two DB experiment rows. Evaluate calibration gain via 5-fold UMAP nested CV to guard against calibration-induced overfitting.

**Architecture:** Single-script pipeline (`run_ensemble_calibrate.py`) that (1) rebuilds the ensemble OOF from per-member OOF + caruana weights, (2) runs nested CV with sklearn `LinearRegression` and `IsotonicRegression`, (3) refits on full OOF and applies to test predictions, (4) writes 2 CSVs + 2 DB experiments. No new framework, only sklearn calibrators.

**Tech Stack:** Python 3.12, sklearn 1.8 (LinearRegression, IsotonicRegression), numpy, pandas, psycopg2 (all already installed).

**Spec:** `docs/superpowers/specs/2026-04-20-ensemble-regression-calibration-design.md`

**Branch:** `feature/ensemble-regression-calibration` (already created)

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `track1_activity/scripts/run_ensemble_calibrate.py` | Create | Complete pipeline: load OOF + weights + test CSV, nested CV, fit, apply, write outputs |

Only one new file. No modifications to existing files.

---

## Task 1: CLI script

**Files:**
- Create: `track1_activity/scripts/run_ensemble_calibrate.py`

- [ ] **Step 1.1: Write the script**

Create `track1_activity/scripts/run_ensemble_calibrate.py` with EXACTLY this content:

```python
"""Post-hoc regression calibration of ens_caruana_bag20 output.

Fits Linear + Isotonic calibrators on the ensemble's 4140-point OOF
predictions vs. true pEC50. Evaluates each calibrator via 5-fold UMAP
nested CV (honest estimate, guards against calibration overfitting),
then refits on full OOF and applies to raw test predictions to produce
two calibrated submission CSVs.

Inputs:
- Per-member OOF predictions from experiment_oof_predictions (10 models)
- Caruana weights from the latest ens_caruana_bag20 row's hyperparameters
- True pEC50 from train_activity via data.load_train_smiles_target
- Raw test predictions: track1_activity/submissions/ens_caruana_bag20.csv

Outputs:
- track1_activity/submissions/ens_caruana_bag20_calibrated_linear.csv
- track1_activity/submissions/ens_caruana_bag20_calibrated_isotonic.csv
- Two experiments in DB: ens_caruana_bag20_calibrated_{linear,isotonic}

Usage:
    pixi run python track1_activity/scripts/run_ensemble_calibrate.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))

from data import DB_PARAMS, load_train_smiles_target, load_test_smiles  # noqa: E402
from evaluate import (  # noqa: E402
    compute_metrics,
    print_metrics,
    record_experiment,
    save_oof_predictions,
)
from splits import umap_split_indices  # noqa: E402

SUBMISSION_DIR = REPO_ROOT.joinpath("track1_activity", "submissions")


def load_latest_caruana_weights() -> dict[str, float]:
    """Fetch weights dict from the most recent ens_caruana_bag20 row."""
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    cur.execute(
        "SELECT hyperparameters FROM experiments "
        "WHERE name = 'ens_caruana_bag20' ORDER BY id DESC LIMIT 1"
    )
    row = cur.fetchone()
    cur.close()
    conn.close()
    if row is None:
        raise RuntimeError("No ens_caruana_bag20 row in DB. Run run_ensemble.py first.")
    hp = row[0] if isinstance(row[0], dict) else json.loads(row[0])
    return hp["weights"]


def load_member_oof_matrix(
    member_names: list[str], n_train: int
) -> np.ndarray:
    """Build (n_train, n_members) OOF matrix, columns aligned to member_names.

    Each column is the OOF predictions for the named experiment, read from
    experiment_oof_predictions ORDER BY train_idx. Raises if any member has
    missing OOF rows.
    """
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    cols: list[np.ndarray] = []
    for name in member_names:
        cur.execute(
            "SELECT oof_prediction FROM experiment_oof_predictions "
            "WHERE experiment_id = (SELECT id FROM experiments WHERE name = %s) "
            "ORDER BY train_idx",
            (name,),
        )
        rows = cur.fetchall()
        if len(rows) != n_train:
            cur.close()
            conn.close()
            raise RuntimeError(
                f"Member '{name}' has {len(rows)} OOF rows, expected {n_train}"
            )
        cols.append(np.array([r[0] for r in rows], dtype=np.float64))
    cur.close()
    conn.close()
    return np.stack(cols, axis=1)


def reconstruct_ensemble_oof(
    X_oof: np.ndarray, weights: dict[str, float], member_names: list[str]
) -> np.ndarray:
    """Compute y_pred_oof = sum(weights[name] * X_oof[:, i]) for i, name."""
    w = np.array([weights[n] for n in member_names], dtype=np.float64)
    return X_oof @ w


def load_raw_test_predictions() -> pd.DataFrame:
    """Read the latest ens_caruana_bag20.csv submission. Returns DataFrame with
    SMILES, Molecule Name, pEC50 columns (513 rows)."""
    path = SUBMISSION_DIR.joinpath("ens_caruana_bag20.csv")
    if not path.exists():
        raise RuntimeError(f"Missing {path}. Run run_ensemble.py to regenerate.")
    df = pd.read_csv(path)
    if len(df) != 513:
        raise RuntimeError(f"Expected 513 test rows, got {len(df)}")
    return df


def fit_calibrator(method: str, y_pred: np.ndarray, y_true: np.ndarray):
    """Fit a calibrator and return an object with a .predict(x) method."""
    if method == "linear":
        model = LinearRegression()
        model.fit(y_pred.reshape(-1, 1), y_true)
        return model
    if method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip")
        model.fit(y_pred, y_true)
        return model
    raise ValueError(f"Unknown method: {method}")


def apply_calibrator(method: str, model, y_pred: np.ndarray) -> np.ndarray:
    """Apply a fitted calibrator to y_pred, returning the calibrated values."""
    if method == "linear":
        return model.predict(y_pred.reshape(-1, 1))
    if method == "isotonic":
        return model.predict(y_pred)
    raise ValueError(f"Unknown method: {method}")


def nested_cv_evaluate(
    y_pred_oof: np.ndarray,
    y_true: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    method: str,
) -> tuple[list[dict], np.ndarray]:
    """5-fold nested CV: fit calibrator on 4 folds, evaluate on 1.

    Returns:
      - list of 5 metric dicts (one per fold)
      - calibrated OOF predictions (4140,) stitched from per-fold held-out predictions
    """
    calibrated_oof = np.zeros_like(y_pred_oof)
    fold_metrics = []
    for fold, (tr_idx, va_idx) in enumerate(splits):
        model = fit_calibrator(method, y_pred_oof[tr_idx], y_true[tr_idx])
        va_pred = apply_calibrator(method, model, y_pred_oof[va_idx])
        calibrated_oof[va_idx] = va_pred
        m = compute_metrics(y_true[va_idx], va_pred)
        fold_metrics.append(m)
        print_metrics(m, label=f"{method} Fold {fold}")
    return fold_metrics, calibrated_oof


def main() -> None:
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading ensemble artifacts...")
    train_df = load_train_smiles_target()
    test_df = load_test_smiles()
    y_true = train_df["pec50"].to_numpy(dtype=np.float64)
    n_train = len(y_true)
    print(f"  train rows: {n_train}, test rows: {len(test_df)}")

    weights = load_latest_caruana_weights()
    member_names = list(weights.keys())
    print(f"  caruana members: {len(member_names)}")
    for name in member_names:
        print(f"    {name:<50} w={weights[name]:.4f}")

    X_oof = load_member_oof_matrix(member_names, n_train)
    y_pred_oof = reconstruct_ensemble_oof(X_oof, weights, member_names)

    raw_oof_metrics = compute_metrics(y_true, y_pred_oof)
    print("\nReconstructed raw ensemble OOF:")
    print_metrics(raw_oof_metrics, label="RAW")

    # Sanity: raw OOF MAE should match the ens_caruana_bag20 DB value closely
    # (will differ very slightly due to float accumulation order).
    if not 0.40 <= raw_oof_metrics["MAE"] <= 0.45:
        raise RuntimeError(
            f"Raw OOF MAE {raw_oof_metrics['MAE']:.4f} outside expected "
            f"range [0.40, 0.45]. Weights/OOF misalignment?"
        )

    # Build 5-fold UMAP splits matching ENSEMBLE_MODELS training CV
    print("\nBuilding 5-fold UMAP splits (seed=42)...")
    splits = umap_split_indices(train_df["smiles"].tolist(), n_splits=5, seed=42)

    raw_test = load_raw_test_predictions()
    raw_test_pred = raw_test["pEC50"].to_numpy(dtype=np.float64)

    print("\n" + "=" * 60)
    print("  NESTED CV EVALUATION (5-fold UMAP)")
    print("=" * 60)

    for method in ("linear", "isotonic"):
        print(f"\n--- {method} ---")
        fold_metrics, calibrated_oof = nested_cv_evaluate(
            y_pred_oof, y_true, splits, method
        )
        nested_cv_metrics = compute_metrics(y_true, calibrated_oof)
        print(f"\n  Overall nested-CV calibrated OOF ({method}):")
        print_metrics(nested_cv_metrics, label=f"{method} NESTED")

        # Sanity: Spearman should be essentially unchanged (monotonic calib)
        dspear = abs(nested_cv_metrics["Spearman_R"] - raw_oof_metrics["Spearman_R"])
        if dspear >= 0.005:
            print(
                f"  WARNING: Spearman changed by {dspear:.4f} "
                f"(expected < 0.005 for monotonic calibrator)"
            )

        # Refit on full OOF and apply to raw test predictions
        full_model = fit_calibrator(method, y_pred_oof, y_true)
        test_pred_cal = apply_calibrator(method, full_model, raw_test_pred)

        # Diagnostics on the fitted calibrator
        if method == "linear":
            fitted = {
                "slope": float(full_model.coef_[0]),
                "intercept": float(full_model.intercept_),
            }
            print(
                f"  Linear fit: slope={fitted['slope']:.4f}, "
                f"intercept={fitted['intercept']:.4f}"
            )
        else:
            fitted = {
                "n_knots": int(len(full_model.X_thresholds_)),
                "y_min": float(full_model.y_min_),
                "y_max": float(full_model.y_max_),
            }
            print(
                f"  Isotonic fit: n_knots={fitted['n_knots']}, "
                f"y_min={fitted['y_min']:.4f}, y_max={fitted['y_max']:.4f}"
            )

        # Write submission CSV
        sub_name = f"ens_caruana_bag20_calibrated_{method}"
        sub_path = SUBMISSION_DIR.joinpath(f"{sub_name}.csv")
        sub_df = pd.DataFrame(
            {
                "SMILES": raw_test["SMILES"],
                "Molecule Name": raw_test["Molecule Name"],
                "pEC50": test_pred_cal,
            }
        )
        sub_df.to_csv(sub_path, index=False)
        print(f"  Wrote submission: {sub_path}")

        # Record experiment with nested-CV metrics as fold_metrics
        exp_id = record_experiment(
            name=sub_name,
            description=(
                f"Post-hoc {method} calibration of ens_caruana_bag20 "
                f"(5-fold UMAP nested CV)"
            ),
            model_type="ensemble_calibrated",
            feature_set="caruana_bag20_output",
            hyperparameters={
                "method": method,
                "fitted_params": fitted,
                "source_ensemble": "ens_caruana_bag20",
                "source_members": member_names,
                "source_weights": weights,
            },
            fold_metrics=fold_metrics,
            submission_path=f"track1_activity/submissions/{sub_name}.csv",
            notes=(
                f"Raw OOF MAE={raw_oof_metrics['MAE']:.4f}, "
                f"calibrated nested-CV MAE={nested_cv_metrics['MAE']:.4f}, "
                f"ΔMAE={nested_cv_metrics['MAE'] - raw_oof_metrics['MAE']:+.4f}, "
                f"{method} calibrator"
            ),
            on_conflict_replace=True,
        )
        save_oof_predictions(exp_id, calibrated_oof)
        print(f"  Recorded experiment id={exp_id}")

    print("\nDone.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 1.2: Lint**

Run: `cd /home/nagaet/pxr-iduction-challenge && pixi run ruff format track1_activity/scripts/run_ensemble_calibrate.py && pixi run ruff check track1_activity/scripts/run_ensemble_calibrate.py`
Expected: clean. (Do NOT run ty — project convention, optional.)

- [ ] **Step 1.3: Run end-to-end**

Run:
```bash
cd /home/nagaet/pxr-iduction-challenge && pixi run db-start 2>&1 | tail -2
LOG="logs/ensemble_calibrate_$(date +%Y%m%d_%H%M).log"
echo "Log: $LOG"
mkdir -p logs
pixi run python track1_activity/scripts/run_ensemble_calibrate.py 2>&1 | tee "$LOG" | tail -60
```

Expected in output:
- `Reconstructed raw ensemble OOF` section with `RAW MAE=0.4309` (±0.001)
- `linear` nested-CV section showing per-fold + overall metrics
- `isotonic` nested-CV section showing per-fold + overall metrics
- `Linear fit: slope=...` and `Isotonic fit: n_knots=...`
- `Wrote submission: .../ens_caruana_bag20_calibrated_linear.csv`
- `Wrote submission: .../ens_caruana_bag20_calibrated_isotonic.csv`
- `Recorded experiment id=<N>` twice

Expected wall clock: under 2 minutes (CPU-only, linear fit + isotonic fit on 4140 points + 5-fold nested).

- [ ] **Step 1.4: Verify the two submission CSVs**

Run:
```bash
cd /home/nagaet/pxr-iduction-challenge
for f in track1_activity/submissions/ens_caruana_bag20_calibrated_{linear,isotonic}.csv; do
  echo "--- $f ---"
  head -2 "$f"
  echo "rows: $(wc -l < "$f")"
  python -c "import pandas as pd; df=pd.read_csv('$f'); print('pEC50 stats:', df['pEC50'].describe().to_dict())"
done
```
Expected: 514 lines (1 header + 513 rows) for each, columns `SMILES,Molecule Name,pEC50`. `pEC50` stats should show non-constant values. Compare to raw:
```bash
python -c "import pandas as pd; print(pd.read_csv('track1_activity/submissions/ens_caruana_bag20.csv')['pEC50'].describe().to_dict())"
```
The linear-calibrated mean should differ from raw by the intercept correction; isotonic may have different tails.

- [ ] **Step 1.5: Verify the DB rows**

Run:
```bash
cd /home/nagaet/pxr-iduction-challenge && pixi run python <<'PY'
import psycopg2
conn = psycopg2.connect(host='/tmp', port=5433, dbname='pxr_challenge')
cur = conn.cursor()
cur.execute("""
SELECT e.id, e.name, e.notes,
  AVG(r.mae) AS mae_nested, AVG(r.spearman_r) AS spear_nested,
  (SELECT COUNT(*) FROM experiment_oof_predictions WHERE experiment_id=e.id) AS n_oof
FROM experiments e JOIN experiment_cv_results r ON r.experiment_id = e.id
WHERE e.name LIKE 'ens_caruana_bag20_calibrated_%'
GROUP BY e.id, e.name ORDER BY e.id DESC LIMIT 2
""")
for row in cur.fetchall():
    print(row)
PY
```
Expected: two rows (linear, isotonic) each with 4140 OOF predictions and nested-CV MAE + Spearman.

- [ ] **Step 1.6: Acceptance check — nested-CV MAE improvement**

From Step 1.3 output, compare `RAW MAE=0.4309` to each method's `NESTED MAE=...`:

- If at least one method shows `MAE <= 0.4309` AND Spearman change < 0.005: PASS
- If both methods show MAE between 0.4309 and 0.4329 (Δ ≤ +0.002): PASS with notes (calibration was roughly neutral, LB submission optional)
- If both methods regress MAE by > 0.002: FAIL. Report to user; skip LB submission. The DB records still provide analysis material.

- [ ] **Step 1.7: Commit**

```bash
cd /home/nagaet/pxr-iduction-challenge
git add track1_activity/scripts/run_ensemble_calibrate.py
git commit -m "feat(calibrate): post-hoc Linear + Isotonic regression calibration for ens_caruana_bag20"
```

---

## Task 2: Push + open draft PR

**Files:** none (git + gh only)

- [ ] **Step 2.1: Push**

Run: `cd /home/nagaet/pxr-iduction-challenge && git push -u origin feature/ensemble-regression-calibration`

- [ ] **Step 2.2: Open draft PR with nested-CV results**

Run:
```bash
cd /home/nagaet/pxr-iduction-challenge && gh pr create --draft --title "feat(calibrate): post-hoc Linear + Isotonic regression calibration" --body "$(cat <<'EOF'
## Summary
Experimental probe for post-hoc regression calibration of `ens_caruana_bag20`. Fits `sklearn.LinearRegression` and `sklearn.IsotonicRegression` on 4140-point OOF predictions vs. true pEC50, applies to test predictions, produces two calibrated submission CSVs.

## Motivation
LB analysis after PR #98 (MoLFormer-c3) revealed "good ordering, wrong scale" signature:
- MAE (primary) rank 10 — gap +0.032 vs top-3
- Spearman rank 4 — gap −0.009 (essentially top-tier)
- Kendall rank 4 — gap −0.017

Monotonic calibration preserves ordering by construction and may correct scale bias.

## Nested CV Results
(Placeholder -- to be filled in from Step 1.3 output before marking ready for review.)

| Method | Nested-CV MAE | Δ vs raw (0.4309) | Spearman |
|---|---|---|---|
| raw | 0.4309 | — | 0.8156 |
| linear | <MAE> | <Δ> | <spearman> |
| isotonic | <MAE> | <Δ> | <spearman> |

## Scope
- 1 new script, no modifications to existing files
- 2 new experiments in DB, 2 new submission CSVs
- No changes to `ENSEMBLE_MODELS` allow-list (calibrated outputs are post-ensemble, not pool members)
- LB submission decision deferred pending user review of nested-CV results

Spec: \`docs/superpowers/specs/2026-04-20-ensemble-regression-calibration-design.md\`
Plan: \`docs/superpowers/plans/2026-04-20-ensemble-regression-calibration.md\`
EOF
)" 2>&1 | tail -3
```
Expected: prints PR URL.

---

## Task 3: Update PR body with actual nested-CV results

**Files:** none (gh only)

- [ ] **Step 3.1: Fill the nested-CV table in the PR body**

Get the actual numbers from Step 1.3 log. Edit the PR body:

```bash
cd /home/nagaet/pxr-iduction-challenge && gh pr edit --body "$(cat <<'EOF'
## Summary
Experimental probe for post-hoc regression calibration of `ens_caruana_bag20`. Fits `sklearn.LinearRegression` and `sklearn.IsotonicRegression` on 4140-point OOF predictions vs. true pEC50, applies to test predictions, produces two calibrated submission CSVs.

## Nested CV Results

| Method | Nested-CV MAE | Δ vs raw | Spearman | Kendall |
|---|---|---|---|---|
| raw | 0.4309 | — | 0.8156 | <K_raw> |
| linear | <L_mae> | <L_dmae> | <L_spear> | <L_kend> |
| isotonic | <I_mae> | <I_dmae> | <I_spear> | <I_kend> |

Linear fit: slope=<S>, intercept=<I>
Isotonic fit: n_knots=<N>

## Acceptance
- [<pass/fail>] At least one method: nested-CV MAE ≤ 0.4309 AND Spearman Δ < 0.005
- [x] Sanity: reconstructed raw OOF MAE 0.4309 matches ens_caruana_bag20 DB record
- [x] 2 submission CSVs written with 513 rows each
- [x] 2 experiments recorded in DB with 4140 OOF predictions each
- [x] ruff format + ruff check clean

Spec: \`docs/superpowers/specs/2026-04-20-ensemble-regression-calibration-design.md\`
Plan: \`docs/superpowers/plans/2026-04-20-ensemble-regression-calibration.md\`
EOF
)"
```

Replace `<L_mae>`, `<L_dmae>`, `<L_spear>`, `<L_kend>`, `<I_mae>`, `<I_dmae>`, `<I_spear>`, `<I_kend>`, `<S>`, `<I>`, `<N>`, `<K_raw>`, `<pass/fail>` with actual values from the Step 1.3 run.

- [ ] **Step 3.2: Mark PR ready**

Run: `cd /home/nagaet/pxr-iduction-challenge && gh pr ready`

- [ ] **Step 3.3: Ask the user for merge approval + LB submission decision**

Present to the user:
- Summary of nested-CV results (which method won, by how much)
- Whether Acceptance 1 passes
- Recommendation: "Shall I merge? For LB, submit linear or isotonic or neither?"

Wait for explicit approval before merging or submitting to LB.

- [ ] **Step 3.4: After approval, merge and clean up**

Run:
```bash
cd /home/nagaet/pxr-iduction-challenge
gh pr merge --squash --delete-branch
git checkout main && git pull
git remote prune origin
git branch -a
```

- [ ] **Step 3.5: If user wants LB submission**

Check cooldown first:
```bash
cd /home/nagaet/pxr-iduction-challenge && pixi run python track1_activity/scripts/api.py cooldown 2>&1 | tail -5
```

If READY and user selected `linear`:
```bash
cd /home/nagaet/pxr-iduction-challenge && pixi run python track1_activity/scripts/api.py submit track1_activity/submissions/ens_caruana_bag20_calibrated_linear.csv --experiment ens_caruana_bag20_calibrated_linear --notes "Linear post-hoc calibration of 10-pool caruana_bag20. Raw OOF MAE 0.4309, nested-CV MAE <L_mae>. Spearman preserved."
```

If `isotonic`:
```bash
cd /home/nagaet/pxr-iduction-challenge && pixi run python track1_activity/scripts/api.py submit track1_activity/submissions/ens_caruana_bag20_calibrated_isotonic.csv --experiment ens_caruana_bag20_calibrated_isotonic --notes "Isotonic post-hoc calibration of 10-pool caruana_bag20. Raw OOF MAE 0.4309, nested-CV MAE <I_mae>. Spearman preserved."
```

Replace `<L_mae>` or `<I_mae>` with the actual value.

---

## Self-Review Checklist

1. **Spec coverage:**
   - Single new script: Task 1 ✓
   - Input artifacts (OOF, weights, test CSV): Task 1.1 (`load_latest_caruana_weights`, `load_member_oof_matrix`, `load_raw_test_predictions`) ✓
   - Reconstructed ensemble OOF + sanity check: Task 1.1 (`reconstruct_ensemble_oof` + `RuntimeError` if MAE out of [0.40, 0.45]) ✓
   - 5-fold UMAP nested CV for both methods: Task 1.1 (`nested_cv_evaluate`) ✓
   - Final fit on full OOF + apply to test: Task 1.1 `main()` ✓
   - 2 submission CSVs written: Task 1.1 `sub_df.to_csv(...)` × 2 ✓
   - 2 DB experiment rows with `on_conflict_replace=True`: Task 1.1 `record_experiment(...)` × 2 ✓
   - Calibrated OOF saved for audit: Task 1.1 `save_oof_predictions(...)` × 2 ✓
   - Spearman preservation check: Task 1.1 warning when `dspear >= 0.005` ✓
   - Acceptance 1 (nested-CV MAE ≤ raw): Task 1.6 ✓
   - Acceptance 2 (Spearman |Δ| < 0.005): Task 1.1 warning ✓
   - Acceptance 3 (2 CSVs written): Task 1.4 ✓
   - Acceptance 4 (2 DB rows): Task 1.5 ✓
   - Acceptance 5 (ruff clean): Task 1.2 ✓
   - LB submission deferred, user decides: Task 3.3–3.5 ✓

2. **Placeholder scan:**
   - `<L_mae>` etc. in the PR body Step 3.1 are runtime values with explicit "Replace with actual values" instructions — not plan placeholders.
   - `<MAE>`, `<Δ>` etc. in Step 2.2 draft PR body are initial placeholders that Step 3.1 fills in; explicitly noted as "(Placeholder -- to be filled in from Step 1.3 output before marking ready for review.)"
   - No "TBD" / "implement later" / vague handoffs in the actual code or steps.

3. **Type consistency:**
   - `load_member_oof_matrix(member_names, n_train) -> np.ndarray` returns (n_train, n_members); consumed by `reconstruct_ensemble_oof(X_oof, weights, member_names)` which indexes columns — consistent ✓
   - `fit_calibrator(method, y_pred, y_true)` returns a sklearn model object; `apply_calibrator(method, model, y_pred)` takes the same object — consistent ✓
   - Linear reshapes `y_pred` to `(-1, 1)` for sklearn; isotonic accepts 1D — correctly branched in both fit and apply ✓
   - `record_experiment(... on_conflict_replace=True)` kwarg added in PR #97 merge — verified present in the current `evaluate.py` ✓

No issues found.

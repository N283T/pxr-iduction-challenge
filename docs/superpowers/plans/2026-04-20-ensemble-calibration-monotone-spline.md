# Positive-Constrained Affine + Monotone Spline Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `run_ensemble_calibrate.py` with `linear_pos` (slope ≥ 0 affine) and `spline_k5` (5-knot PCHIP monotone spline) calibrators. 4-way nested CV, pick best by MAE with Spearman guardrail, write one submission CSV.

**Architecture:** Single-script extension of `track1_activity/scripts/run_ensemble_calibrate.py`. Dispatch branches added to `fit_calibrator()` / `apply_calibrator()`. Main loop iterates over 4 methods. After the loop, a selection step picks the winner and writes 1 submission CSV plus a 5th DB row (`ens_caruana_bag20_calibrated_best`).

**Tech Stack:** `scipy.optimize.lsq_linear` for the positive-constrained affine, `scipy.interpolate.PchipInterpolator` for the spline. Existing `sklearn.isotonic`, `sklearn.linear_model`, `psycopg2` dependencies already present.

---

## Task 1: Add `linear_pos` calibrator

**Files:**
- Modify: `track1_activity/scripts/run_ensemble_calibrate.py:33-46` (imports)
- Modify: `track1_activity/scripts/run_ensemble_calibrate.py:118-137` (fit/apply)

- [ ] **Step 1: Add `lsq_linear` import**

Edit the imports block (line 33 area) to add `lsq_linear`. After the existing `from sklearn.linear_model import LinearRegression` line, add:

```python
from scipy.optimize import lsq_linear
```

- [ ] **Step 2: Add `linear_pos` branch to `fit_calibrator`**

In `fit_calibrator()` (line 118), after the existing `if method == "linear":` block and before `if method == "isotonic":`, add:

```python
    if method == "linear_pos":
        # Positive-constrained affine: slope >= 0 to guarantee order preservation.
        X = np.column_stack([y_pred, np.ones_like(y_pred)])
        res = lsq_linear(X, y_true, bounds=([0.0, -np.inf], [np.inf, np.inf]))
        return {"slope": float(res.x[0]), "intercept": float(res.x[1])}
```

- [ ] **Step 3: Add `linear_pos` branch to `apply_calibrator`**

In `apply_calibrator()` (line 131), after the existing `if method == "linear":` block and before `if method == "isotonic":`, add:

```python
    if method == "linear_pos":
        return model["slope"] * y_pred + model["intercept"]
```

- [ ] **Step 4: Smoke-fit sanity check (REPL)**

Run:

```bash
pixi run python -c "
import numpy as np
from scipy.optimize import lsq_linear
# Case 1: positive slope data -> unconstrained OLS answer
y_pred = np.linspace(4, 8, 100)
y_true = 1.1 * y_pred - 0.3 + np.random.default_rng(0).normal(scale=0.2, size=100)
X = np.column_stack([y_pred, np.ones_like(y_pred)])
res = lsq_linear(X, y_true, bounds=([0.0, -np.inf], [np.inf, np.inf]))
print('Case 1 (positive):', res.x)
# Case 2: adversarial flipped data -> constraint should kick in (slope=0)
y_flip = -0.5 * y_pred + 10.0 + np.random.default_rng(1).normal(scale=0.2, size=100)
res2 = lsq_linear(X, y_flip, bounds=([0.0, -np.inf], [np.inf, np.inf]))
print('Case 2 (flipped):', res2.x, 'constraint_active=', res2.x[0] == 0.0)
"
```

Expected: Case 1 slope ≈ 1.1, Case 2 slope = 0.0 with constraint_active=True.

- [ ] **Step 5: Commit**

```bash
git add track1_activity/scripts/run_ensemble_calibrate.py
git commit -m "feat(calibrate): add linear_pos (slope >= 0 affine) calibrator"
```

---

## Task 2: Add `spline_k5` calibrator

**Files:**
- Modify: `track1_activity/scripts/run_ensemble_calibrate.py:33-46` (imports)
- Modify: `track1_activity/scripts/run_ensemble_calibrate.py:118-137` (fit/apply)
- Modify: same file, add helper function above `fit_calibrator`.

- [ ] **Step 1: Add `PchipInterpolator` import**

In the imports block, add:

```python
from scipy.interpolate import PchipInterpolator
```

- [ ] **Step 2: Add `_fit_spline_k5` helper**

Immediately above `fit_calibrator()` (line 118), add:

```python
def _fit_spline_k5(y_pred: np.ndarray, y_true: np.ndarray, k: int = 5) -> dict:
    """Fit a k-knot monotone PCHIP spline on quantile bins.

    Uses OOF prediction quantiles as knot x-positions and bin-mean pEC50 as
    knot y-values. Monotonicity of y-values is enforced via cumulative max
    before PCHIP interpolation.
    """
    quantiles = np.linspace(0.0, 1.0, k + 1)
    edges = np.quantile(y_pred, quantiles)
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    bin_idx = np.clip(np.digitize(y_pred, edges) - 1, 0, k - 1)

    xs = np.zeros(k)
    ys = np.zeros(k)
    for kk in range(k):
        mask = bin_idx == kk
        if mask.sum() < 2:
            xs[kk] = 0.5 * (edges[kk] + edges[kk + 1])
            ys[kk] = ys[kk - 1] if kk > 0 else float(y_true.min())
        else:
            xs[kk] = float(y_pred[mask].mean())
            ys[kk] = float(y_true[mask].mean())

    # Enforce x strictly increasing (quantile bin-means can tie on extremely
    # discrete OOF); if ties occur, nudge.
    for i in range(1, k):
        if xs[i] <= xs[i - 1]:
            xs[i] = xs[i - 1] + 1e-6

    # Enforce monotone non-decreasing y.
    ys = np.maximum.accumulate(ys)

    spline = PchipInterpolator(xs, ys, extrapolate=False)
    return {
        "xs": xs,
        "ys": ys,
        "spline": spline,
        "x_min": float(xs[0]),
        "x_max": float(xs[-1]),
    }
```

- [ ] **Step 3: Add `spline_k5` branch to `fit_calibrator`**

In `fit_calibrator()`, after the `linear_pos` branch, add:

```python
    if method == "spline_k5":
        return _fit_spline_k5(y_pred, y_true, k=5)
```

- [ ] **Step 4: Add `spline_k5` branch to `apply_calibrator`**

In `apply_calibrator()`, after the `linear_pos` branch, add:

```python
    if method == "spline_k5":
        y_clipped = np.clip(y_pred, model["x_min"], model["x_max"])
        return np.asarray(model["spline"](y_clipped), dtype=np.float64)
```

- [ ] **Step 5: Smoke-fit sanity check (REPL)**

```bash
pixi run python -c "
import sys
sys.path.insert(0, 'track1_activity/scripts')
import numpy as np
import importlib.util
spec = importlib.util.spec_from_file_location('calmod', 'track1_activity/scripts/run_ensemble_calibrate.py')
m = importlib.util.module_from_spec(spec)
# Skip module execution (has DB side effects on import path); just test helpers by re-declaring.
from scipy.interpolate import PchipInterpolator
rng = np.random.default_rng(0)
y_pred = rng.uniform(4, 8, size=500)
y_true = 0.8 * y_pred + 1.0 + rng.normal(scale=0.3, size=500)
quantiles = np.linspace(0.0, 1.0, 6)
edges = np.quantile(y_pred, quantiles)
edges[0] -= 1e-9; edges[-1] += 1e-9
bin_idx = np.clip(np.digitize(y_pred, edges) - 1, 0, 4)
xs = np.array([y_pred[bin_idx == k].mean() for k in range(5)])
ys = np.maximum.accumulate(np.array([y_true[bin_idx == k].mean() for k in range(5)]))
spl = PchipInterpolator(xs, ys, extrapolate=False)
# Monotone check
x_dense = np.linspace(xs[0], xs[-1], 1000)
y_dense = spl(x_dense)
print('monotone:', np.all(np.diff(y_dense) >= -1e-9))
print('range:', y_dense.min(), y_dense.max())
"
```

Expected: `monotone: True`, reasonable range.

- [ ] **Step 6: Commit**

```bash
git add track1_activity/scripts/run_ensemble_calibrate.py
git commit -m "feat(calibrate): add spline_k5 (5-knot PCHIP monotone) calibrator"
```

---

## Task 3: Expand main loop to 4 methods + per-method diagnostics

**Files:**
- Modify: `track1_activity/scripts/run_ensemble_calibrate.py:202-291` (main loop)

- [ ] **Step 1: Change method iteration tuple**

In `main()`, change line ~206:

```python
    for method in ("linear", "isotonic"):
```

to:

```python
    for method in ("linear", "linear_pos", "spline_k5", "isotonic"):
```

- [ ] **Step 2: Extend per-method diagnostic print block**

Replace the existing if/else for diagnostics (lines ~228-249) with:

```python
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
        elif method == "linear_pos":
            fitted = {
                "slope": full_model["slope"],
                "intercept": full_model["intercept"],
                "constraint_active": full_model["slope"] == 0.0,
            }
            print(
                f"  Linear(pos) fit: slope={fitted['slope']:.4f}, "
                f"intercept={fitted['intercept']:.4f}, "
                f"constraint_active={fitted['constraint_active']}"
            )
        elif method == "spline_k5":
            fitted = {
                "n_knots": int(len(full_model["xs"])),
                "x_knots": full_model["xs"].tolist(),
                "y_knots": full_model["ys"].tolist(),
                "x_min": full_model["x_min"],
                "x_max": full_model["x_max"],
            }
            print(
                f"  Spline(k5) fit: n_knots={fitted['n_knots']}, "
                f"x_range=[{fitted['x_min']:.3f}, {fitted['x_max']:.3f}], "
                f"y_range=[{full_model['ys'].min():.3f}, {full_model['ys'].max():.3f}]"
            )
        else:  # isotonic
            fitted = {
                "n_knots": int(len(full_model.X_thresholds_)),
                "x_min": float(full_model.X_min_),
                "x_max": float(full_model.X_max_),
                "y_min": float(full_model.y_thresholds_.min()),
                "y_max": float(full_model.y_thresholds_.max()),
            }
            print(
                f"  Isotonic fit: n_knots={fitted['n_knots']}, "
                f"x_range=[{fitted['x_min']:.3f}, {fitted['x_max']:.3f}], "
                f"y_range=[{fitted['y_min']:.3f}, {fitted['y_max']:.3f}]"
            )
```

- [ ] **Step 3: Collect per-method results for the selection step**

Just above the `for method in ...` loop (around line 206, inside `main()`), add:

```python
    method_results: dict[str, dict] = {}
```

Inside the loop, just before the `# Record experiment with nested-CV metrics` block (around line 265), add:

```python
        method_results[method] = {
            "nested_cv_metrics": nested_cv_metrics,
            "fold_metrics": fold_metrics,
            "calibrated_oof": calibrated_oof,
            "full_model": full_model,
            "fitted": fitted,
            "test_pred_cal": test_pred_cal,
        }
```

- [ ] **Step 4: Commit**

```bash
git add track1_activity/scripts/run_ensemble_calibrate.py
git commit -m "refactor(calibrate): iterate 4 methods, collect results for selection"
```

---

## Task 4: Add selection rule + best submission CSV + 5th DB row

**Files:**
- Modify: `track1_activity/scripts/run_ensemble_calibrate.py` (end of `main()`, after the per-method loop)

- [ ] **Step 1: Add selection step after the per-method loop**

After the per-method for-loop closes (after the existing `print(f"  Recorded experiment id={exp_id}")` and before the final `print("\nDone.")`), add:

```python
    # ---- Selection step: pick best by MAE with Spearman guardrail ---------
    print("\n" + "=" * 60)
    print("  SELECTION (MAE min, |ΔSpearman| < 0.005 guardrail)")
    print("=" * 60)
    raw_spearman = raw_oof_metrics["Spearman_R"]
    raw_mae = raw_oof_metrics["MAE"]

    summary_rows = []
    for name, res in method_results.items():
        m = res["nested_cv_metrics"]
        d_spear = abs(m["Spearman_R"] - raw_spearman)
        d_mae = m["MAE"] - raw_mae
        passes = (d_spear < 0.005) and (m["MAE"] < raw_mae)
        summary_rows.append((name, m["MAE"], d_mae, m["Spearman_R"], d_spear, passes))
        print(
            f"  {name:<12} MAE={m['MAE']:.4f} (Δ{d_mae:+.4f})  "
            f"Spearman={m['Spearman_R']:.4f} (Δ{d_spear:+.4f})  "
            f"passes_guardrail={passes}"
        )

    candidates = [r for r in summary_rows if r[5]]
    if not candidates:
        print(
            "\n  WARNING: No calibrator improves MAE with Spearman preserved. "
            "Skipping best submission CSV."
        )
        print("\nDone.")
        return

    best_name = min(candidates, key=lambda r: r[1])[0]
    best_res = method_results[best_name]
    print(f"\n  BEST: {best_name}")

    # Write single 'best' submission CSV
    best_sub_name = "ens_caruana_bag20_calibrated_best"
    best_sub_path = SUBMISSION_DIR.joinpath(f"{best_sub_name}.csv")
    best_sub_df = pd.DataFrame(
        {
            "SMILES": raw_test["SMILES"],
            "Molecule Name": raw_test["Molecule Name"],
            "pEC50": best_res["test_pred_cal"],
        }
    )
    best_sub_df.to_csv(best_sub_path, index=False)
    print(f"  Wrote best submission: {best_sub_path}")

    # Record meta experiment row
    best_exp_id = record_experiment(
        name=best_sub_name,
        description=(
            f"Best post-hoc calibration of ens_caruana_bag20 "
            f"(selected: {best_name}, 5-fold UMAP nested CV, MAE min + "
            f"|ΔSpearman| < 0.005 guardrail)"
        ),
        model_type="ensemble_calibrated",
        feature_set="caruana_bag20_output",
        hyperparameters={
            "selected_method": best_name,
            "fitted_params": best_res["fitted"],
            "candidates_considered": [r[0] for r in summary_rows],
            "selection_rule": "min MAE s.t. |Spearman - raw| < 0.005",
            "source_ensemble": "ens_caruana_bag20",
            "source_members": member_names,
            "source_weights": weights,
        },
        fold_metrics=best_res["fold_metrics"],
        submission_path=f"track1_activity/submissions/{best_sub_name}.csv",
        notes=(
            f"Raw OOF MAE={raw_mae:.4f}, "
            f"calibrated nested-CV MAE={best_res['nested_cv_metrics']['MAE']:.4f} "
            f"(method={best_name}), ΔMAE="
            f"{best_res['nested_cv_metrics']['MAE'] - raw_mae:+.4f}"
        ),
        on_conflict_replace=True,
    )
    save_oof_predictions(best_exp_id, best_res["calibrated_oof"])
    print(f"  Recorded best meta-experiment id={best_exp_id}")

    print("\nDone.")
```

- [ ] **Step 2: Commit**

```bash
git add track1_activity/scripts/run_ensemble_calibrate.py
git commit -m "feat(calibrate): select best by MAE+Spearman guardrail, write 1 best CSV"
```

---

## Task 5: Smoke test + ruff clean

**Files:**
- Run: `track1_activity/scripts/run_ensemble_calibrate.py`

- [ ] **Step 1: Run end-to-end**

```bash
pixi run python track1_activity/scripts/run_ensemble_calibrate.py 2>&1 | tee /tmp/calibrate_run.log
```

Expected output:
- "Reconstructed raw ensemble OOF" block showing raw MAE ≈ 0.4309
- 4 per-method nested CV blocks (linear, linear_pos, spline_k5, isotonic)
- Diagnostic print for each (slope/intercept or knot info)
- SELECTION block with 4 summary rows
- One "BEST: <name>" line
- "Wrote best submission: .../ens_caruana_bag20_calibrated_best.csv"
- "Recorded best meta-experiment id=..."

- [ ] **Step 2: Verify DB rows**

```bash
pixi run python -c "
import psycopg2
c = psycopg2.connect(dbname='pxr_challenge', host='/tmp', port=5433)
cur = c.cursor()
cur.execute('''
SELECT e.name, AVG(r.mae) as mae_mean, AVG(r.spearman_r) as spr_mean
FROM experiments e
JOIN experiment_cv_results r ON e.id = r.experiment_id
WHERE e.name LIKE 'ens_caruana_bag20%calibrated%'
   OR e.name = 'ens_caruana_bag20'
GROUP BY e.id, e.name
ORDER BY e.id DESC LIMIT 10
''')
for r in cur.fetchall():
    print(f'{r[0]:<55} MAE={float(r[1]):.4f} Spearman={float(r[2]):.4f}')
"
```

Expected: 5 calibrated rows (linear, linear_pos, spline_k5, isotonic, best) + 1 raw row.

- [ ] **Step 3: Verify CSV**

```bash
head -3 /home/nagaet/pxr-iduction-challenge/track1_activity/submissions/ens_caruana_bag20_calibrated_best.csv
wc -l /home/nagaet/pxr-iduction-challenge/track1_activity/submissions/ens_caruana_bag20_calibrated_best.csv
```

Expected: header `SMILES,Molecule Name,pEC50` + 513 data rows (= 514 total lines).

- [ ] **Step 4: Ruff clean**

```bash
pixi run ruff format track1_activity/scripts/run_ensemble_calibrate.py
pixi run ruff check track1_activity/scripts/run_ensemble_calibrate.py
```

Expected: no errors.

- [ ] **Step 5: Commit any ruff fixes**

If ruff made changes:

```bash
git add track1_activity/scripts/run_ensemble_calibrate.py
git commit -m "style: ruff format run_ensemble_calibrate.py"
```

---

## Task 6: Push + PR

- [ ] **Step 1: Push branch**

```bash
git push -u origin feat/calibrate-monotone-spline
```

- [ ] **Step 2: Create PR**

```bash
gh pr create --title "feat(calibrate): positive-constrained + monotone-spline calibrators" --body "$(cat <<'EOF'
## Summary

Extends `run_ensemble_calibrate.py` with two calibrators that close the gaps identified in the ChatGPT Deep-Research report:

- **`linear_pos`**: positive-constrained affine (`slope >= 0` via `scipy.optimize.lsq_linear`). Order-preservation safety guarantee.
- **`spline_k5`**: 5-quantile-knot PCHIP monotone spline. Middle ground between the 2-parameter linear and full-resolution isotonic.

Runs 4-way nested CV across `{linear, linear_pos, spline_k5, isotonic}`, picks the best by MAE subject to a Spearman guardrail (`|Δ| < 0.005`), writes a single submission CSV `ens_caruana_bag20_calibrated_best.csv`, and records a 5th DB meta-row `ens_caruana_bag20_calibrated_best` pointing to the winning method.

Earlier today's calibration PR observed that `isotonic` regressed MAE by +0.003 (matches the report's warning that full isotonic overfits on 4140-point OOF). The monotone spline is the missing middle option.

## Design doc

`docs/superpowers/specs/2026-04-20-ensemble-calibration-monotone-spline-design.md`

## RankRefine pivot context

RankRefine was considered earlier in the session. Feasibility check showed test 513 has 0.0% `log2_fc` coverage (organizer withholds all labels), so log2_fc cannot serve as a ranker. LLM-ranker variant was judged too expensive; path abandoned in favor of deeper calibration.

## Test plan
- [x] Smoke-run `run_ensemble_calibrate.py` end-to-end
- [x] Verify 5 calibrated DB rows (4 per-method + 1 meta-best)
- [x] Verify `ens_caruana_bag20_calibrated_best.csv` has 513 rows + correct header
- [x] Verify selection rule fires correctly (best method printed, reason logged)
- [x] Ruff format + ruff check clean
EOF
)"
```

- [ ] **Step 3: Report PR URL to user**

Wait for `gh pr create` output, report the URL, ask the user whether to merge after CI / review.

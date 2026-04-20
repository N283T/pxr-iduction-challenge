# Design: Positive-Constrained Affine + Low-DoF Monotone Spline Calibration

- **Status**: Proposed (2026-04-20 evening)
- **Author**: Claude Code session, brainstormed with user
- **Related**:
  - Follow-up to PR-level spec `2026-04-20-ensemble-regression-calibration-design.md` (linear + isotonic, committed earlier today).
  - Motivated by ChatGPT Deep-Research report on post-hoc regression calibration in molecular property prediction.
  - RankRefine path abandoned in the same session: test 513 has 0.0% `log2_fc` coverage (organizer withholds all labels), so log2_fc cannot serve as a ranker.

## Goal

Extend `track1_activity/scripts/run_ensemble_calibrate.py` with two additional calibrators that close the gaps identified in the Deep-Research report:

1. **Positive-constrained affine** (`linear_pos`) — replaces the unconstrained `LinearRegression` with a slope ≥ 0 constraint. Safety guarantee that order is preserved on the full OOF range.
2. **Low-DoF monotone spline** (`spline_k5`) — 5 quantile knots, monotone-enforced, PCHIP interpolation. Fills the gap between 2-parameter linear and full-resolution isotonic.

Run a **4-way nested CV** (`linear`, `linear_pos`, `spline_k5`, `isotonic`), pick the best by MAE with a Spearman guardrail, write exactly **one** submission CSV.

## Background

Earlier today's nested CV (5-fold UMAP, seed=42) returned:

| Method | MAE | ΔMAE vs raw | Spearman_R | ΔSpearman |
|---|---|---|---|---|
| Raw (no calibration) | 0.4309 | — | 0.8156 | — |
| `linear` (unconstrained) | 0.4299 | **−0.0010** | 0.8126 | −0.0030 |
| `isotonic` (full resolution) | 0.4338 | **+0.0029** (regression) | 0.8120 | −0.0036 |

Two observations motivate this PR:

- `linear` improvement is marginal (within nested-CV noise) but directionally correct.
- `isotonic` regresses, consistent with the Deep-Research report's warning that "full isotonic may memorize noise" on a 4,140-point OOF.

The Deep-Research priority ranking (1 = highest):

1. **Positive-constrained affine** — order-preserving safety.
2. **Low-DoF strictly monotone spline** — middle ground, lower variance than full isotonic.
3. Full isotonic (only when it measurably wins; empirically doesn't here).
4. Importance-weighted affine (shift-aware; deferred to a future PR).

## Non-goals (deferred)

- **Importance-weighted calibration** (Option Z from the session). Needs a domain classifier + density ratio. Tracked as future work.
- **Per-member calibration** before ensembling. Structurally redundant with caruana (monotonic transform would be selected against).
- **LLM-ranker RankRefine**. Abandoned this session; see Background.
- **Alternative knot counts** (K=3, K=10) as separate Optuna pass. Fixed K=5 per Deep-Research "3–5 knots" recommendation; revisit only if `spline_k5` results are anomalous.
- **Conformal / CQR / Venn-Abers**. Report explicitly flagged these as interval-only, not point-MAE.

## Architecture

### Single-file extension

Modify `track1_activity/scripts/run_ensemble_calibrate.py`:

- Extend `fit_calibrator(method, y_pred, y_true)` with two new methods:
  - `"linear_pos"` — `scipy.optimize.lsq_linear` with bounds `([0, -inf], [inf, inf])` on `[slope, intercept]`.
  - `"spline_k5"` — quantile-bin + `scipy.interpolate.PchipInterpolator`.
- Extend `apply_calibrator(method, model, y_pred)` symmetrically.
- Change the main loop from iterating over `("linear", "isotonic")` to iterating over `("linear", "linear_pos", "spline_k5", "isotonic")`.
- Add post-loop **selection step**: pick the best method by nested-CV MAE subject to a Spearman guardrail; write a single best submission CSV.

### `linear_pos` implementation

```python
from scipy.optimize import lsq_linear

def _fit_linear_pos(y_pred, y_true):
    X = np.column_stack([y_pred, np.ones_like(y_pred)])
    res = lsq_linear(X, y_true, bounds=([0.0, -np.inf], [np.inf, np.inf]))
    return {"slope": float(res.x[0]), "intercept": float(res.x[1])}

def _apply_linear_pos(model, y_pred):
    return model["slope"] * y_pred + model["intercept"]
```

Diagnostic print: slope, intercept, whether constraint was active (slope == 0 exactly).

### `spline_k5` implementation

```python
from scipy.interpolate import PchipInterpolator

def _fit_spline_k5(y_pred, y_true, K: int = 5):
    # 1. Bin by OOF prediction quantile
    quantiles = np.linspace(0.0, 1.0, K + 1)
    edges = np.quantile(y_pred, quantiles)
    edges[0] -= 1e-9  # include min
    edges[-1] += 1e-9  # include max
    bin_idx = np.digitize(y_pred, edges) - 1
    bin_idx = np.clip(bin_idx, 0, K - 1)

    # 2. Bin-mean pEC50 and bin-center OOF prediction
    xs = np.zeros(K)
    ys = np.zeros(K)
    for k in range(K):
        mask = bin_idx == k
        if mask.sum() < 2:
            # Degenerate bin: use midpoint of edge + previous y_true
            xs[k] = 0.5 * (edges[k] + edges[k + 1])
            ys[k] = ys[k - 1] if k > 0 else y_true.min()
        else:
            xs[k] = y_pred[mask].mean()
            ys[k] = y_true[mask].mean()

    # 3. Enforce monotonicity on y values
    ys = np.maximum.accumulate(ys)

    # 4. PCHIP (shape-preserving cubic Hermite)
    spline = PchipInterpolator(xs, ys, extrapolate=False)
    return {"xs": xs, "ys": ys, "spline": spline, "x_min": xs[0], "x_max": xs[-1]}

def _apply_spline_k5(model, y_pred):
    # Clip to fit range, then evaluate
    y_clipped = np.clip(y_pred, model["x_min"], model["x_max"])
    return model["spline"](y_clipped)
```

Design notes:

- **K=5** per Deep-Research report recommendation ("3–5 knots").
- **Quantile-bin knot placement** concentrates resolution where OOF data density is highest.
- **`np.maximum.accumulate`** enforces monotonicity at knot positions; PCHIP then preserves monotonicity between knots.
- **Clip-on-extrapolation** matches `IsotonicRegression(out_of_bounds="clip")` behavior.

### Selection rule

After nested-CV for all 4 methods:

```python
# Candidates: methods with Spearman guardrail satisfied
candidates = [
    m for m in method_results
    if abs(m["spearman"] - raw_spearman) < 0.005
]

if not candidates or all(c["mae"] >= raw_mae for c in candidates):
    print("WARNING: No calibrator improves MAE with Spearman preserved. "
          "Skipping submission.")
else:
    best = min(candidates, key=lambda c: c["mae"])
    # Refit on full OOF, apply to raw test, write single CSV.
```

Output CSV: `track1_activity/submissions/ens_caruana_bag20_calibrated_best.csv`
Experiment name: `ens_caruana_bag20_calibrated_best` (DB row with method and reason in `notes`).

### DB records

5 experiment rows (idempotent via `on_conflict_replace=True`):

1. `ens_caruana_bag20_calibrated_linear` (existing; overwritten)
2. `ens_caruana_bag20_calibrated_linear_pos` (new)
3. `ens_caruana_bag20_calibrated_spline_k5` (new)
4. `ens_caruana_bag20_calibrated_isotonic` (existing; overwritten)
5. `ens_caruana_bag20_calibrated_best` (new: meta-row pointing to the winning method; submission CSV attached here)

OOF predictions saved for all 5.

## Data flow

```
X_oof (4140, 10) @ caruana_weights  =>  y_pred_oof (4140,) raw ensemble OOF
                   ↓
     5-fold UMAP nested CV, 4 methods:
         linear, linear_pos, spline_k5, isotonic
                   ↓
      4 calibrated OOF vectors + 4 metric dicts
                   ↓
          selection rule (MAE min, |ΔSpearman| < 0.005)
                   ↓
       best method refit on full OOF, applied to raw test
                   ↓
     1 submission CSV + 5 DB rows (4 per-method + 1 meta-best)
```

## Acceptance criteria

1. All 4 methods converge on all 5 folds without NaN.
2. `linear_pos`: slope >= 0 constraint verified; if slope equals the existing unconstrained value, log "constraint inactive".
3. `spline_k5`: knot y-values are monotone non-decreasing after `np.maximum.accumulate`; spline output is finite on the full OOF range.
4. Best method's Spearman delta from raw < 0.005 (guardrail).
5. Best method's MAE <= raw MAE; otherwise no CSV is written and the user is told "no calibrator improved MAE — not submitting".
6. `ruff format` and `ruff check` clean on the modified script.

Failure handling:

- **No method wins**: skip submission; keep the 4 per-method DB rows for diagnosis.
- **All methods violate Spearman guardrail**: same as above, with the reason logged.
- **Any method NaNs during CV**: RuntimeError; abort before writing submission.

## Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| `spline_k5` at K=5 is still too flexible given 4140 points / 5 folds | Low | Monotonicity enforcement + PCHIP shape-preservation limits over-curling |
| Degenerate quantile bin with <2 points | Very low | Guarded in `_fit_spline_k5`; inherits previous bin's y-value |
| `linear_pos` hits the slope=0 boundary and collapses to intercept-only | Low | Diagnostic print; Spearman guardrail naturally rejects the collapsed fit |
| All 4 methods regress MAE | Low | Submission skipped; audit trail preserved in DB |
| Selection picks a noise-driven winner (e.g., `spline_k5` barely beats `linear`) | Medium | Guardrail is MAE + Spearman only; accept the tie-winner — the report is explicit that LB transfer is uncertain regardless |

## ETA

- Implementation: ~1.5 h
- Smoke test (run_ensemble_calibrate.py on the existing ens_caruana_bag20 OOF): ~5 min CPU-only
- LB submission: 1 cooldown slot (user-approved)

## Future work

- **Option Z (importance-weighted affine)**: domain classifier (train vs test, Morgan FP + LR) → density ratio → weighted least squares for `linear_pos`. Separate PR.
- **Per-cluster calibration**: different calibrator per chemical-space region (UMAP cluster). Requires a separate design pass on cluster-vs-cluster variance.
- **Calibrator stacking**: linear_pos → spline_k5 residual fit. Rejected for now as it doubles the degrees of freedom.

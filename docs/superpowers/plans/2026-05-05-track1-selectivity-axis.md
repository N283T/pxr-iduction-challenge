# Track 1 Selectivity Axis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an internal-only selectivity-axis analysis that creates small gated correction candidates against the id48 Track 1 anchor.

**Architecture:** Add one standalone script under `track1_activity/analysis/selectivity_axis/`. It reads DB-backed train/test data and existing id48 predictions, trains cross-fit auxiliary selectivity predictors, fits a low-dimensional gated residual correction, writes candidate CSVs and diagnostics, and leaves production ensemble code untouched.

**Tech Stack:** Python 3.12, pandas, numpy, scikit-learn, RDKit fingerprints via existing helpers, LightGBM if available with sklearn fallback.

---

### Task 1: Standalone Selectivity Axis Builder

**Files:**
- Create: `track1_activity/analysis/selectivity_axis/build_selectivity_axis.py`

- [ ] **Step 1: Implement data loading**

Load train/test rows in canonical order, including `compound_id`, SMILES, pEC50,
counter pEC50, counter Emax, and narrow descriptors. Load the id48 test CSV and
reconstruct id48 OOF using the existing internal-decorrelation helper.

- [ ] **Step 2: Implement auxiliary labels**

Create:

```python
counter_active = counter_pec50.notna()
nonselective = counter_active & ((pec50 - counter_pec50).abs() <= 0.30)
selectivity_delta = pec50 - counter_pec50
```

Train `selectivity_delta` only on active counter rows.

- [ ] **Step 3: Train cross-fit auxiliary predictors**

Use five KFold splits with shuffled `random_state=42`. Predict:

```text
counter_active_prob
nonselective_prob
selectivity_delta_pred
```

Train final full-data models to produce the same auxiliary columns on test.

- [ ] **Step 4: Fit gated residual correction**

Fit a small ridge model:

```text
residual = pec50 - id48_oof
features = [
  counter_active_prob,
  nonselective_prob,
  selectivity_delta_pred,
  id48_oof,
  id48_oof * nonselective_prob,
  id48_oof * selectivity_delta_pred,
]
```

Generate candidates with shrink factors `[0.10, 0.20, 0.30]` and clip values
`[0.03, 0.05]`.

- [ ] **Step 5: Write diagnostics and candidates**

Write:

```text
track1_activity/analysis/selectivity_axis/outputs/summary.csv
track1_activity/analysis/selectivity_axis/outputs/report.md
track1_activity/submissions/ens_selectivity_axis_*.csv
```

Each candidate row must include OOF MAE delta, Spearman delta, test mean absolute
shift, p90 shift, max shift, and path.

### Task 2: Verification

**Files:**
- Verify: `track1_activity/analysis/selectivity_axis/build_selectivity_axis.py`
- Verify: generated candidate CSVs

- [ ] **Step 1: Format and lint**

Run:

```bash
pixi run ruff format track1_activity/analysis/selectivity_axis/build_selectivity_axis.py
pixi run ruff check track1_activity/analysis/selectivity_axis/build_selectivity_axis.py
```

Expected: ruff check passes.

- [ ] **Step 2: Run the builder**

Run:

```bash
pixi run python track1_activity/analysis/selectivity_axis/build_selectivity_axis.py
```

Expected: report and summary files are written. Candidate CSVs have 513 rows and
columns `SMILES`, `Molecule Name`, `pEC50`.

- [ ] **Step 3: Update issue #100**

Post a short note with the best candidate, OOF diagnostics, shift diagnostics,
and whether the axis should be submitted after the scheduled `g10` result.

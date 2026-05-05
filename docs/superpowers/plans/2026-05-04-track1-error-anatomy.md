# Track 1 Error Anatomy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible Track 1 analysis report that identifies where the current ensemble makes large OOF errors and which internal signals may correct those slices.

**Architecture:** Add one focused analysis package under `track1_activity/analysis/error_anatomy/`. The main script loads current ensemble/member OOF predictions from PostgreSQL, joins train descriptors and auxiliary assay columns, computes residual slice summaries, and writes CSV/Markdown outputs. A small unittest file covers the pure summary helpers.

**Tech Stack:** Python 3.12, pandas, numpy, scipy/sklearn metrics where needed, existing `track1_activity/src` loaders, PostgreSQL via SQLAlchemy.

---

### Task 1: Pure Helper Tests

**Files:**
- Create: `track1_activity/analysis/error_anatomy/test_error_anatomy.py`
- Create: `track1_activity/analysis/error_anatomy/error_anatomy.py`

- [ ] **Step 1: Write helper tests for quantile bins and slice summaries**

Create `test_error_anatomy.py` with tests for `safe_qcut()` and `summarize_binary_slice()` using small deterministic DataFrames.

- [ ] **Step 2: Add minimal helper implementations**

Create `error_anatomy.py` with pure functions only: `safe_qcut`, `mean_abs_error`, and `summarize_binary_slice`.

- [ ] **Step 3: Run tests**

Run: `pixi run python track1_activity/analysis/error_anatomy/test_error_anatomy.py`
Expected: all tests pass.

### Task 2: Residual Dataset Builder

**Files:**
- Modify: `track1_activity/analysis/error_anatomy/error_anatomy.py`

- [ ] **Step 1: Load current ensemble and member OOF**

Use `experiments` + `experiment_oof_predictions` to load `ens_caruana_bag20` and current `run_ensemble.ENSEMBLE_MODELS`. Join by `train_idx` and assert full coverage for 4140 train rows.

- [ ] **Step 2: Join train metadata**

Load train rows in `ORDER BY t.id`, including `compound_id`, `molecule_name`, `std_smiles`, `pec50`, counter assay fields, single-conc fields, and `compound_descriptors` columns.

- [ ] **Step 3: Compute residual features**

Add columns: `pred`, `residual=y-pred`, `abs_error`, `member_std`, `member_range`, `chemprop_family_mean`, `non_chemprop_mean`, and `family_gap`.

### Task 3: Slice Reports

**Files:**
- Modify: `track1_activity/analysis/error_anatomy/error_anatomy.py`

- [ ] **Step 1: Add descriptor/assay quantile summaries**

Generate slice summaries for `pec50`, `pred`, `logp`, `tpsa`, `exactmw`, `num_heavy_atoms`, `member_std`, `family_gap`, `counter_pec50`, `log2fc_8_25e_6`, and `log2fc_3_30e_5`.

- [ ] **Step 2: Add binary slice summaries**

Generate binary summaries for high-disagreement, high/low residual tails, counter assay availability, single-conc availability, high logP, high MW, and high TPSA.

- [ ] **Step 3: Write outputs**

Write `outputs/residuals.csv`, `outputs/quantile_slices.csv`, `outputs/binary_slices.csv`, `outputs/top_errors.csv`, `outputs/member_family_gaps.csv`, and `outputs/report.md`.

### Task 4: Verification

**Files:**
- Existing analysis files only.

- [ ] **Step 1: Format and lint**

Run: `pixi run ruff format track1_activity/analysis/error_anatomy/error_anatomy.py track1_activity/analysis/error_anatomy/test_error_anatomy.py`
Run: `pixi run ruff check track1_activity/analysis/error_anatomy/error_anatomy.py track1_activity/analysis/error_anatomy/test_error_anatomy.py`

- [ ] **Step 2: Run tests and analysis**

Run: `pixi run python track1_activity/analysis/error_anatomy/test_error_anatomy.py`
Run: `pixi run python track1_activity/analysis/error_anatomy/error_anatomy.py`

- [ ] **Step 3: Read report and choose next candidate**

Open `track1_activity/analysis/error_anatomy/outputs/report.md` and identify whether the next action should be disagreement routing, assay-tail correction, or no-op.

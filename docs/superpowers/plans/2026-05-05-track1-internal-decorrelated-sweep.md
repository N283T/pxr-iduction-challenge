# Track 1 Internal Decorrelated Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build no-external-data Track 1 candidates that add decorrelated internal model signal while staying close to the LB-proven id48 meta-axis anchor.

**Architecture:** Add an analysis-only script under `track1_activity/analysis/internal_decorrelation/`. It loads existing DB OOF predictions plus submission CSVs, filters to internal candidates, runs residual-correlation-capped Caruana sweeps, applies the existing importance affine calibrator to each candidate, then materializes only small blends against `ens_meta_axis_a343`.

**Tech Stack:** Python 3.12 via pixi, pandas, numpy, scipy, scikit-learn, psycopg2, existing Track 1 DB helpers.

---

### Task 1: Create Analysis Script

**Files:**
- Create: `track1_activity/analysis/internal_decorrelation/decorrelated_caruana_sweep.py`

- [ ] **Step 1: Load data and anchors**

Use `load_train_smiles_target()`, existing OOF rows, `ens_meta_axis_a343.csv`, and the id42/id43 OOF reconstruction logic already in `lb_proxy_battery_v2.py`.

- [ ] **Step 2: Build internal candidate pool**

Query experiments with 4,140 OOF rows, submission CSVs, MAE <= 0.50, and no external-data markers in the experiment name such as `admet`, `drugclip`, or `oe_`.

- [ ] **Step 3: Run decorrelated Caruana sweeps**

Filter by residual correlation to the id48 OOF proxy, run bagged Caruana with several caps and bag settings, then apply the existing importance affine calibrator to each candidate blend.

- [ ] **Step 4: Materialize conservative id48 blends**

For each candidate, create blends with id48 at lambda 0.10, 0.20, and 0.30. Write only candidates with moderate test shift to `track1_activity/submissions/`.

### Task 2: Verify and Summarize

**Files:**
- Output: `track1_activity/analysis/internal_decorrelation/outputs/decorrelated_caruana_sweep/summary.csv`
- Output: `track1_activity/analysis/internal_decorrelation/outputs/decorrelated_caruana_sweep/report.md`

- [ ] **Step 1: Run lint**

Run: `pixi run ruff check track1_activity/analysis/internal_decorrelation/decorrelated_caruana_sweep.py`

- [ ] **Step 2: Run the sweep**

Run: `pixi run python track1_activity/analysis/internal_decorrelation/decorrelated_caruana_sweep.py`

- [ ] **Step 3: Pick recommendation**

Recommend a submit candidate only if it moves id48 by mean absolute shift <= 0.04 and improves the id48 OOF proxy without using failed residual correction features.

# Boltz Trunk Fast Inventory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a factual audit of Boltz full-run and fast embeddings-only trunk coverage before choosing a new Boltz candidate.

**Architecture:** Add one focused inventory script under `track1_activity/scripts/boltz_affhead/` plus a direct Python test. The script queries DB coverage, samples raw `embeddings_*.npz` files, summarizes existing Boltz-family experiments, and writes a markdown report. It does not register new features or train models.

**Tech Stack:** Python 3.12, pixi, pandas, numpy, psycopg2, pathlib.

---

### Task 1: Inventory Helpers

**Files:**
- Create: `track1_activity/scripts/boltz_affhead/37_trunk_fast_inventory.py`
- Create: `track1_activity/tests/test_boltz_trunk_fast_inventory.py`

- [ ] **Step 1: Write tests**

Test pure helpers for:

- missing ID computation from compound and trunk ID lists
- recycling count formatting
- NPZ shape summary on a temporary synthetic `s/z` file
- Boltz experiment name filter

- [ ] **Step 2: Verify tests fail**

Run:

```bash
pixi run python track1_activity/tests/test_boltz_trunk_fast_inventory.py
```

Expected: fail because the inventory module does not exist.

- [ ] **Step 3: Implement inventory script**

Script responsibilities:

- query `compounds`, `compound_boltz2`, `compound_boltz2_trunk_fast`
- count full-run rows, embedding paths, rcycle split, missing compound IDs
- sample readable NPZ files by rcycle and report shapes/token counts/file sizes
- list top Boltz-family experiments from `experiment_summary`
- write `track1_activity/analysis/boltz_trunk_fast_inventory/outputs/report.md`

- [ ] **Step 4: Verify tests and style**

Run:

```bash
pixi run python track1_activity/tests/test_boltz_trunk_fast_inventory.py
pixi run ruff check track1_activity/scripts/boltz_affhead/37_trunk_fast_inventory.py track1_activity/tests/test_boltz_trunk_fast_inventory.py
```

Expected: pass.

### Task 2: Run Audit

**Files:**
- Create: `track1_activity/analysis/boltz_trunk_fast_inventory/outputs/report.md`

- [ ] **Step 1: Execute inventory**

Run:

```bash
pixi run python track1_activity/scripts/boltz_affhead/37_trunk_fast_inventory.py
```

Expected: markdown report created with coverage and existing Boltz-family experiment summary.

### Task 3: Decide Next Candidate

**Files:**
- No code changes unless the report identifies a clear next candidate.

- [ ] **Step 1: Read report**

Choose one next implementation direction:

- raw NPZ re-pooling from 13,134 source files, or
- residual-aware trunk pretraining, or
- stop if audit shows the Boltz family is exhausted for now.

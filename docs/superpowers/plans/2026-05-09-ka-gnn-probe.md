# KA-GNN Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the KA-GNN idea from `LongLee220/KA-GNN` into the PXR Track 1 PyG stack and evaluate whether it creates a decorrelated graph-model axis.

**Architecture:** Implement a PyG-native Fourier KA-GNN layer matching the upstream `KA_GNN` spirit: Fourier KAN edge/message transforms, residual message passing, global pooling, and KAN readout. Train it directly on pEC50 UMAP folds with fold-local target standardization. Use existing PyG SMILES conversion and DB/evaluation utilities.

**Tech Stack:** Python 3.12, pixi, PyTorch, PyG, RDKit, existing Track 1 loaders/evaluation, upstream KA-GNN as design reference.

---

### Task 1: PyG KA-GNN model module

**Files:**
- Create: `track1_activity/src/ka_gnn.py`
- Create: `track1_activity/tests/test_ka_gnn.py`

- [ ] Test Fourier KAN linear output shape and finite values.
- [ ] Test edge-feature aggregation concatenates node and incoming-edge mean features.
- [ ] Test KA-GNN model forwards a PyG batch to one graph-level prediction per graph.
- [ ] Implement the model module.

### Task 2: Training/evaluation script

**Files:**
- Create: `track1_activity/scripts/run_ka_gnn.py`
- Modify: `track1_activity/scripts/README.md`

- [ ] Load train/test SMILES and convert to PyG graphs.
- [ ] Train KA-GNN per canonical UMAP fold with early stopping and target standardization.
- [ ] Save fold metrics, OOF/test predictions, summary, and report.
- [ ] Optional DB record for best full-fold run.

### Task 3: Probes and decision

**Files:**
- Output: `track1_activity/analysis/ka_gnn/outputs/<run>/report.md`

- [ ] Run smoke test.
- [ ] Run at least one full KA-GNN configuration.
- [ ] Compare single-model OOF, residual correlation, and Caruana ADD weight.

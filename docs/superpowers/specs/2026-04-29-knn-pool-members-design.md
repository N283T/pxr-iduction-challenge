# kNN Pool Members (all-train + potent-46 anchored)

Date: 2026-04-29
Status: Approved (brainstorming compressed per user direction)
Owner: N283T

## Background

Track 1 LB rank 3 after id=44 anchor residual reverse-amplified
(0.4075 → 0.4090). Codex re-consult 2026-04-29 (after the 2-strike
analog-prior null/regress chain, PR #151 + PR #152): the calibrator/
correction tweak family is exhausted. Pivot to a different mechanism —
**retrieval-based prediction as an independent caruana pool member**.

Codex specific guidance:
- "kNN-derived prediction を 1 本の独立 member として作る方がよい"
- "Caruana pool に 1-5% でも乗れば勝ち筋がある"
- "重要なのは potent-46 限定ではなく、`all-train kNN` と `potent-46 anchored kNN` を分けること"

## Goal

Add two new caruana_bag20 pool members:

1. `knn_alltrain_umap` — sim²-weighted mean of pEC50 over all train
   compounds (Morgan r=2, Tanimoto, self-exclude on train).
2. `knn_potent46_umap` — sim²-weighted mean of pEC50 over the 46 potent
   anchors only (Morgan r=2, Tanimoto, self-exclude on potent-46 members).

These are **standalone predictors**, not corrections on top of the base.
Caruana decides whether to include them and at what weight. Success target:
either member achieves caruana weight ≥ 1% AND new caruana_bag20 OOF MAE
Δ ≤ -0.003 vs current 9-pool (above bag noise floor).

## Definitions

- **Morgan FP**: r=2, 2048 bit, uint8 (matches existing pipeline).
- **Tanimoto**: standard `popcount(A AND B) / popcount(A OR B)`.
- **sim²-weighted mean**: for query q with anchors `{a_i}` having pec50
  values `{y_i}` and similarities `{s_i}`:
  `pred(q) = sum_i (s_i² * y_i) / sum_i (s_i²)`
  - Avoids needing to choose k (low-sim anchors get near-zero weight automatically).
  - Power 2 standard for Tanimoto IDW.
- **potent-46**: train compounds with `pec50 ≥ 6 AND pec50 - counter_pec50 ≥ 1.5`
  (definition reused from `splits.py::analog_aware_split_indices`, expected count 46).

## Method

### Step 1 — Compute Morgan FPs
- All 4140 train + 513 test compounds, uint8 (4140+513, 2048).

### Step 2 — Compute pairwise Tanimoto
- For knn_alltrain: train↔train (4140×4140) and test→train (513×4140).
- For knn_potent46: anchor pool is the 46 potent-46 rows; train→potent-46 (4140×46) and test→potent-46 (513×46).

### Step 3 — OOF prediction (knn_alltrain_umap)
- 5-fold UMAP CV (canonical: Morgan+Jaccard, k=50, seed=42).
- For each fold f: predict fold-val rows using sim²-weighted mean over the (k-1) folds' train rows. **No self-exclude needed** within a fold (val rows are not in fold-train).
- For full-train test prediction: predict each test row using sim²-weighted mean over ALL 4140 train rows (no self-exclude — test ∉ train).

### Step 4 — OOF prediction (knn_potent46_umap)
- 5-fold UMAP CV (canonical, same folds as Step 3 for consistency).
- For each fold f: predict fold-val rows using sim²-weighted mean over the **fold-train ∩ potent-46** subset.
  - This is critical: if a fold's val happens to contain potent-46 members, their corresponding anchors are excluded by the fold split (potent member is in val, not in fold-train). For non-potent val rows, all 46 potent-46 anchors that happen to be in fold-train are used.
  - Practical effect: for any fold, anchor pool = potent-46 ∩ fold-train ≈ 36-37 anchors (46 × 0.8).
- Self-exclude: a val row that IS a potent-46 member predicts using the fold-train anchors (which exclude it by definition).
- For test prediction: anchor pool = ALL 46 potent-46 rows (test is blind, none of test is in potent-46).

### Step 5 — DB recording
For each member, call `evaluate.record_experiment` with:
- `name`: `knn_alltrain_umap` / `knn_potent46_umap`
- `model_type`: `"knn"`
- `feature_set`: `"morgan_r2_2048_tanimoto_sim_squared"`
- `hyperparameters`: `{"weight_power": 2, "anchor_pool": "alltrain"|"potent46", "self_exclude": true}`
- `fold_metrics`: per-fold MAE, RMSE, R², Spearman, Kendall (5 entries)
- `submission_path`: `null` (this is a pool member, not a final submission)
- OOF rows written to `experiment_oof_predictions` (one row per train compound)
- `on_conflict_replace=True` (idempotent re-runs)

Test predictions written to a separate location for caruana to consume:
- `track1_activity/data/oof/knn_alltrain_umap_test.parquet`
- `track1_activity/data/oof/knn_potent46_umap_test.parquet`

(Need to check if existing pool members have a similar test-pred storage convention; if so follow it.)

### Step 6 — Caruana bake-off
- Add both members to `ENSEMBLE_MODELS` in `run_ensemble.py` (11 total).
- Run 4 caruana_bag20 variants for comparison:
  - vanilla 9-pool (baseline)
  - +knn_alltrain (10 members)
  - +knn_potent46 (10 members)
  - +both (11 members)
- For each variant: caruana_bag20 OOF MAE / Sp / member weights printed.
- Decision: pick the variant with best OOF MAE Δ AND Sp not regressed > 0.005 AND chemprop family share in 0.65-0.80 band.

### Step 7 — Gate / submit
Same gate as recent submissions:
- caruana OOF MAE Δ ≤ -0.003 (above bag noise floor)
- caruana OOF Sp Δ ≥ -0.005
- chemprop family share in 0.65-0.80 (`project_family_share_lb_u_curve`)
- new member weight ≥ 0.01 (proves caruana actually used it)

If pass: re-run calibration (importance affine), submit to LB after cooldown.
If fail: defensive resubmit of id=43 baseline.

## Risks

1. **knn_potent46 prediction range collapse**: anchor pec50 range is [6.0, 6.86], so all predictions fall in that band. For non-analog test compounds (low nn_tanimoto), the prediction is biased high (~6.26 mean). This is by design — caruana sees a member that's systematically biased and orthogonal to other members. Risk: caruana gives it large weight thinking it's "strong" on potent compounds, but it overshoots non-potent compounds. Mitigation: caruana_bag20 averaging + the gate's "weight ≥ 1%" floor catches degenerate weights.
2. **knn_alltrain redundancy with existing TabPFN members**: TabPFN already does similarity-based regression internally. The kNN member may correlate r > 0.95 with `tabpfn_*` members and contribute nothing. If caruana weight = 0, declare null.
3. **Self-exclude correctness**: tested extensively in PR #151 (proximity calibrator). Reuse same numpy bitwise + index broadcasting pattern.
4. **OOF/LB reverse amplification (yet again)**: documented family of failures. If gate passes, LB A/B is still required. Submit during free cooldown window.

## Non-goals

- **NOT trying k=N hard cutoff variants** — sim²-weighted mean with full pool is mathematically equivalent for k large enough that distant anchors get negligible weight. v2 may sweep k.
- **NOT trying other weight powers** (sim¹, sim⁴) for v1 — sim² is the standard IDW exponent.
- **NOT trying other distance metrics** (Dice, cosine on dense FPs, MACCS) — Morgan+Tanimoto is the canonical pipeline.
- **NOT trying mixed anchor pools** (e.g. potent-46 + top-200 by activity) — keep the two members orthogonal in their anchor selection.
- **NOT touching base calibrator** — importance affine remains.
- **NOT a residual / correction** — these are standalone members feeding caruana.

## Deliverables

- `track1_activity/scripts/run_knn_pool_member.py` (~300 lines)
- 2 new experiments in DB (`knn_alltrain_umap`, `knn_potent46_umap`)
- 2 new test prediction parquets
- `run_ensemble.py` edit: add 2 names to `ENSEMBLE_MODELS`
- 4 caruana_bag20 bakeoff runs (manual sequential or scripted)
- New submission CSV (if gate passes)
- Console report:
  - per-member single OOF MAE / Sp
  - per-member residual r against existing 9 pool members
  - 4-variant caruana_bag20 weights + OOF MAE / Sp
  - gate pass/fail summary

## Open questions (revisit if v1 succeeds)

- k cutoff sweep (k ∈ {5, 10, 20, all})
- weight power sweep (p ∈ {1, 2, 4})
- IDW vs Gaussian kernel weighting
- mixing anchor pools (top-N by activity, by selectivity)
- combined with proposal #3 (rank-preserving correction) — but only if proposal #3 is independently revisited

# potent-46 Proximity-Gated Calibrator

Date: 2026-04-29
Status: Approved (brainstorming)
Owner: N283T

## Background

Track 1 LB rank 2 (N283T 0.4075 / Sp 0.8470, gap to sia +0.0008 MAE / -0.0044 Sp).
Pool of 9 caruana_bag20 members + global importance-weighted affine calibrator
(Morgan-FP train-vs-test density ratio, slope clipped to [1/3, 3]) is saturated:
new pool members hit a hard +/-0.002 OOF ceiling and reverse-amplify at LB.

Codex consultation (2026-04-29) recommended exploiting the structural fact that
**48.9% of the test set is NN to potent-46** (train pEC50 >= 6, sel >= 1.5, 45x
random enrichment). The current pipeline absorbs this only globally through
calibration / importance weighting; the analog-prior is not exploited
*locally* in the calibrator. Three sequential proposals:

1. **potent-46 proximity-gated calibrator** (this spec)
2. anchor residual model
3. rank-preserving MAE correction

This design covers proposal 1 only.

## Goal

Replace the single global affine in `run_ensemble_calibrate_importance.py`
with a **two-stratum** affine: train and test compounds are split by their
NN-Tanimoto distance to potent-46, and a separate weighted affine is fit per
stratum. The intuition is that the analog-rich (`near`) stratum behaves
differently from the chemotype-distant (`far`) stratum, and a single global
slope/intercept compromises both.

Success target: LB MAE Δ <= -0.0008 (close gap to sia) without Sp regress
greater than -0.005 (preserve current Sp = 0.8470 baseline within bag noise).

## Definitions

- **potent-46 set**: train compounds with `pec50 >= 6 AND selectivity >= 1.5`,
  where `selectivity = train_pec50 - counter_pec50` (NaN where the train
  compound has no `counter_assay` row → excluded from potent set). Computed
  inline (no stored column); matches the definition in `splits.py::analog_aware_split_indices`.
  Expected size = 46 (per CLAUDE.md memory
  `project_test_is_45x_enriched_in_potent46_analogs`).
- **NN-Tanimoto proximity** for a query compound q:
  `prox(q) = max_{p in potent-46, p != q} Tanimoto(FP(q), FP(p))`
  - FP = Morgan r=2, 2048 bit (matches existing importance calibrator)
  - For train compounds in potent-46, exclude self (avoid the self-match leak
    documented in `feedback_self_match_leak_in_similarity_features`)
  - For test compounds, no self-exclude needed
- **threshold T**: median of `prox(test)` over the 513 test compounds. Fixed,
  not OOF-tuned. (Avoids overfitting a small-sample optimum; Codex case for
  "local version of importance calibrator" is the simplest faithful read.)
- **strata**:
  - `near` = `prox >= T`
  - `far`  = `prox < T`
  - By construction test splits ~50/50; train split depends on its own NN
    distribution to potent-46 and may be unbalanced (potent-46 members concentrate
    in near-train).

## Method

### Inputs (unchanged from importance calibrator)
- caruana_bag20 OOF predictions for 4140 train compounds (reconstructed from
  member OOF + stored weights, exactly as in `run_ensemble_calibrate_importance.py`)
- caruana_bag20 test predictions for 513 test compounds (from
  `submissions/ens_caruana_bag20.csv`)
- y_train pEC50 (from `train_activity`)
- Morgan r=2 2048-bit fingerprints for 4140 train + 513 test (computed inline)

### Steps

1. Load OOF + test predictions + train labels (reuse existing helpers).
2. Compute Morgan FPs for train, test (existing helper).
3. Identify potent-46 set from DB query.
4. Compute `prox(train)` and `prox(test)` against potent-46 (excluding self
   for any train compound that is itself a potent-46 member).
5. T = median(prox(test)).
6. Stratify train and test into `near` / `far`.
7. For each stratum independently:
   - Compute density-ratio sample weights via the existing Morgan-FP domain
     classifier (LogisticRegression on the **same** stratum's train + test FPs).
     Clip to [1/3, 3]. Renormalise to sum to N_stratum.
   - Fit weighted `LinearRegression` on (oof_pred, y_train) with sample_weight.
   - Apply (slope, intercept) to that stratum's test predictions.
8. Report per-stratum + global OOF MAE / Spearman, deltas vs raw OOF and vs
   the global importance calibrator.
9. Write submission `ens_caruana_bag20_calibrated_proximity.csv`.

### Domain classifier scope (intentional choice)

The domain classifier is fit **per stratum** (not globally then masked) so the
sample weights reflect train-vs-test shift *within* that stratum's chemotype
neighborhood. This is the local analogue of the global importance recipe.

Alternative considered: fit global classifier once, take per-stratum slices.
Risk: the global classifier may not separate near-train / near-test well if
they're chemically similar; per-stratum fit is the conservative read of
"local importance calibrator".

### Self-exclude implementation

When computing `prox(q)` for a train compound q:
- if q in potent-46: compute Tanimoto against potent-46 \ {q}
- else: compute Tanimoto against full potent-46

For test compounds: always full potent-46 (test is blind, can't be in potent-46).

## Validation

### OOF gate (must pass before submitting)

Compare against:
- `raw OOF MAE` (un-calibrated caruana_bag20)
- `global importance OOF MAE` (current production calibrator)

Pass criteria for LB submission:
- `proximity OOF MAE` <= `global importance OOF MAE` (no regress)
- `proximity OOF Spearman` >= `global importance OOF Spearman - 0.005`
- per-stratum sanity: neither stratum's calibrated MAE worse than raw by > 0.01
- min stratum size: each train stratum >= 200 compounds (regress check)

If gate fails: report and abandon (this submission slot). Do not LB-submit.

### LB A/B (mandatory regardless of OOF)

Per `feedback_oof_lb_reverse_amplification` and the May calibrator drama:
calibrator family changes are NOT reliably predicted by OOF. Even if OOF gate
passes, LB A/B is required. Submit during a free 4h cooldown window. Compare
against id=43 (current best, 0.4075 / Sp 0.8470).

Decision rule:
- LB MAE Δ <= -0.0005 AND Sp Δ >= -0.005: keep, mark as new prod calibrator
- LB MAE Δ in [-0.0005, +0.0005]: hold, treat as bag noise
- LB MAE Δ > +0.0005: revert to global importance calibrator

## Non-goals

- **NOT touching the 9-pool composition** — calibrator-only change keeps blast
  radius small.
- **NOT tuning T via OOF** — fixed at test-median to avoid small-sample overfit
  (513 test, single split point would be noisy).
- **NOT > 2 bins** — defer to v2 if v1 shows directional signal.
- **NOT continuous gating (sigmoid)** — defer to v2; binary first for
  interpretability and clean LB read.
- **NOT changing FP type or radius** — match existing importance calibrator
  exactly so the only delta is stratification.

## Risks

1. **Self-match leak resurgence** — guarded by explicit self-exclude in step 4.
2. **Per-stratum domain classifier overfitting** — small near-train (~50%)
   may give noisy density-ratio. Mitigation: same [1/3, 3] clip as global.
3. **OOF -> LB reverse amplification** — calibrator changes have a documented
   history of OOF/LB sign flips (`feedback_oof_lb_reverse_amplification`).
   LB A/B is mandatory.
4. **Threshold = test-median assumes test stratification matches train semantics** —
   if near-test contains many never-seen chemotypes, near-train calibrator may
   not generalise. Surface per-stratum train sizes in the report.
5. **Family share unaffected** — calibrator-only change, no pool composition
   delta, so the U-curve constraint (`project_family_share_lb_u_curve`) does
   not apply.

## Deliverables

- `track1_activity/scripts/run_ensemble_calibrate_proximity.py` (new, ~250
  lines, structurally derived from `run_ensemble_calibrate_importance.py`)
- `track1_activity/submissions/ens_caruana_bag20_calibrated_proximity.csv`
  (gitignored)
- Console report:
  - potent-46 size
  - threshold T
  - train near/far counts, test near/far counts
  - per-stratum slope, intercept
  - per-stratum OOF MAE / Spearman (raw and calibrated)
  - global OOF MAE / Spearman vs raw, vs global importance baseline
  - explicit gate pass/fail summary

## Open questions (to revisit if v1 succeeds)

- continuous gating (sigmoid on prox) instead of hard 2-bin
- 3+ strata (e.g. quantile-based)
- T tuning (nested CV, with stronger guardrails)
- per-stratum calibrator family (spline / isotonic / linear_pos in addition to affine)
- combine with anchor residual (proposal 2)

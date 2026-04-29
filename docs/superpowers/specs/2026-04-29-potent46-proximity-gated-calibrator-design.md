# potent-46 Proximity-Gated Calibrator

Date: 2026-04-29
Status: v2 — asymmetric design (post-Codex re-consult after T=test-median early-exit)
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
- **threshold T = 0.28** (fixed chemical anchor, weak-analog Tanimoto floor;
  derived from data sweep, see "Why T=0.28" below). Not OOF-tuned. v1 used
  T = `median(prox(test))` = 0.4375, which produced only 18 train-near rows
  and triggered the `MIN_STRATUM_TRAIN=200` early-exit. Post-Codex re-consult
  confirmed test is "shifted re-sample of potent-46 region", not "extreme of
  train distribution", so a chemical anchor is the right T-source, not a
  test-side quantile.
- **strata** (asymmetric — near gets local treatment, far falls back to global):
  - `near` = `prox >= T`: per-stratum local importance calibrator (the
    `fit_stratum_calibrator` from v1)
  - `far`  = `prox < T`: applies the existing global importance calibrator
    (fit on all train + all test, density-ratio weighted, slope-clipped). No
    far-stratum-specific fit. Codex rationale: "test が非対称に作られている
    ので、補正も非対称でよい".
  - Counts at T=0.28: train near = 338, train far = 3802, test near = 322,
    test far = 191.

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
5. T = 0.28 (fixed chemical anchor).
6. Stratify train and test into `near` / `far`.
7. **Near branch**: fit one weighted affine on `near-train` only.
   - Density-ratio sample weights via LogisticRegression on near-train +
     near-test FPs only (stratum-local domain classifier).
   - Clip to [1/3, 3]. Renormalise.
   - Fit `LinearRegression(oof_near, y_near, sample_weight=w_near)`.
8. **Far branch (global fallback)**: fit the existing global importance
   affine on ALL train + ALL test (the v1 `run_ensemble_calibrate_importance.py`
   recipe, re-fit inline). Apply this single (slope, intercept) to far rows
   on both train (for OOF) and test (for submission).
9. Apply: test near rows → near affine; test far rows → global affine.
10. Report per-stratum + global OOF MAE / Spearman, deltas vs raw OOF and vs
    the pure-global-importance OOF (computed inline in main).
11. Write submission `ens_caruana_bag20_calibrated_proximity.csv`.

### Why T=0.28

Train and test NN-Tanimoto distributions barely overlap:
- train: q75=0.24, q90=0.27, q95=0.29, q98=0.33, q99=0.37
- test:  median=0.44, q25=0.24

T=0.28 is the data-driven sweet spot of Codex's two guidelines ("chemical
anchor in 0.25-0.40 range" and "train near 200-600"):
- T=0.25 → train near 863 (too large; "local" loses meaning)
- **T=0.28 → train near 338, test near 322, test far 191** (sweet spot)
- T=0.30 → train near 183 (just below MIN_STRATUM_TRAIN=200 floor)
- T=0.33 → train near 87 (too few for stable LogReg + LinReg fit)

### Domain classifier scope (intentional choice)

The **near** domain classifier is fit on near-train + near-test rows only
(stratum-local). This reflects train-vs-test shift within the analog
neighborhood, not the global library.

The **far** branch reuses the global importance classifier (fit on all
train + all test). This is asymmetric by design: test is the "shifted
re-sample" and the analog region warrants special handling, but the rest
of test is well-served by the existing production calibrator.

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
- near stratum sanity: near's calibrated MAE not worse than raw by > 0.01
  (far is the global fallback; sanity already implicit in baseline parity)
- min stratum size: train near >= 200 (regress check; far always large
  since most train is far at T=0.28)

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
- **NOT tuning T via OOF** — fixed at chemical anchor 0.28 to avoid
  small-sample overfit and to preserve chemical interpretability.
- **NOT > 2 bins** — defer to follow-up if v1 shows directional signal.
- **NOT continuous gating (sigmoid)** — defer; binary first for interpretability.
- **NOT a far-stratum-specific fit** — far falls back to the existing global
  importance calibrator. Asymmetric by Codex's recommendation.
- **NOT changing FP type or radius** — match existing importance calibrator
  exactly so the only delta is stratification.

## Risks

1. **Self-match leak resurgence** — guarded by explicit self-exclude in step 4.
2. **Per-stratum domain classifier overfitting** — small near-train (~50%)
   may give noisy density-ratio. Mitigation: same [1/3, 3] clip as global.
3. **OOF -> LB reverse amplification** — calibrator changes have a documented
   history of OOF/LB sign flips (`feedback_oof_lb_reverse_amplification`).
   LB A/B is mandatory.
4. **Train near (n=338) is small relative to test near (n=322)** — domain
   classifier on 338 + 322 (= 660) rows could overfit. Mitigation: same
   [1/3, 3] clip; LogReg `C=1.0` already provides L2.
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
  - threshold T (constant 0.28)
  - train near/far counts, test near/far counts
  - near slope/intercept, global affine slope/intercept (used for far)
  - per-stratum OOF MAE (raw and calibrated)
  - global OOF MAE / Spearman vs raw, vs global importance baseline
  - explicit gate pass/fail summary

## Open questions (to revisit if v1 succeeds)

- continuous gating (sigmoid on prox) instead of hard 2-bin
- 3+ strata (e.g. quantile-based)
- T tuning (nested CV, with stronger guardrails)
- per-stratum calibrator family (spline / isotonic / linear_pos in addition to affine)
- combine with anchor residual (proposal 2)

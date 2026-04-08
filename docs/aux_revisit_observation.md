# Auxiliary data revisit — observation only (issue followup to PR #41 disaster)

**Branch**: `feature/aux-revisit-observation`
**Status**: observation only — no model changes, no DB writes
**Script**: `track1_activity/scripts/eda_aux_revisit.py`

## Why we're re-looking

After PR #41 (`mordred_singleconc` direct concat) blew up the LB from 0.622 → 1.027, we confirmed that `counter_assay` and `single_concentration` cannot be used as **direct test features** because **0 of 513 test compounds** appear in either table. Test compounds were ordered separately during analog expansion (Octant blog: step 5 "Analog Expansion"), so they were never put through the primary screen or the counter assay.

This document re-examines the aux data with a usability-for-test lens before we burn another submission slot.

## Setup

| Table | Rows | Test ∩ this table |
|---|---:|---:|
| `train_activity` | 4,140 | — |
| `test_activity` | 513 | — |
| `counter_assay` | 2,860 | **0** |
| `single_concentration` (all conc) | 21,014 | **0** |
| `single_concentration` @ 8.25e-6 M | 10,753 | **0** |

Octant primary screen concentrations: 9.8e-7, 8.25e-6, 3.3e-5, 9.9e-5 M (the blog mentions 10/30 µM as the main pair, with a 100 µM pilot and a 1 µM follow-up).

## Part 1 — Counter-selectivity (Δ = pec50_PXR − pec50_counter)

`train ∩ counter_assay` n=2,860. Octant defines a "true PXR agonist" as Δ ≥ 1.5 logunits — that exact rule was used to pick the 63 hits from which the test set was expanded.

Overall: **50.9% (1,456/2,860)** of train-with-counter compounds clear that bar.

### Selectivity rate climbs with potency

| pec50 bin | n | n_selective | mean Δ | rate |
|---|---:|---:|---:|---:|
| (0, 4.0]   | 306 |  29 | 0.16 | 9.5% |
| (4.0, 4.5] | 381 | 136 | 1.00 | 35.7% |
| (4.5, 5.0] | 894 | 444 | 1.43 | 49.7% |
| (5.0, 5.5] | 923 | 580 | 2.07 | 62.8% |
| (5.5, 6.0] | 294 | 222 | 2.86 | 75.5% |
| (6.0, 6.5] |  54 |  39 | 3.20 | 72.2% |
| (6.5, 7.0] |   8 |   6 | 4.22 | 75.0% |

Reading: a low-potency compound (pec50 ~4) only has a 1-in-3 chance of being PXR-selective, while a hit (pec50 ≥ 5.5) is selective ~75% of the time. The test set is the **expansion around the 63 selective hits**, so the test distribution is biased toward both higher pec50 *and* higher selectivity.

**Implication for modeling**: a sample-weight scheme that up-weights selective train compounds (or down-weights non-selective ones) shifts the model toward the regime test actually lives in. For the 1,280 train compounds without counter data we have to default to "unknown" — most of those are low-pec50 anyway (see Part 3).

## Part 2 — Single-conc → DRC consistency

`train ∩ single_conc @ 8.25e-6 M` n=2,374. Pearson r(log2fc, pec50) = **0.7237** (matches prior EDA).

Linear fit: `pec50 ≈ 1.198 × log2fc + 3.856`. Residual quantiles: |q90| = 0.787, |q95| = 1.190 → 238 compounds beyond q90.

### Caveat: the noisy top-20 are not noise, they are an artifact

Looking at the top-20 by |residual|, every single one has `cohens_d ≈ 19.3` or `19.79` — these are **saturated values from a near-zero stderr**, not legitimate effect sizes. They are compounds with `log2fc ≈ 0` (flat at primary screen), low `pec50` (1.7-2.0), but cohens_d looks "huge" because the control plate noise was tiny.

This means the simple `|residual|` proxy is dominated by compounds that **were correctly inactive in the primary screen**. They are not bad train rows. Trying to down-weight them would actually delete the cleanest negatives from the dataset.

**Implication**: data-cleaning via single-conc consistency needs a smarter signal than a residual-from-linear-fit. Possible refinements:
- Require `|residual| > q90` AND `p_value < 0.05` (real activity disagreeing with DRC)
- Require both single-conc rows (8.25 µM AND 33 µM) to disagree consistently
- Use Bayesian DRC posterior uncertainty (`pec50_std_error`) directly instead of single-conc residuals

For now, mark this lane as "not as easy as it looked" and do not implement any reweighting.

## Part 3 — 2×2 subset decomposition (the most useful finding)

| has_counter | has_single | n | pec50 mean | pec50 std | pec50 median |
|---|---|---:|---:|---:|---:|
| False | False | **1,248** | **3.33** | 0.95 | 3.30 |
| False | True  |    32 | 3.48 | 1.38 | 3.30 |
| True  | False |   518 | 4.37 | 1.22 | 4.81 |
| True  | True  | **2,342** | **4.85** | 0.76 | 4.95 |

### Two distinct populations live in train_activity

- **"Neither" subset (n=1,248, ~30% of train)**: mean pec50 ≈ 3.33. These are compounds that were profiled in DRC mode but never went through the primary screen *and* never got a counter assay. Their pec50 distribution is dominated by inactives (median 3.3 = the assay floor). They are essentially the *low-activity training tail*.
- **"Both" subset (n=2,342, ~57% of train)**: mean pec50 ≈ 4.85, std 0.76. These are compounds that went through the full assay flow — primary screen → DRC → counter screen. Their distribution is much tighter and centered at moderate activity.

### Test set positioning

The test set comes from analog expansion around 63 *potent and selective* hits (pec50 ≥ 6 + Δ ≥ 1.5). So test compounds should look like:
- **chemically related to the "both" subset's hit tail** (not the "neither" subset)
- **distributionally probably wider** than the "both" subset, because analog series sweep activity from 0 to 7

We do not know the true test pec50 distribution, but the LB best-known RAE of ~0.62 with predictions in the 4.5-5.0 range suggests the test mean is somewhere around there, much closer to the "both" subset than the "neither" subset.

**Implication**: the 1,248 "neither" rows may be **distributionally pulling the model toward inactives**. They are likely a different chemotype set (the late additions Octant mentions in the blog: "an additional diversity library" outside the primary screen). Three things to try:
1. Train-only-on-"both" baseline (n=2,342) and compare OOF on the same UMAP folds
2. Stratified sample weighting: weight ∝ 1 if in "both", < 1 if in "neither"
3. Include "neither" but add a binary `is_screened` feature so the model can learn the bias internally

Option 3 is leakage-safe (test would all get `is_screened=False`, identical to the "neither" train rows — same distribution match as the bias correction).

## Figures

- `docs/figures/aux_revisit_part1_selectivity.png` — Δ histogram + scatter
- `docs/figures/aux_revisit_part2_consistency.png` — log2fc vs pec50 + residuals
- `docs/figures/aux_revisit_part3_subsets.png` — pec50 boxplots per 2×2 cell

## Decision points (no code changes from this PR)

1. **Selectivity sample weighting** — promising. Direct alignment with the test set selection rule. Risk: 1,280 train rows have no counter data and need a default. Mitigation: use Δ ≥ 1.5 weight=1.5, no_counter weight=1.0, Δ < 1.5 weight=0.7.
2. **Single-conc data cleaning** — parked. The naive residual signal is contaminated by artifact cohens_d. Needs smarter filter design before it is worth trying.
3. **2×2 subset reweighting / `is_screened` flag** — most novel. The "neither" subset is qualitatively different from the rest of train and likely far from test. **Try this first** because it has the smallest implementation cost (one binary feature) and a clear distributional rationale.
4. **Counter pec50 as a multi-task aux (revisit)** — already tried in PR #42 with aux_weight ∈ {0.02, 0.05, 0.1}, all worse than tuned single-task. Not promising.

## Next concrete experiment (proposed)

Add `is_screened` (binary: has_counter OR has_single @ 8.25 µM) as a feature to the LightGBM Mordred baseline. Train + OOF on the existing UMAP folds. Test compounds get `is_screened=False`. Expect: small but real OOF improvement, no LB downside because no missing branches at inference (the feature is well-defined for every test compound).

If that works, layer on the selectivity sample weighting in a second experiment.

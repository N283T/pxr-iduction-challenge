# Track 1: CV Bake-off — scaffold vs UMAP vs analog-aware

**Branch**: `feature/analog-aware-split`
**Date**: 2026-04-17
**Predecessor**: `docs/track1_cv_prep.md` (PR #68). This document takes
the hypotheses from that EDA and actually runs the bake-off.

## TL;DR

Across two feature sets (`rdkit_desc_full`, `mordred_jazzy`), the same
LightGBM default-params model evaluated under three CV splits shows a
consistent pattern:

- **OOF MAE**: scaffold > UMAP > analog (analog is lowest)
- **OOF RAE**: scaffold < UMAP < analog (analog is highest)

The analog MAE is dead-on LB MAE for `rdkit_desc_full` (0.494 vs
0.499) and within 0.03 for `mordred_jazzy`. The analog RAE
**overshoots** LB RAE by ~0.04-0.08 — conservative, safe for tuning.
UMAP does the opposite: MAE ~0.04 above LB MAE, RAE ~0.03-0.04 below
LB RAE — **optimistic**, unsafe for tuning.

**Primary conclusion**: for selecting models whose gains transfer to
LB, **MAE on analog-aware CV** is the most reliable OOF signal. This
matches the theoretical prediction from `docs/track1_cv_prep.md`
(OOF->LB gap is mostly RAE denominator, not prediction degradation).

## Bake-off results

All numbers are 5-fold OOF, LightGBM default params, same feature
matrices across splits. Source: `experiments.rae_mean`, `.mae_mean`.

| Feature | Split | OOF RAE | OOF MAE | vs LB RAE | vs LB MAE |
|---|---|---:|---:|---:|---:|
| rdkit_desc_full | scaffold | 0.5813 | 0.5249 | -0.045 | +0.026 |
| rdkit_desc_full | umap     | 0.5980 | 0.5388 | -0.028 | +0.040 |
| **rdkit_desc_full** | **analog** | **0.7053** | **0.4942** | **+0.079** | **-0.005** |
| mordred_jazzy   | scaffold | 0.5640 | 0.5091 | -0.062 | +0.010 |
| mordred_jazzy   | umap     | 0.5856 | 0.5280 | -0.040 | +0.029 |
| **mordred_jazzy**   | **analog** | **0.6660** | **0.4671** | **+0.040** | **-0.032** |

LB reference: `ens_l2_a0.1` at RAE 0.6263 / MAE 0.4989 (rank 17).

### MAE pattern (the important signal)

Across both features, **analog MAE brackets LB MAE from below**. For
`rdkit_desc_full` the gap is 0.005 — practically identical. For
`mordred_jazzy` the model predicts analog val compounds slightly
_better_ than LB (-0.032); this is likely because val analogs are
closer to train potents than LB's actually-further diverse tail.

Scaffold and UMAP MAE are consistently _higher_ than LB MAE (by
0.01-0.04). They penalise the model for holding out compounds that
aren't structurally similar to LB's test-generation mode.

### RAE pattern (the denominator inflation)

Analog RAE is **above** LB RAE for both features. This is the
expected behaviour given the narrow val y distribution — even with
nearly-perfect prediction accuracy (MAE=0.467 for mordred_jazzy),
the RAE denominator `E|y - median(y)|` collapses so RAE inflates.
It's a conservative, safe OOF signal: "if analog RAE is X, LB RAE
is likely <=X".

UMAP and scaffold RAE are _below_ LB RAE. They're optimistic: "OOF
RAE is X, but LB RAE is likely >X". This is the exact bias that
produced the 0.095 OOF->LB gap we documented in `track1_cv_prep.md`.

### Why analog split behaves this way (recap)

1. Potent-46 seeds (train compounds with pEC50>=6 AND sel>=1.5)
   are always in train, matching the LB structural premise.
2. Non-potent train compounds with Morgan NN Tanimoto >=0.25 to any
   potent seed are classified "analog" and distributed across folds.
3. Non-analogs (NN<0.25 to any potent) are always in train. They
   don't resemble LB's test generation mode and holding them out
   measures a different thing.

The resulting val subsets per fold are:
- Size ~170 compounds (roughly 1/3 of LB's 513)
- Potent mean ~=4.6 (vs train-wide ~=4.3)
- Abs-dev-from-median ~=0.68 (vs train-wide 0.86, LB ~=0.80)

Val is **structurally** and **distributionally** the closest we can
construct to LB test from a train-only split.

## Implementation notes

### Partial-coverage bug found and fixed

The existing `run_train.py` initialised `oof_preds = np.zeros(...)`
and only filled `va_idx` positions. For scaffold/UMAP this is fine
(every index appears in val exactly once), but analog-aware only
covers ~20.5% of train. The remaining 79.5% stayed at zero, which:
- Corrupted the "Overall OOF" metric (apparent RAE 3.8, MAE 3.5)
- Would have corrupted DB OOF rows if saved.

Fix (this PR):
- Added `oof_covered` boolean mask tracking which indices were
  predicted.
- Overall OOF metrics now computed on `y_train[oof_covered]` only.
- `save_oof_predictions` gained `covered_mask` parameter; when
  provided, only covered indices are persisted.

This is backward-compatible: scaffold/UMAP runs have
`covered_mask.all() == True` and behave identically to before.

### Analog threshold sensitivity

Analog pool size as a function of Tanimoto threshold to potent-46:

| threshold | analog pool | val per fold | val y-dispersion |
|---:|---:|---:|---:|
| 0.40 |  27 |  5-6 | 0.51 |
| 0.35 |  54 | 10-11 | 0.58 |
| 0.30 | 177 |  35  | 0.70 |
| **0.25** | **849** | **170** | **0.68** |
| 0.20 | 2744 | 549  | 0.72 |

Default is **0.25**: balances fold-noise (large enough val) against
analog specificity (val y-dispersion 0.68, close enough to LB's
0.80 without diluting into the general train tail). Available to
override via `--analog-threshold`.

## What this does NOT address

1. **Ensemble weight optimisation**. `run_ensemble.py` expects full
   4140-length OOF arrays per candidate model. Analog-split OOF
   covers only 849 indices, and the covered set can differ per
   seed/threshold. Combining analog-split models with UMAP-split
   models in the existing ensemble pipeline is a separate integration
   task. For now, analog CV is a **model-selection and tuning**
   signal; final submission should still come from a UMAP-split
   ensemble until that's unblocked.

2. **Diversity tail (~41.5% of LB)**. Analog CV measures only the
   58.5% close-analog regime. The NN<0.3 tail is a different
   distribution and a different failure mode. A mixed "analog +
   diversity" fold design is a plausible next iteration.

3. **Direct LB correlation verification**. We infer the analog CV is
   a better LB proxy from the MAE/RAE pattern, but confirmation
   requires tuning a model to analog CV and submitting. That's the
   next concrete experiment.

## Reproducers

- `track1_activity/src/splits.py` — `analog_aware_split_indices`
- `track1_activity/scripts/run_train.py` — `--split analog` +
  `--analog-threshold FLOAT`
- `track1_activity/scripts/eda_cv_prep/03_analog_split_sanity.py` —
  verifies disjointness, potents-always-train, val y-dispersion
- Example: `pixi run python track1_activity/scripts/run_train.py
  --model lgbm --feature mordred_jazzy --split analog --trials 0`

## Proposed next steps

1. **Tune an Optuna LightGBM on analog CV** with MAE as the
   objective. Compare resulting hyperparameters against
   `lgbm_mordred_jazzy_umap`. If they differ meaningfully, submit the
   analog-tuned version — the first direct test of "analog OOF ->
   better LB".
2. **Retrofit the ensemble pipeline** to tolerate partial-coverage
   OOF (join on covered indices, compute weighted averages only
   where all candidates have predictions). Unlocks analog-split
   models as ensemble members.
3. **Mixed analog + diversity fold design** to bring the 42% NN<0.3
   test regime under measurement.

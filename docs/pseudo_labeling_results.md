# Pseudo-labeling results (issue #35)

**Status**: Negative result — pseudo-labels hurt OOF RAE at every weight tested.

**Branch**: `feature/pseudo-labeling`
**Plan**: `docs/plans/pseudo-labeling-plan.md`
**Prior analysis**: `docs/auxiliary_data_analysis.md`

## Hypothesis

From the auxiliary-data EDA, `single_concentration.log2_fc_estimate` at 8.25e-6 M correlates with `train_activity.pec50` at r=0.724. We expected to:

1. Fit a mapping `single-conc features → pEC50` on the 2,374-compound overlap
2. Apply it to ~8,378 compounds that have single-conc data but no train label
3. Augment LightGBM training with these weakly-labeled compounds (lower sample weight)
4. Gain a meaningful OOF RAE improvement

## What was built

- `track1_activity/scripts/build_pseudo_labels.py` — LightGBM regression from single-conc features to pEC50. **OOF R²=0.76** (strong mapping).
- `track1_activity/src/pseudo_labels.py` + `run_train.py` extensions — fold-safe augmentation with `--pseudo PATH --pseudo-weight FLOAT` flags. Pseudo compounds never appear in validation folds (compound_id disjointness enforced).
- `track1_activity/scripts/run_train_pseudo_sweep.py` — sweep driver. Dedupes against DB.

## Results

LightGBM + Mordred + UMAP split, trials=0, default hyperparameters.

| Experiment | OOF RAE | Δ vs baseline |
|---|---:|---:|
| baseline (no pseudo) | **0.5817** | — |
| pseudo weight = 0.05 | 0.5923 | +0.0106 |
| pseudo weight = 0.10 | 0.5987 | +0.0170 |
| pseudo weight = 0.30 | 0.6169 | +0.0352 |
| pseudo weight = 0.50 | 0.6265 | +0.0448 |
| pseudo weight = 1.00 | 0.6355 | +0.0538 |

**Clean monotonic degradation.** Even at the smallest weight (0.05), pseudo-labels cost ~0.011 RAE. No point along the sweep recovers baseline.

## Why did it fail?

Despite the mapping having strong in-sample R²=0.76, augmenting with pseudo-labels consistently hurts. Candidate explanations (not yet verified):

1. **Distribution shift**. The 8,378 pseudo compounds may inhabit a different region of chemical space than the 4,140 real train compounds. UMAP-split CV clusters real train only, so augmenting training folds with out-of-cluster pseudo samples pulls the model toward a distribution the validation doesn't reward.

2. **Label noise propagation**. R²=0.76 means ~24% of pseudo-label variance is noise. That noise enters the loss proportionally to `pseudo_weight * confidence`, and LightGBM does not discount it beyond the weight. Even at weight=0.05, the cumulative mass of noise across 8,378 samples exceeds the signal of ~3,300 clean real samples per fold.

3. **Range collapse**. Pseudo labels are predictions of a model fit on the overlap. They inherit the regression-toward-the-mean property — pseudo pEC50 mean=3.87, std=0.66, vs real train mean=4.74, std=0.88. The compressed range shifts the predicted distribution downward and tightens it.

4. **UMAP-CV versus leaderboard**. CV RAE 0.58 vs LB RAE 0.62 is a known 0.04 gap. It is *possible* (not tested) that pseudo-labels help LB even while hurting UMAP CV, because LB test is more out-of-distribution. Testing this requires an LB submission — not done in this PR.

## What's shipped anyway

The infrastructure is kept on the branch behind opt-in flags:
- Baseline path is **byte-for-byte unchanged** when `--pseudo` is not passed (verified in final review)
- All checks remain in place (R²≥0.45 gate, fold-safety assertion, column alignment)
- Pseudo-labels support can be re-activated in future experiments with minimal code changes

## Follow-up ideas

Ordered by a priori plausibility:

### 🥇 Use single-conc features as direct inputs, not as labels

Instead of pseudo-labeling, concatenate single-conc features (log2fc @ 8.25e-6, 3.30e-5, cohens_d, stderr, n_concs) to the Mordred feature matrix. For compounds lacking single-conc data, use NaN — LightGBM handles this natively. This injects the r=0.72 signal **directly** at feature-level without introducing label noise.

This is the most promising redirect from this experiment. Issue to open.

### 🥈 Confidence-thresholded subset

Filter pseudo labels to the top confidence quantile (e.g., confidence ≥ 0.95, ~4,000 compounds). This reduces noise but also reduces augmentation. Could be tried with a `--pseudo-min-confidence` flag on the existing infrastructure.

### 🥉 Submit a pseudo run to LB anyway

Pick `pseudo0.05` (least damaging to CV) and submit. If LB improves despite CV regression, it confirms the CV-LB distribution mismatch and opens a new lane: "models tuned on OOF CV are too conservative for LB." High information value for one submission slot.

### Deferred

- Residual learning on pseudo predictions
- Multi-task (issue #36) — independent approach, probably better target for counter_assay data

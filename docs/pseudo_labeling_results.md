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

Despite the mapping having strong in-sample R²=0.76, augmenting with pseudo-labels consistently hurts. We initially blamed feature-space distribution shift, but a follow-up analysis (`track1_activity/scripts/eda_pseudo_distribution_shift.py`, merged separately) provided mixed evidence and a better explanation.

**Distribution shift evidence (mixed)**:

| | median nearest-train distance (UMAP-10D) |
|---|---:|
| train→train | 0.190 |
| test→train | 0.350 |
| pseudo→train | 0.427 |

Per-point, pseudo IS farther from train than test is. BUT at the cluster level (50 KMeans clusters, same params as `splits.umap_split_indices`):

- cos(train_share, pseudo_share) = 0.890, JS = 0.035 (close)
- cos(train_share, test_share) = 0.651, JS = 0.152 (more divergent)

So pseudo compounds occupy **the same coarse chemotypes as train**, just at the edges of those clusters (intra-cluster sparsity), while test contains genuinely novel chemotypes. Feature-space OOD is therefore NOT the dominant cause.

**The real dominant cause: label-space compression.**

- pseudo_pec50: mean = 3.87 ± 0.66
- train pec50: mean = 4.74 ± 0.88

The pseudo distribution is shifted down by 0.87 and is narrower by ~25%. This is regression-toward-the-mean in the inner mapping model — the LightGBM regressor on 2,374 overlap compounds collapses extremes toward the population mean. Augmenting training folds with these compressed labels pulls predictions downward in proportion to `pseudo_weight × num_pseudo`, which exactly matches the linear OOF degradation observed across weights {0.05 → 1.0}.

**Other contributors** (lesser):

- **Label noise propagation**: R²=0.76 means ~24% of pseudo-label variance is noise, entering the loss proportionally to `pseudo_weight × confidence`. Concentrated in the same direction as the mean shift, so it amplifies the bias rather than averaging out.
- **UMAP-CV vs leaderboard distribution**: CV RAE 0.58 vs LB RAE 0.62 is a known 0.04 gap. It is *possible* (not tested) that pseudo-labels help LB even while hurting UMAP CV; testing requires an LB submission, not done in this PR.

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

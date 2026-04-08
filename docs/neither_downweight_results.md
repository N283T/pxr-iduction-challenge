# Neither-subset down-weight sweep — negative result

**Branch**: `feature/neither-downweight`
**Status**: **Negative**. Down-weighting the "neither" subset always hurts OOF. Aux-data utilization parked for now.
**Script**: `track1_activity/scripts/run_neither_downweight_sweep.py`
**Predecessor**: `docs/aux_revisit_observation.md`

## Hypothesis (from observation PR #43)

`train_activity` contains 1,248 "neither" compounds (no `counter_assay` row, no `single_concentration` row), with pec50 mean 3.33 — much lower than the 2,342 "both" compounds (pec50 mean 4.85). Test compounds come from analog expansion of potent hits and should be distributionally closer to "both". Therefore: down-weighting "neither" compounds during training should align the model with the test distribution and improve OOF.

## Setup

- Model: LightGBM with the published baseline params (matches `lgbm_mordred+morgan_r2`, OOF RAE ~0.553 in DB)
- Features: Mordred only (1,460-ish columns), NaN→0
- Split: UMAP fold (n_splits=5, n_clusters=50, seed=42)
- Sample weight applied only to the 1,248 "neither" rows; all other rows weight=1.0
- 6-point sweep over neither_weight ∈ {0.0, 0.1, 0.3, 0.5, 0.7, 1.0}

Leakage safety: the "neither" assignment is used solely to construct training-time `sample_weight`. No aux features enter the model, no test-time dependence on aux data, no missing-feature branches. This is the structural opposite of PR #41.

## Results

| neither weight | OOF RAE | OOF MAE | OOF R² | both RAE | neither RAE | Δ vs baseline |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.6625 | 0.5970 | 0.4760 | — | — | **+0.0807** |
| 0.10 | 0.5928 | 0.5392 | 0.5570 | — | — | +0.0110 |
| 0.30 | 0.5861 | 0.5333 | 0.5667 | — | — | +0.0043 |
| 0.50 | 0.5853 | 0.5326 | 0.5682 | 0.857 | 0.707 | +0.0035 |
| 0.70 | 0.5890 | 0.5359 | 0.5660 | 0.869 | 0.700 | +0.0072 |
| **1.00** | **0.5818** | **0.5294** | **0.5746** | 0.868 | 0.673 | **0.0000** ← best |

(Stratified RAE columns shown only where they were captured; full data in script stdout if needed.)

**The baseline (do nothing) wins. Every down-weighting hurts.**

## Interpretation

The hypothesis is wrong. Three observations explain why:

1. **The "neither" subset carries real signal.** Removing it entirely (weight=0.0) costs **+0.08 RAE**, a large regression. These 1,248 compounds — the late-arrival diversity library Octant added outside the primary screen — contribute structural diversity and inactive exemplars that the model needs to generalize.
2. **Tree models are not "pulled toward inactives" by mass.** A LightGBM regression on Mordred features does not memorize the marginal pec50 distribution; it conditions on chemistry. The "neither" subset's lower mean pec50 is a property of *those compounds' chemistry*, not a label-distribution bias the model is forced to absorb.
3. **The weight curve is monotone away from 1.0.** Even small down-weights (0.7 → +0.0072, 0.5 → +0.0035) hurt slightly, and aggressive down-weights hurt a lot. There is no productive interior point.

This is the third consecutive failed attempt to extract value from `counter_assay`/`single_concentration`:

| PR | Approach | OOF | LB |
|---|---|---|---|
| #39 | Pseudo-label augmentation from single-conc | Worse (all weights) | n/a |
| #41 | Direct concat of single-conc features into model input | "Improved" (false; selection bias) | **+0.4 disaster** |
| #43→this | Sample-weight down-weighting of "neither" subset | Worse (every weight) | not submitted |

## Lesson

> The train set contains signal we keep underestimating. Attempts to "clean" or "correct" it based on aux-data structure have so far always hurt. The baseline tuned LightGBM on the full 4,140 rows is harder to beat than it looks.

Future aux-data work should not start from "train has a problem" but from "what new information does aux data provide that the encoder cannot already extract from SMILES?" — and the answer must be testable on the actual blinded test compounds, none of which have any aux data.

## What's shipped

- `track1_activity/scripts/run_neither_downweight_sweep.py` — self-contained sweep, no DB writes, no submission
- `docs/neither_downweight_results.md` — this file

## Forward path

Aux-data utilization is parked. Next leverage will come from one of:

- More diverse base models (already explored MoLFormer, ChemProp, AttentiveFP — there may be others e.g. Uni-Mol, MolBERT-Zinc)
- Smarter ensemble strategies (stacking with cross-fold meta-features instead of weighted blend)
- Feature engineering that does not depend on assay metadata (3D conformer descriptors, scaffold-aware encodings)
- Augmentation strategies that don't compress the label space (SMILES enumeration, conformer ensembles)

Counter / single-concentration data can be revisited later if a fundamentally different framing emerges, but no further weight-tuning or feature-concat experiments should be attempted without a new mechanism.

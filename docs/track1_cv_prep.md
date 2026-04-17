# Track 1: CV Prep EDA (pre-bake-off)

**Branch**: `research/cv-prep-eda`
**Date**: 2026-04-17
**Motivation**: Discord leak that a competitor hits MAE ~0.44 with
ChemProp + TabPFN alone (no single-conc, no structure) raised the
question: is our OOF CV the right measurement device? This document
captures two pieces of EDA that were missing from
`track1_eda_report.md` and `track1_eda_redo.md` before deciding CV
bake-off design.

Reproducers: `track1_activity/scripts/eda_cv_prep/0{1,2}_*.py`
Artefacts: `data/eda_cv_prep/*.{parquet,csv}`, figures under
`docs/figures/eda_cv_prep_*.png`.

## TL;DR

1. The 0.095 OOF->LB gap is **overwhelmingly RAE denominator shrinkage,
   not prediction degradation**. MAE gap is only ~0.016 (+3%), but RAE
   gap is 0.096 (+18%). Test's pEC50 dispersion is about 12% tighter
   than train's, so the RAE normaliser collapses on LB.
2. **Test is strongly enriched for analogs of the 46 potent training
   hits** (pEC50>=6, selectivity>=1.5). 48.9% of test compounds have
   their NN in a set that is only 1.1% of train — a **~45x enrichment**.
   Current Morgan+UMAP+KMeans CV doesn't model this structure.
3. Implication for CV design: a split that mimics the "analog-of-potent"
   structure is expected to give a narrower val target distribution
   closer to LB, and therefore to correlate better with LB RAE. This is
   a concrete, testable alternative to "throw more clusters at Morgan
   UMAP and hope".

## EDA 1 - Test-to-train analog distance

`01_test_analog_distance.py`: for each of the 513 test compounds,
compute max-Tanimoto NN similarity (Morgan r=2, 2048 bits) to three
reference subsets.

**Potent-46 definition** (matches CLAUDE.md memory):
```
SELECT compound_id FROM train_activity t JOIN counter_assay c USING (compound_id)
WHERE t.pec50 >= 6.0 AND (t.pec50 - c.pec50) >= 1.5
```
Result: 46 compounds (1.1% of the 4,140 train rows).

### NN similarity distributions

| Reference set | n | mean | std | p25 | median | p75 | p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| All train | 4,140 | 0.532 | 0.073 | 0.484 | 0.523 | 0.571 | 0.672 |
| Non-potent train (4,094) | 4,094 | 0.464 | 0.101 | 0.377 | 0.472 | 0.535 | 0.626 |
| **Potent-46** | 46 | **0.395** | **0.154** | 0.237 | 0.437 | 0.521 | 0.631 |

Key observations:

- **Potent-46 NN distribution is wider and bimodal-ish** (std 0.154 is
  ~2x the others). 1.2% of test is very close (NN>=0.7), 34.3% is close
  (NN>=0.5), but 41.5% is not particularly close to any potent (NN<0.3).
  This matches the Octant pipeline: 46 seeds -> Enamine/FDA analog
  expansion (giving the close-analog tail) + additional diversity
  library (giving the NN<0.3 tail).

- **48.9% of test compounds have their global NN inside the 46-compound
  potent set.** That is a ~45x enrichment vs the 1.1% base rate. If
  test were uniformly sampled from chemical space around train, we'd
  expect ~1.1%.

### Top closest test<->potent pairs

| test cid | NN to potent-46 | top-5 mean to potent-46 |
|---:|---:|---:|
| 4161 | 0.806 | 0.345 |
| 4420 | 0.750 | 0.331 |
| 4595 | 0.745 | 0.313 |
| 4182 | 0.740 | 0.346 |
| 4605 | 0.721 | 0.308 |
| 4424 | 0.712 | 0.301 |

Every top-10 test compound's top-5 mean-to-potent is ~0.3, which means
each test is analog of **exactly one** potent seed, not a cluster of
potents. Consistent with "Enamine analog expansion per hit".

Figures: `docs/figures/eda_cv_prep_01_test_analog_distance.png`,
`docs/figures/eda_cv_prep_01_test_vs_potent46_hist.png`.

### Why current UMAP split misses this structure

`splits.umap_split_indices(smiles, n_clusters=50)` takes Morgan FPs ->
UMAP(10D, Jaccard) -> KMeans(50). Each potent seed lives in whatever
cluster it fell into; the 46 potents are scattered across ~30 clusters.
When one cluster is held out as val, that val contains 0-3 potents and
a mix of random non-potents. It doesn't mimic "val = analogs of held-in
potents", which is what LB looks like.

## EDA 2 - OOF vs LB gap stability

`02_oof_lb_gap.py`: join `lb_submissions` <-> `experiment_summary`.
n=3 pairs (all ensembles; historical ens_v6/v7 submissions predate the
api.py tracking infra and were not backfilled).

| submission | OOF RAE | LB RAE | gap | ratio | OOF MAE | LB MAE | MAE gap | rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ens_l2_a0.05 | 0.5312 | 0.6300 | +0.099 | 1.186 | 0.4834 | 0.5018 | +0.018 | 17 |
| ens_l2_a0.1  | 0.5327 | 0.6263 | +0.094 | 1.176 | 0.4847 | 0.4989 | +0.014 | 17 |
| ens_vanilla  | 0.5318 | 0.6263 | +0.095 | 1.178 | 0.4839 | 0.4989 | +0.015 | 26 |

Historical reference (from `docs/ensemble_cleanup.md`):
- `ens_v7_vanilla`: OOF RAE 0.5304 / LB RAE ~0.62 -> gap ~0.09. Same
  shape as the current 3 points.

**Summary stats (n=3)**:
- RAE gap: mean 0.0956, std 0.0022 (very stable)
- RAE ratio (LB/OOF): mean 1.180, std 0.004
- MAE gap: mean 0.016, std 0.002

### Interpretation: it's mostly the RAE denominator

RAE = MAE / mean(|y - median(y)|). The MAE gap is only +0.016 (+3.3%),
but the RAE gap is +0.096 (+18%). The only way both can be true is if
the LB set's `mean(|y - median(y)|)` is smaller than the OOF set's.

Back-of-envelope:
- OOF: `E|y - median_y| = OOF_MAE / OOF_RAE = 0.4840 / 0.5319 = 0.910`
- LB:  `E|y - median_y| = LB_MAE / LB_RAE   = 0.4999 / 0.6275 = 0.797`

Test's pEC50 dispersion is ~12.4% tighter than train's. This is
consistent with EDA 1: test is enriched for analogs of potent-46, so
its pEC50 distribution is narrower (concentrated around the potent
mass, plus a diversity tail) than the wide train distribution (which
spans inactives at pEC50=1.6 all the way up to 7.5).

**This reframes the "CV is broken" narrative**:
- Our models' absolute accuracy (MAE) is almost the same on LB as on
  OOF. The predictor is not broken.
- RAE is a worse comparison metric on a narrower target, and the
  competition evaluates on RAE.
- Two levers follow: (a) build a CV that uses a val subset whose y
  distribution matches LB's narrower shape; (b) at ensemble /
  hyperparameter selection time, monitor MAE, not just RAE, since MAE
  is more stable between OOF and LB.

### Limitations

- n=3 is too few to fit a real OOF->LB regression. We can only claim
  "gap is stable at ~0.095 across the ensembles we have tracked".
- All 3 points are dense ensembles from the same pool, so they share
  most of their signal. Adding single-model submissions (e.g.
  `chemprop_optuna_umap` alone) would probably widen the gap
  distribution and is the highest-value next data point.

## Implications for CV bake-off

### Concrete hypotheses to test

1. **Analog-aware split correlates better with LB than Morgan+UMAP
   split.** Design: for each of 5 folds, pick ~9 potent seeds from the
   46 and hold out every train compound with Tanimoto NN>=0.4 to any
   of those 9 seeds as val. This mimics "potents in train -> analogs
   in val", the structure LB actually has. Expect: higher OOF RAE
   (val is narrower), smaller OOF->LB gap.

2. **MAE-selected ensembles out-perform RAE-selected ensembles on LB.**
   If the ratio 1.18 is really a denominator effect, selecting weights
   / models by OOF MAE rather than OOF RAE should transfer better.
   Testable at zero LB cost: re-optimise `run_ensemble.py` with MAE
   objective, compare weight distributions.

3. **Multi-seed UMAP variance is a better stability signal than fold
   variance.** Current run reports fold RAE std within one seed. A
   seed sweep (say seed in {0,1,2,3,4}) over UMAP-50 tells us whether
   OOF RAE reflects a stable underlying property of the model or
   sampling noise in the cluster assignment.

### What to de-prioritise

- "Try ChemBERTa/MoLFormer embedding space instead of Morgan FP for
  UMAP". The point of moving away from Morgan isn't embedding choice,
  it's the *shape* of the holdout (analog-of-potent). Swapping Morgan
  for ChemBERTa + UMAP + KMeans would still produce round clusters, not
  the needed seed-plus-neighbours geometry.

### What we still do not know

- True LB target distribution. We inferred ~12% narrower spread from
  MAE/RAE arithmetic; a single LB submission that includes the
  underlying raw predictions (already the case) + comparing
  `mean|pred - median(pred)|` vs train can roughly triangulate. But
  the ground truth test distribution remains blinded.
- Whether the 41.5% of test that is NN<0.3 to potent-46 behaves like
  the analog-rich 58.5% or like something else. An analog-aware split
  only improves CV fidelity for the close-analog regime; the diversity
  tail may need a second fold type.

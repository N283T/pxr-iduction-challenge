# CheMeleon raw-2048 validation

This directory isolates the audit of the original CheMeleon feature pipeline.
It does not modify or replace the legacy `compound_chemeleon` table, whose
300-dimensional values include a randomly initialized projection.

The corrected representation is stored separately in
`compound_chemeleon_raw2048` and is defined as:

```text
CheMeleon pretrained BondMessagePassing (2048d) -> mean aggregation
```

There is no randomly initialized predictor layer in this path.

## Root cause

The legacy extractor did load the pretrained CheMeleon weights into
`BondMessagePassing`, so the stored vectors were not completely detached from
pretraining. The problem occurred immediately afterward:

1. It constructed a new `RegressionFFN(input_dim=mp.output_dim)` without
   loading predictor weights.
2. ChemProp's `MPNN.encoding(bmg)` called
   `predictor.encode(model.fingerprint(bmg), i=-1)`.
3. ChemProp sliced the predictor's top-level MLP with `ffn[:-1]`. The first
   block contains only `Linear(2048, 300)`; ReLU, dropout, and the output layer
   are all in the excluded second block.
4. The stored legacy vector was therefore the pretrained 2,048-dimensional
   fingerprint passed through an untrained random affine projection to 300
   dimensions, before ReLU.

The extractor did not set a Torch seed, so recomputing the legacy table could
also produce a different 300-dimensional projection. These are not independent
random vectors per compound: for fixed projection weights they are a
deterministic compressed view of the pretrained representation. The audit
therefore keeps `compound_chemeleon` untouched and stores the direct pretrained
message-passing/mean-aggregation output in a separate 2,048-dimensional table.

## Commands

```bash
pixi run python \
  track1_activity/analysis/chemeleon_raw2048_validation/extract_raw2048.py

pixi run python \
  track1_activity/analysis/chemeleon_raw2048_validation/run_cv.py \
  --cases raw2048 raw2048-mixed-full raw2048-mixed-top500

pixi run python \
  track1_activity/analysis/chemeleon_raw2048_validation/compare_results.py

pixi run python \
  track1_activity/analysis/chemeleon_raw2048_validation/evaluate_ensemble_swap.py \
  --new-member top500

pixi run python \
  track1_activity/analysis/chemeleon_raw2048_validation/evaluate_ensemble_swap.py \
  --new-member full

pixi run python \
  track1_activity/analysis/chemeleon_raw2048_validation/probe_random_linear_fast.py

pixi run python \
  track1_activity/analysis/chemeleon_raw2048_validation/probe_random_linear_test_fast.py

pixi run python \
  track1_activity/analysis/chemeleon_raw2048_validation/probe_pca300_fast.py
```

The CV runner uses the canonical UMAP split (seed 42, 50 clusters), TabPFN
v2.6, eight estimators, and softmax temperature 0.9. Runtime result JSON files
are written under `data/chemeleon_raw2048_validation/` and predictions are
recorded in the experiment database under `audit_chemeleon_raw2048_*` names.
`compare_results.py` compares the audited runs with the historical 300d runs
on both OOF predictions and every released test label currently present in
`test_activity_phase1_labels`; it records the observed label count rather than
assuming that the table still contains only the original 253-compound AS1 set.

## 2026-08-09 result

| Feature path | Experiment | OOF MAE | Released-test MAE (n=513) |
|---|---:|---:|---:|
| Legacy random-projected 300d | 381 | 0.5118 | 0.5099 |
| Raw pretrained 2048d | 2491 | 0.4983 | 0.5132 |
| Legacy 300d + 2D/Boltz/pred, full | 1608 | 0.4056 | 0.4281 |
| Raw 2048d + 2D/Boltz/pred, full | 2492 | 0.4525 | 0.4692 |
| Legacy 300d + 2D/Boltz/pred, fold-local top-500 | 1609 | 0.3968 | 0.4327 |
| Raw 2048d + 2D/Boltz/pred, fold-local top-500 | 2493 | 0.3992 | 0.4259 |

The raw-2048 top-500 run selected 317--325 CheMeleon dimensions and 175--183
2D/Boltz/pred dimensions per fold. It was 0.0024 worse on OOF MAE but 0.0068
better on all released test labels than the legacy-300d top-500 run.

### Raw-2048 top-K sweep

| K | Experiment | OOF MAE | Released-test MAE (n=513) | Test Spearman | Mean selected CheMeleon dims |
|---:|---:|---:|---:|---:|---:|
| 300 | 2494 | 0.3994 | 0.4277 | 0.8231 | 188.6 |
| 400 | 2495 | **0.3981** | 0.4290 | 0.8214 | 256.2 |
| 500 | 2493 | 0.3992 | **0.4259** | 0.8290 | 322.0 |
| 600 | 2496 | 0.4001 | 0.4278 | 0.8279 | 388.4 |
| 700 | 2497 | 0.4021 | 0.4270 | 0.8299 | 451.4 |
| 800 | 2498 | 0.4078 | 0.4287 | **0.8312** | 512.4 |
| 1000 | 2499 | 0.4086 | 0.4312 | 0.8312 | 634.2 |

K=400 is the OOF optimum, while K=500 is the released-test MAE optimum. The
300--700 region is relatively flat on released-test MAE; degradation becomes
clear at K>=800 on OOF and at K=1000 on released-test MAE.

### Top-500 ensemble SWAP audit

The canonical pool was evaluated without changing `run_ensemble.py`. Only the
legacy-300d top-500 member was replaced by experiment 2493.

| Variant | OOF MAE | Released-test MAE (n=513) | Test Spearman | Top-500 weight |
|---|---:|---:|---:|---:|
| Canonical seed-42 baseline | **0.397089** | **0.420871** | 0.836962 | 0.2704 |
| Fixed baseline weights, prediction SWAP | 0.399458 | 0.421068 | **0.837621** | 0.2704 |
| Reoptimized seed-42 SWAP | 0.399236 | 0.421036 | 0.837340 | 0.2684 |
| Five-seed mean-weight baseline | **0.395756** | **0.421042** | 0.835752 | 0.3375 |
| Five-seed mean-weight SWAP | 0.398136 | 0.421149 | **0.836909** | 0.3395 |

Across Caruana seeds 42--46, reoptimized SWAP deltas were consistently
positive (worse): OOF MAE +0.00147 to +0.00283 and released-test MAE +0.00011
to +0.00055. The raw-2048 member improves the standalone released-test MAE but
does not improve this ensemble because the legacy member has slightly better
complementary errors in the current pool.

### Full-feature ensemble SWAP audit

The same legacy-300d top-500 pool slot was separately replaced by the
raw-2048 mixed-full experiment 2492 (3,851 input dimensions). This is a direct
slot-replacement diagnostic, not a change to the canonical pool.

| Variant | OOF MAE | Released-test MAE (n=513) | Test Spearman | Swapped-slot weight |
|---|---:|---:|---:|---:|
| Canonical seed-42 baseline | **0.397089** | **0.420871** | 0.836962 | 0.2704 |
| Fixed baseline weights, full SWAP | 0.412847 | 0.429402 | 0.835111 | 0.2704 |
| Reoptimized seed-42 full SWAP | 0.407087 | 0.424162 | **0.837318** | 0.0515 |
| Five-seed mean-weight baseline | **0.395756** | **0.421042** | 0.835752 | 0.3375 |
| Five-seed mean-weight full SWAP | 0.407404 | 0.425119 | **0.836359** | 0.0814 |

Across Caruana seeds 42--46, every reoptimized full SWAP was worse: OOF MAE
changed by +0.00737 to +0.01435 (mean +0.01099), and released-test MAE changed
by +0.00276 to +0.00476 (mean +0.00368). Reoptimization sharply reduced the
replacement member's weight, from the baseline slot weight of 0.2704 to 0.0515
for seed 42, but did not recover the baseline error.

### Fast 300d compression probe

A lightweight follow-up tested whether the saved legacy projection was a
lucky random draw. It used the same mixed feature set and canonical UMAP
five-fold split, but reduced TabPFN to five estimators. OOF and all 513 released
test labels were evaluated separately to keep each run short.

| 300d representation | OOF MAE | Released-test MAE | Test Spearman |
|---|---:|---:|---:|
| Saved legacy random linear projection | 0.437576 | 0.453159 | **0.818788** |
| New random linear projection, seed 0 | 0.438944 | **0.452014** | 0.818716 |
| New random linear projection, seed 1 | 0.437392 | 0.456174 | 0.814023 |
| New random linear projection, seed 2 | **0.436700** | 0.455128 | 0.814432 |
| New random linear projection, mean | 0.437679 | 0.454439 | 0.815724 |
| Global train-only PCA, 300 components | 0.437865 | 0.455202 | 0.816664 |

The three new random projections averaged within 0.00010 OOF MAE of the saved
legacy projection. On released-test replay, the saved projection was 0.00128
better than the three-seed mean, but seed 0 was 0.00115 better than legacy.
The saved projection is therefore not an obvious exceptional draw, although
projection seed still moved test MAE by about 0.0042 across this small sample.

PCA retained 92.552% of raw-2048 variance but did not improve OOF or released-
test MAE. This PCA was fitted once on all training features without labels, not
inside each CV fold, so it is a lightweight diagnostic rather than a strict
fold-local estimator. The result supports random linear compression as a
useful low-dimensional interface for TabPFN rather than evidence that the
legacy projection learned a task-specific activity axis.

## Conclusion

The legacy feature path was technically flawed and non-reproducible, but its
random projection did not explain away the historical model performance. The
direct pretrained 2,048-dimensional fingerprint improved the standalone OOF
result and, after fold-local selection, improved the 513-label test replay
relative to the legacy top-500 member. However, the raw-2,048 top-500 member
was slightly worse in the fixed canonical ensemble context, and the unselected
full feature set was substantially worse both standalone and as a slot swap.
The useful signal is concentrated enough that feature selection is essential;
simply increasing the dimensionality does not improve the ensemble. The fast
compression probe further suggests that most of the legacy 300d benefit came
from compression itself, not from a uniquely lucky random seed. In this mixed
model, the compressed CheMeleon representation is best understood as a weak
complement to the dominant predicted-log2fc/activity features.

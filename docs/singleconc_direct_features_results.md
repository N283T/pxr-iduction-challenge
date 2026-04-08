# Direct single_concentration features (negative result)

**Branch**: `feature/singleconc-direct-features`
**Related**: PR #39 (pseudo-labeling, also negative), `docs/auxiliary_data_analysis.md`, `docs/pseudo_labeling_results.md`

## Summary

Adding `single_concentration` features directly to the LightGBM input matrix produces a striking OOF improvement (RAE 0.5817 → 0.4672, −0.115) but **regresses LB catastrophically** (0.622 → 1.027). The root cause is a train-set selection bias that the model learns and then misapplies to test compounds.

This experiment, combined with the LB submission, **definitively closes the "auxiliary data as direct feature" approach**. Auxiliary data must be used in ways that do not allow the model to exploit the missingness pattern.

## What was tried

### Experiment A: direct feature concat

Added `mordred_singleconc` feature mode to `run_train.py` that loads Mordred descriptors and concatenates 6 per-compound single_concentration features:
- `log2fc_8_25e_6`, `log2fc_3_30e_5`
- `log2fc_stderr_8_25e_6`, `cohens_d_8_25e_6`, `p_value_8_25e_6`
- `n_concs`

Compounds without single_concentration data (1,748 train, all 513 test) get NaN values, which LightGBM handles natively via missing-value branches.

### Experiment B: confidence-restricted pseudo-labeling

Added `--pseudo-min-confidence FLOAT` CLI flag to filter pseudo labels by their per-row confidence before augmentation. Tested at conf≥0.9 (top 25% = 2,111 compounds) with weights {0.05, 0.1, 0.3}.

## Results

### Experiment A: OOF vs LB

| | OOF RAE | OOF MAE | LB RAE |
|---|---:|---:|---:|
| baseline (mordred only) | 0.5817 | 0.529 | 0.622 |
| **+ single_conc concat** | **0.4672** | **0.425** | **1.027** |

**The OOF improvement does not survive submission.** LB MAE 0.815 is consistent with predictions being systematically shifted ~0.43 lower than true pEC50 (baseline LB MAE 0.50 + 0.43 ≈ 0.93).

### Stratified OOF analysis (the smoking gun)

| Subset | n | baseline RAE | + single_conc RAE | Δ |
|---|---:|---:|---:|---:|
| has single_conc | 2,392 | 0.878 | 0.539 | **−0.339** |
| no single_conc | 1,748 | 0.626 | 0.629 | +0.003 |
| **total** | 4,140 | 0.582 | 0.467 | −0.115 |

The OOF gain is **entirely concentrated** in the subset of train compounds that have single_conc data. The remaining 1,748 compounds (and all 513 test compounds, which have no single_conc) see no benefit at all.

### The pEC50 distribution shift in the no-single_conc subset

| Subset | n | pEC50 mean | pEC50 std |
|---|---:|---:|---:|
| has single_conc | 2,392 | **4.82** | 0.80 |
| no single_conc | 1,748 | **3.63** | 1.14 |

The two groups differ by **1.2 pEC50 units** in their mean. This is a train-set construction artifact: compounds from different cohorts or screens ended up with different aux-data coverage, and the missing-data indicator became a strong signal for low-activity compounds.

### Why test predictions collapsed

LightGBM trees split on the new single_conc features. The "missing" branch absorbs all 1,748 train compounds with no single_conc PLUS all 513 test compounds. The leaf at the end of that branch is fit on a population whose mean pEC50 is **3.63** (vs the global train mean 4.74).

At inference, every test compound traverses missing branches and lands in low-pEC50 leaves. Comparing test-prediction distributions:

| | mean | std |
|---|---:|---:|
| baseline test preds | 4.696 | 0.637 |
| + single_conc test preds | **4.264** | 0.648 |
| (shift) | **−0.43** | — |

The −0.43 shift fits the LB MAE gap exactly. The OOF score "improved" because OOF is computed on train compounds (which contain the high-pEC50 has-single_conc subset), while LB is computed on test (which behaves like the no-single_conc subset).

This is a **selection-bias leakage**: a feature ("missing single_conc") that correlates with the target through training-set construction but not through any property of the underlying chemistry.

### Experiment B: confidence-restricted pseudo-labeling

`build_pseudo_labels.py` produces confidence values via `clip(1/(1+stderr), 0.1, 1.0)`. Empirically, the confidence distribution is **highly degenerate**: mean 0.894, std 0.012, range [0.872, 0.911]. The clip at 1.0 collapses most of the variation. Only 25% of pseudo labels have confidence ≥ 0.9; nothing exceeds 0.95.

| Setting | OOF RAE | Δ vs baseline |
|---|---:|---:|
| baseline | 0.5817 | — |
| conf≥0.9, w=0.05 | 0.5845 | +0.0028 |
| conf≥0.9, w=0.1 | 0.5859 | +0.0042 |
| conf≥0.9, w=0.3 | 0.5928 | +0.0111 |
| (full, w=0.05) | 0.5923 | +0.0106 |
| (full, w=0.30) | 0.6169 | +0.0352 |

Filtering to high-confidence reduces the damage by roughly half but **does not produce improvement**. The dominant cause of pseudo-labeling failure (label-space compression: pseudo mean 3.87 vs train 4.74) is not addressed by confidence filtering — high-confidence rows are not systematically less compressed than low-confidence ones.

## What's shipped

Both Experiment A's `mordred_singleconc` feature and Experiment B's `--pseudo-min-confidence` flag are kept in `run_train.py` behind opt-in flags. The baseline path is unchanged. These are useful primarily for future experiments that want to probe related ideas (e.g., a fixed-offset bias correction at inference, or a stratified two-model approach).

## Lessons

1. **OOF gains that concentrate in a subset present at training but absent at test are usually fictitious.** Always stratify OOF by the presence of any feature whose distribution differs between train and test.
2. **Missing-value indicators can become target proxies.** Whenever a feature has very different missingness rates between train and test, the model can learn to predict the train-conditional target distribution from the missingness pattern alone.
3. **In-database confidence values are only useful when they cover a wide dynamic range.** Our confidence (1/(1+stderr)) saturated at the upper end and gave no real signal.
4. **One LB submission is worth a thousand CV experiments** when CV-LB mismatch is suspected. The submission cost ranking 14 → 41 but proved the bias hypothesis decisively.

## Forward path (NOT in this PR)

The two negative results (pseudo-labeling and direct feature concat) leave **multi-task learning** (issue #36) as the cleanest remaining auxiliary-data lever:

- counter_assay (or single_concentration log2fc) becomes an auxiliary regression target
- A ChemProp / AttentiveFP shared encoder is forced to learn representations that explain both PXR pEC50 and the auxiliary signal
- At inference, only the PXR head is used
- Test compounds are not penalized by missing aux features (no missing-feature branches)

This is the next experiment to run.

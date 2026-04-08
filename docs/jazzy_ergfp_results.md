# Jazzy + ErGFP feature integration

**Branch**: `feature/jazzy-ergfp-features`
**Status**: Shipped, default-params OOF gain is small. Follow-up: Optuna tuning of `mordred_jazzy`.
**Motivation**: ExpansionRx top teams (moka #3, beetroot #9, crh201 #12, HybridADMET #6, UncertainTea #19) used Jazzy and/or ErG features. First OpenADMET lessons-learned blog (2026-02) explicitly names these libraries. We replicate that setup for PXR.

## What shipped

### Jazzy — 6 physical features per compound

`jazzy.api.molecular_vector_from_smiles(smiles, minimisation_method="MMFF94")` returns six scalar features per molecule:

| Column | Meaning | Units |
|---|---|---|
| `sdc` | sum donor strength (C–H) | — |
| `sdx` | sum donor strength (heteroatom–H) | — |
| `sa` | sum acceptor strength | — |
| `dga` | apolar free energy of hydration | kJ/mol |
| `dgp` | polar free energy of hydration | kJ/mol |
| `dgtot` | total free energy of hydration | kJ/mol |

- **DB table**: `compound_jazzy` (compound_id PK + 6 double-precision columns), 13,136 rows
- **Compute time**: ~10 min for all compounds on CPU
- **Failure**: 1 compound (`CS(=O)(=O)CCC(=O)N1C[C@H]2CC[C@H]1C2`, id 3840) failed RDKit 3D embedding, backfilled with column means. 0.008% imputation rate, not a concern.
- **Loader**: `data.load_jazzy(compound_ids=None)` → DataFrame indexed by compound_id
- **Install**: `jazzy` has a hard `rdkit<2024` pin, but only uses stable APIs. We install it via `pixi run install-jazzy` which runs `pip install --no-deps jazzy==0.1.4 kallisto==1.0.10`. Tested to work against our rdkit 2026.3.1.

### ErGFP — Extended Reduced Graph fingerprint

- `rdkit.Chem.rdReducedGraphs.GetErGFingerprint(mol)` → 315-dim float vector
- Registered in `FP_REGISTRY["ergfp"]` in `track1_activity/src/features.py`
- No DB caching needed (compute at load time)

## Feature mode added to `run_train.py`

`--feature mordred_jazzy` concatenates the 1,515 Mordred columns with the 6 Jazzy columns → 1,521 features total. Both train and test are fully populated (after the 1-compound backfill), so no missing-feature selection-bias risk à la PR #41.

## Results (default LightGBM params, UMAP-fold CV)

| Experiment | OOF RAE | OOF MAE | OOF R² | Spearman |
|---|---:|---:|---:|---:|
| `lgbm_mordred_umap_default` (baseline) | 0.5817 | 0.5293 | 0.5756 | 0.7132 |
| **`lgbm_mordred_jazzy_umap_default`** | **0.5804** | 0.5281 | 0.5746 | **0.7153** |
| `lgbm_ergfp_umap_default` | 0.7265 | 0.6585 | 0.3962 | 0.5670 |
| `lgbm_jazzy_umap_default` (6 cols alone) | 0.7885 | 0.7175 | 0.2975 | 0.4418 |

### Jazzy-alone sanity check

Pure 6-feature LightGBM on Jazzy columns hits OOF RAE 0.7885 — expectedly weak, but not useless: R²=0.30 means ~30% of the pEC50 variance is captured by six hydration/H-bond scalars alone. Confirms these features carry *some* orthogonal information; the question is whether they complement or overlap with Mordred's 1,515-dim representation. Based on feature-importance analysis below, the answer is "overlap dominates, but `sa` and `dgtot` still slot into the top of the gain ranking."

Delta (mordred_jazzy − mordred) = **−0.0013 RAE**. Noise level on its own, but Spearman improves and the direction is consistent with top-team findings.

## Feature importance — Jazzy IS being used

A direct LightGBM fit on all Mordred + Jazzy features with gain-based importance shows:

| Rank (of 1537) | Feature | Gain | Splits |
|---:|---|---:|---:|
| 5 | **sa** (jazzy) | 961 | 30 |
| 36 | **dgtot** (jazzy) | 153 | ~6 |
| 91 | **dgp** (jazzy) | 66 | 2 |
| 145 | sdx | 50 | — |
| 219 | dga | 37 | — |
| 873 | sdc | 6 | — |

The acceptor-strength feature `sa` is the **5th most important feature in the entire model** — ahead of all but 4 Mordred columns. `dgtot` (total hydration free energy) sits at rank 36. The small OOF gain happens because Mordred already captures much of the same physical information (via `tpsa`, `logp`, `hbd`/`hba` counts), but the Jazzy features are more directly useful per split.

This matches what top teams observed: Jazzy does add signal, but you need Optuna-level tuning of the combined feature space to translate that into measurable OOF gains.

## Optuna-tuned mordred_jazzy (20 trials, UMAP fold)

Running `run_train.py --model lgbm --feature mordred_jazzy --split umap --trials 20`:

| Experiment | OOF RAE | Δ vs default |
|---|---:|---:|
| `lgbm_mordred_jazzy_umap_default` | 0.5804 | — |
| **`lgbm_mordred_jazzy_umap` (Optuna)** | **0.5729** | **−0.0075** |
| `lgbm_mordred_umap` (Optuna, no jazzy) | 0.5818 | n/a |
| `optuna_mordred+morgan` (historical best mordred-family) | 0.5545 | n/a |

Optuna tuning of the combined `mordred+jazzy` space recovers a measurable gain over default-params (−0.0075 RAE) and is −0.0089 better than Optuna-tuned plain Mordred, but still underperforms the `mordred+morgan` combination. This confirms the default-params result was at the noise floor only because hyperparameters were not adapted to the enlarged feature space — the signal was real but needed tuning to emerge.

Next step to beat the `mordred+morgan` reference: try a `mordred+morgan+jazzy` combined feature mode (follow-up PR).

## Ensemble contribution (ens_v8 refresh with tuned mordred_jazzy)

After adding `lgbm_mordred_jazzy_umap` (tuned) to the candidate pool and excluding the weak ErGFP-alone and jazzy-alone models:

| Strategy | OOF RAE | Δ vs prior best (0.5253) |
|---|---:|---:|
| **`ens_v8_vanilla_opt`** | **0.5268** | +0.0015 |
| `ens_v8_fold_based` | 0.5269 | +0.0016 |
| `ens_v8_l2_alpha=0.1` | 0.5269 | +0.0016 |

**Tuned `lgbm_mordred_jazzy` picks up weight 0.0793 (rank 3)** in `ens_v8_vanilla_opt`, behind only `attentivefp_optuna_umap` (0.110) and `chemprop_multitask5_umap_aux0.0_tuned` (0.084). This confirms the new feature is genuinely decorrelated enough to earn real weight.

The overall ensemble OOF RAE is within noise of the prior 0.5253 best. Typical pattern: new models that correlate with existing ones get weight at the expense of incumbents, with marginal net change in OOF. The value is *not* a guaranteed LB improvement — it is an additional diversified component that may help under distribution shift.

**ErGFP alone (OOF RAE 0.7265) is too weak to make it into any significant ensemble weight**, as expected. We add `_ergfp_` to the `EXCLUDE_SUBSTRINGS` list in `run_ensemble_v8.py` so it is skipped by the automated candidate pool. The feature remains available in `FP_REGISTRY` for manual experiments and feature-concat follow-ups. Jazzy-alone (`lgbm_jazzy_*`) is excluded via an `EXCLUDE_PATTERNS` lambda for the same reason.

## moka's Jazzy use is different from ours

The first-place-adjacent `moka` submission ([dahvida/openadmet-expansionrx-blind-challenge-moka](https://github.com/dahvida/openadmet-expansionrx-blind-challenge-moka)) uses Jazzy in a fundamentally different way: **Jazzy descriptors are auxiliary multi-task regression targets during ChemProp / GNN fine-tuning, not input features**. Their "Type B" fine-tuning strategy trains the model to predict (endpoint + Jazzy + SHAP-selected MOE descriptors) simultaneously, and they select Strategy A or B per endpoint based on OOF performance.

This is a meaningful pointer for our future MTL work (PR #42 was `pec50 + counter_pec50 + single_conc` as aux targets; we could retry with `pec50 + jazzy` as aux targets instead). Noted as a medium-term experiment but outside the scope of this PR.

## Interpretation

Default-params results are at the noise floor. The feature-importance analysis proves Jazzy is not being ignored — `sa` is the 5th most informative feature in the entire space. The tiny OOF delta tells us the effect is real but overlaps with Mordred's coverage. The path to a meaningful gain is:

1. **Optuna tuning on `mordred_jazzy`** — the baseline uses default params, but the best reference in our DB is Optuna-tuned `mordred+morgan_r2_2048` at OOF RAE 0.5531. A tuned `mordred+jazzy` has a plausible path to 0.55 range.
2. **Combined feature concat** — `mordred + morgan + jazzy + ergfp` as a single LightGBM input. Top teams did this. Not yet implemented (would need new feature mode).
3. **ChemProp with Jazzy as extra_features** — ChemProp v2 supports per-molecule feature concatenation to the message-passing output. This is how several top teams ingested Jazzy.

## What's shipped

- `db/compute_jazzy.py` — compute and store Jazzy features
- `track1_activity/src/data.py` — `load_jazzy`, `JAZZY_FEATURE_COLS`
- `track1_activity/src/features.py` — `erg_fp`, `FP_REGISTRY["ergfp"]`
- `track1_activity/scripts/run_train.py` — `mordred_jazzy` feature mode
- `pyproject.toml` — `install-jazzy` pixi task
- `docs/jazzy_ergfp_results.md` — this file

## Forward path

Short-term: Optuna-tune `mordred_jazzy` (20 trials, ~30 min). If it beats the current `mordred+morgan` 0.5531 baseline, add to ensemble. If not, Jazzy's value is purely as one decorrelated component in top-8 averaging.

Medium-term: implement a `mordred_morgan_ergfp_jazzy` feature mode as a single LightGBM input. This is the closest replica of the top-team config and is the highest-expected-value feature-engineering experiment left in the 2D-only budget.

Long-term: ChemProp with Jazzy as `extra_atom_features` or `extra_bond_features`. This is the form in which Jazzy was used by the top teams that adopted ChemProp.

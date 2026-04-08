# Ensemble cleanup — canonical `run_ensemble.py`

**Branch**: `feature/ensemble-cleanup`
**Status**: Done, awaiting LB verification
**Reason**: Structural bugs in the legacy `run_ensemble_v*.py` scripts were silently contaminating the candidate pool and biasing fold-weight optimization. Fix collapses the three versioned scripts (`run_ensemble.py`, `run_ensemble_v2.py`, `run_ensemble_v8.py`) to a single canonical `run_ensemble.py` with an explicit allow list.

## What was broken

### 1. Silent scaffold contamination

`run_ensemble_v2.py` (introduced in PR #26) had no split filter — it queried `experiments` for anything with OOF predictions and a `submission_path`. When the project moved to UMAP split for new models (PR #31 onward), the old scaffold-split `single_*`, `xgboost_mordred`, and `catboost_mordred` entries were never deleted from the DB and never re-run with UMAP. Once they had submission CSVs generated, they started appearing in the ensemble pool alongside genuinely UMAP-trained models.

By the time PR #42 forked `run_ensemble_v8.py`, the pool had grown to ~54 models with **mixed scaffold and UMAP splits in the same weighted blend**. The optimizer couldn't distinguish "this model is truly better" from "this model was evaluated on an easier split" — both look like lower OOF RAE.

Legacy scaffold experiments accumulated ~20% of `ens_v8_vanilla_opt`'s total weight:

| Scaffold / legacy model in ens_v8_vanilla_opt | Weight |
|---|---:|
| `xgboost_mordred` | 0.053 |
| `chemprop_scaffold` | 0.051 |
| `lgbm_count_morgan_r2_2048_scaffold` | 0.047 |
| `single_mordred` | 0.024 |
| `single_avalon_2048` | 0.023 |
| **Total** | **0.198** |

### 2. Scaffold folds for weight optimization despite UMAP OOFs

Both `run_ensemble_v2.py` and `run_ensemble_v8.py` called `scaffold_split_indices(...)` at main() for the `fold_based` and `fold_l2` weight optimization strategies. When the majority of models in the pool had OOF predictions generated from UMAP folds, optimizing on scaffold folds meant **evaluating UMAP-trained models on a fold structure they were never held out on**. This isn't technically incorrect (the OOF predictions are still predictions) but it biases weight selection toward models whose scaffold-fold prediction distribution matches the scaffold-fold validation points, which has no a priori reason to match LB behavior.

### 3. Confusing `v*` naming

The script filenames jumped `run_ensemble.py` → `run_ensemble_v2.py` → **`run_ensemble_v8.py`**. There was never a v3, v4, v5, v6, or v7 script. The "v8" was chosen in PR #42 to match the submission name prefix (`ens_v8_*`) rather than incrementing the previous script version. This made it hard to figure out which script was authoritative, or what ens_v3..v7 even were (answer: submission tags only, using the same v2 script).

## What `run_ensemble.py` does now

### One file, explicit allow list

The legacy scripts are moved to `track1_activity/scripts/archive/`:

- `run_ensemble_original.py` (was `run_ensemble.py`, commit 42347fe from PR #15)
- `run_ensemble_v2.py` (was in PR #26)
- `run_ensemble_v8.py` (was in PR #42)

The new canonical `run_ensemble.py` hard-codes the candidate pool as `ENSEMBLE_MODELS: tuple[str, ...]`. Every model is one line with an OOF RAE comment. Adding or removing a candidate is a single-line diff that shows up in code review.

**Why allow list over filter**: the filter-based approach (`WHERE name LIKE '%_umap%' AND rae < 0.68`) was the source of the contamination. Automatic pickup means any future mistake in naming or metadata silently changes the ensemble. The allow list makes the pool explicit and auditable.

### UMAP folds for fold-based optimization

`run_ensemble.py` imports `umap_split_indices` from `splits.py` and uses those folds for `fold_based` and `fold_l2` strategies, matching the candidates' training splits.

### Hard-fail on missing models

If any name in `ENSEMBLE_MODELS` doesn't resolve to (OOF rows, submission CSV), the script raises `RuntimeError` with the list of missing models. The legacy scripts silently skipped missing models, which meant the pool could shrink without warning.

## Final pool (20 models)

After user review, the mordred family was collapsed because of high internal correlation (>0.95 Pearson):

| pair | Pearson r | decision |
|---|---:|---|
| `lgbm_mordred_umap` ↔ `lgbm_mordred_jazzy_umap` | **0.983** | mordred_jazzy is a strict superset of mordred; drop plain mordred |
| `lgbm_mordred_umap` ↔ `lgbm_mordred_umap_gap0.5` | 0.963 | gap variants have ~0.96 correlation with every mordred version; marginal diversity |
| `lgbm_mordred_umap` ↔ `lgbm_mordred_umap_gap1.0` | 0.958 | same; drop both gap variants |
| reference: `lgbm_mordred_umap` ↔ `chemprop_optuna_umap` | 0.869 | cross-architecture diversity is meaningful |

`residual_physprop+mordred_umap` is kept despite its name — it uses a two-stage residual architecture (physchem Stage 1 → Mordred Stage 2), so its inductive bias is genuinely different from plain LightGBM on Mordred.

| # | Model | OOF RAE | Notes |
|---:|---|---:|---|
| 1 | `lgbm_mordred_jazzy_umap` | 0.5784 | New this PR cycle |
| 2 | `chemprop_multitask5_umap_aux0.0_tuned` | 0.5817 | Best 5-task MTL variant |
| 3 | `chemprop_optuna_umap` | 0.5785 | |
| 4 | `attentivefp_optuna_umap` | 0.5871 | |
| 5 | `residual_physprop+mordred_umap` | 0.5861 | 2-stage residual arch |
| 6 | `lgbm_chemeleon_umap` | 0.6137 | |
| 7 | `lgbm_chemberta_5m_mtr_umap` | 0.6218 | |
| 8 | `lgbm_chemeleon_umap_gap1.0` | 0.6511 | |
| 9 | `lgbm_chemberta_5m_mtr_umap_gap1.0` | 0.6521 | |
| 10 | `lgbm_molformer_xl_umap` | 0.6522 | |
| 11 | `lgbm_count_morgan_r2_2048_umap` | 0.6225 | |
| 12 | `lgbm_count_atompair_2048_umap` | 0.6280 | |
| 13 | `lgbm_count_morgan_r3_2048_umap` | 0.6310 | |
| 14 | `lgbm_count_morgan_r2_2048_umap_gap1.0` | 0.6413 | |
| 15 | `lgbm_avalon_2048_umap` | 0.6536 | |
| 16 | `lgbm_morgan_r2_2048_umap` | 0.6579 | |
| 17 | `lgbm_atompair_2048_umap` | 0.6623 | |
| 18 | `lgbm_feat_morgan_r2_2048_umap` | 0.6774 | borderline |
| 19 | `lgbm_rdkit_desc_umap` | 0.6338 | |
| 20 | `lgbm_rdkit_desc_umap_gap1.0` | 0.6442 | |

Threshold: every model has OOF RAE < 0.68 (project policy). Dropped from the ens_v7 pool: `chemprop_scaffold` (weight 0.140, scaffold), `chemeleon_finetune` (weight 0.050, scaffold + weak).

## Results

| Strategy | OOF RAE | OOF MAE | OOF R² | Spearman |
|---|---:|---:|---:|---:|
| `ens_l2_a0.05` | **0.5312** | 0.4834 | 0.6309 | 0.7599 |
| `ens_vanilla` | 0.5318 | 0.4839 | 0.6299 | 0.7587 |
| `ens_l2_a0.1` | 0.5327 | 0.4847 | 0.6306 | 0.7604 |
| `ens_fold` | 0.5338 | 0.4857 | 0.6306 | 0.7597 |
| `ens_fold_l2_a0.1` | 0.5343 | 0.4862 | 0.6301 | 0.7604 |
| `ens_l2_a0.3` | 0.5397 | 0.4911 | 0.6259 | 0.7600 |
| `ens_fold_l2_a0.3` | 0.5402 | 0.4915 | 0.6257 | 0.7600 |
| `ens_l2_a0.5` | 0.5452 | 0.4960 | 0.6214 | 0.7584 |
| `ens_simple_avg` | 0.5607 | 0.5102 | 0.6063 | 0.7517 |

### Weight distribution for `ens_vanilla` (20-model pool)

| Model | Weight |
|---|---:|
| `chemprop_multitask5_umap_aux0.0_tuned` | 0.197 |
| `chemprop_optuna_umap` | 0.193 |
| `attentivefp_optuna_umap` | 0.162 |
| `residual_physprop+mordred_umap` | 0.151 |
| `lgbm_mordred_jazzy_umap` | 0.082 |
| `lgbm_count_morgan_r2_2048_umap` | 0.049 |
| `lgbm_rdkit_desc_umap_gap1.0` | 0.049 |
| `lgbm_chemeleon_umap` | 0.039 |
| `lgbm_chemeleon_umap_gap1.0` | 0.020 |
| `lgbm_chemberta_5m_mtr_umap` | 0.020 |
| Other 10 models | < 0.02 each |

The top 5 models (4 DL architectures + tuned Mordred+Jazzy) account for **78.4%** of total weight. This is dramatically cleaner than `ens_v8_vanilla_opt`'s weight distribution, which had 16 models each taking 0.02-0.11 with significant scaffold contamination.

## Comparison to legacy ensembles

| Ensemble | OOF RAE | Pool | Split consistency |
|---|---:|---:|---|
| `ens_v7_vanilla` (prior best, LB 0.62) | 0.5304 | 23 (mostly UMAP + 1 scaffold) | 96% clean |
| `ens_v8_vanilla_opt` (broken) | 0.5268 | ~54 mixed | ~80% clean |
| **`ens_l2_a0.05` (canonical)** | **0.5312** | **20 UMAP-only** | **100% clean** |

The canonical OOF RAE is slightly *higher* than the broken v8 — this is the expected cost of removing over-fitting degrees of freedom. The clean pool's LB behavior should be closer to the OOF because the optimizer had fewer misleading signals. Verification requires the next submission window.

## DB cleanup

Legacy experiments have been archived with a `model_type` prefix:

```sql
UPDATE experiments
SET model_type = 'archived_' || model_type,
    notes = COALESCE(notes, '') || ' [ARCHIVED 2026-04-09: legacy scaffold / dropped policy]'
WHERE name IN (
    'single_rdkit_desc', 'single_morgan_r2_2048', 'single_morgan_r3_2048',
    'single_maccs', 'single_avalon_2048', 'single_mordred', 'single_chemeleon',
    'single_chemberta_77m_mlm', 'single_chemberta_5m_mtr', 'single_bert_base_smiles',
    'xgboost_mordred', 'catboost_mordred',
    'chemprop_scaffold', 'chemeleon_finetune',
    'lgbm_feat_morgan_r2_2048_scaffold', 'lgbm_atompair_2048_scaffold',
    'lgbm_count_atompair_2048_scaffold', 'lgbm_count_morgan_r2_2048_scaffold',
    'lgbm_count_morgan_r3_2048_scaffold', 'lgbm_count_rdkit_fp_2048_scaffold',
    'lgbm_rdkit_fp_2048_scaffold', 'lgbm_topo_torsion_2048_scaffold',
    'lgbm_maccs_scaffold_default'
);
```

Legacy ensemble rows (`ens_v7_*`, `ens_v8_*`) are also archived. Total archived: 39 rows. No rows were physically deleted — historical OOF predictions and submission paths remain in DB for forensic lookup but are filtered out of any query that excludes `model_type LIKE 'archived_%'`.

## Forward path

1. Submit `ens_l2_a0.05.csv` at the next available window — direct test of the hypothesis that the cleaner pool improves LB RAE
2. If LB improves: add more genuinely diverse models to the allow list (KERMT, CheMeleon-initialized ChemProp, moka-style Jazzy-as-MTL-target)
3. If LB regresses: the gap wasn't split contamination. Investigate other sources (OOF overfitting of weights themselves, feature engineering mismatch)

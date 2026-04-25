#!/usr/bin/env -S pixi run python
"""Unified training script for single model+feature experiments.

Usage:
    pixi run python run_train.py --model lgbm --feature mordred
    pixi run python run_train.py --model xgboost --feature morgan_r2_2048 --split umap
    pixi run python run_train.py --model catboost --feature count_morgan_r2_2048 --trials 0

Key changes from legacy scripts:
- No inner CV: Optuna tunes directly on outer fold's val set
- No final re-tuning: reuses median best params from CV folds
- Unified model/feature/split selection via CLI args
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.joinpath("src")))
REPO_ROOT = Path(__file__).resolve().parents[2]

import numpy as np
import pandas as pd
import psycopg2

from data import (
    DB_PARAMS,
    JAZZY_FEATURE_COLS,
    get_conn,
    load_jazzy,
    load_mordred,
    load_rdkit_full,
    load_test_smiles,
    load_train_mordred,
    load_train_smiles_target,
    load_train_smiles_with_counter,
)
from evaluate import (
    compute_metrics,
    print_fold_summary,
    print_metrics,
    record_experiment,
    save_oof_predictions,
)
from features import FP_REGISTRY, smiles_to_mols
from pseudo_labels import (
    augment_fold,
    build_pseudo_feature_matrix,
    load_pseudo_labels,
)
from splits import (
    adversarial_split_indices,
    analog_aware_split_indices,
    mixed_analog_diversity_split_indices,
    scaffold_split_indices,
    test_nn_split_indices,
    umap_split_indices,
)

SUBMISSION_DIR = Path(__file__).resolve().parent.parent.joinpath("submissions")
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

# Embedding tables in DB
EMBEDDING_TABLES = {
    "chemeleon": "compound_chemeleon",
    "chemberta_77m_mlm": "compound_chemberta",
    "chemberta_77m_mtr": "compound_chemberta_mtr",
    "chemberta_100m_mlm": "compound_chemberta_100m",
    "chemberta_10m_mlm": "compound_chemberta_10m",
    "chemberta_10m_mtr": "compound_chemberta_10m_mtr",
    "chemberta_5m_mlm": "compound_chemberta_5m",
    "chemberta_5m_mtr": "compound_chemberta_5m_mtr",
    "chemberta_zinc_v1": "compound_chemberta_zinc_v1",
    "bert_base_smiles": "compound_bert_smiles",
    "molformer_xl": "compound_molformer",
    # ChemFM-1B (TheLuoFengLab, Nature Comm Chem 2025): Llama-style causal LM
    # pretrained on UniChem SMILES. Two pooling variants exposed via SQL views.
    "chemfm_1b_last": "compound_chemfm_1b_last",
    "chemfm_1b_mean": "compound_chemfm_1b_mean",
    "chemfm_3b_last": "compound_chemfm_3b_last",
    "chemfm_3b_mean": "compound_chemfm_3b_mean",
}

# Default hyperparams per model type (used when --trials 0)
DEFAULT_PARAMS = {
    "lgbm": {
        "objective": "regression",
        "metric": "mae",
        "boosting_type": "gbdt",
        "verbose": -1,
        "seed": 42,
        "num_leaves": 63,
        "learning_rate": 0.02,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "lambda_l1": 0.01,
        "lambda_l2": 1.0,
    },
    "xgboost": {
        "objective": "reg:absoluteerror",
        "eval_metric": "mae",
        "tree_method": "hist",
        "seed": 42,
        "verbosity": 0,
        "max_depth": 6,
        "learning_rate": 0.02,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "min_child_weight": 10,
        "reg_alpha": 0.01,
        "reg_lambda": 1.0,
        "gamma": 0.01,
    },
    "catboost": {
        "iterations": 2000,
        "loss_function": "MAE",
        "eval_metric": "MAE",
        "verbose": 0,
        "random_seed": 42,
        "allow_writing_files": False,
        "depth": 6,
        "learning_rate": 0.03,
        "l2_leaf_reg": 3.0,
        "bagging_temperature": 0.5,
        "random_strength": 1.0,
        "border_count": 128,
    },
    "tabpfn": {
        "n_estimators": 8,
        "device": "cuda",
        "softmax_temperature": 0.9,
        "random_state": 42,
        "ignore_pretraining_limits": True,
    },
}


# ---------------------------------------------------------------------------
# Feature loading
# ---------------------------------------------------------------------------


def load_compound_ids(split: str) -> list[int]:
    conn = get_conn()
    cur = conn.cursor()
    table = "train_activity" if split == "train" else "test_activity"
    cur.execute(f"SELECT compound_id FROM {table} ORDER BY id")
    ids = [r[0] for r in cur.fetchall()]
    cur.close()
    conn.close()
    return ids


def load_embeddings(table: str, compound_ids: list[int]) -> np.ndarray:
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    ph = ",".join(["%s"] * len(compound_ids))
    cur.execute(
        f"SELECT compound_id, embedding FROM {table} WHERE compound_id IN ({ph})",
        compound_ids,
    )
    rows = {cid: emb for cid, emb in cur.fetchall()}
    cur.close()
    conn.close()
    missing = set(compound_ids) - set(rows)
    if missing:
        raise ValueError(
            f"Embeddings missing for {len(missing)} compounds in {table}: "
            f"{sorted(missing)[:5]}..."
        )
    return np.array([rows[cid] for cid in compound_ids])


def _build_umap_split_features(
    embedding_space: str | None, train_ids: list[int]
) -> tuple[np.ndarray | None, str]:
    """Feature matrix + UMAP metric for umap_split_indices, for the given
    embedding-space selector. Returns (None, "jaccard") for the default
    Morgan-FP path (backward-compatible)."""
    if embedding_space in (None, "", "morgan"):
        return None, "jaccard"
    if embedding_space == "mordred":
        return _load_mordred_split_matrix(train_ids), "cosine"
    if embedding_space == "morgan_mordred":
        # Column-wise z-score Mordred so it is on the same ~O(1) scale as
        # each binary Morgan bit; concatenate. Use cosine so that the
        # angular distance reflects contributions from both spaces roughly
        # proportionally. NaN/inf in Mordred -> 0 before scaling (same
        # rule as the mordred-only branch).
        from splits import _morgan_fp_matrix

        train_smiles = _train_smiles_for_ids(train_ids)
        morgan = _morgan_fp_matrix(train_smiles)
        mordred = _load_mordred_split_matrix(train_ids)
        # z-score Mordred
        mordred_mean = mordred.mean(axis=0, keepdims=True)
        mordred_std = mordred.std(axis=0, keepdims=True)
        mordred_std[mordred_std == 0] = 1.0
        mordred_z = (mordred - mordred_mean) / mordred_std
        combined = np.concatenate(
            [morgan.astype(np.float32), mordred_z.astype(np.float32)], axis=1
        )
        return combined, "cosine"
    if embedding_space in EMBEDDING_TABLES:
        mat = load_embeddings(EMBEDDING_TABLES[embedding_space], train_ids)
        return mat.astype(np.float32), "cosine"
    raise ValueError(
        f"Unknown --embedding-space {embedding_space!r}. "
        f"Use 'morgan' (default), 'mordred', 'morgan_mordred', or one of "
        f"{list(EMBEDDING_TABLES)}."
    )


def _load_mordred_split_matrix(train_ids: list[int]) -> np.ndarray:
    """Load Mordred descriptors aligned to train_ids, NaN/inf -> 0.
    Used only for clustering, never as model features."""
    mordred_train, _ = load_train_mordred()
    missing = set(train_ids) - set(mordred_train.index)
    if missing:
        raise ValueError(
            f"Mordred missing for {len(missing)} train compounds when "
            f"building split features"
        )
    mat = mordred_train.loc[train_ids].to_numpy(dtype=np.float32)
    return np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)


def _train_smiles_for_ids(train_ids: list[int]) -> list[str]:
    """Return std_smiles aligned to train_ids order (for Morgan FP
    construction in combined split spaces)."""
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    ph = ",".join(["%s"] * len(train_ids))
    cur.execute(
        f"SELECT id, std_smiles FROM compounds WHERE id IN ({ph})",
        train_ids,
    )
    rows = {cid: s for cid, s in cur.fetchall()}
    cur.close()
    conn.close()
    return [rows[cid] for cid in train_ids]


def load_features(feature_name: str, train_df, test_df):
    """Load feature matrices for train and test."""
    train_ids = load_compound_ids("train")
    test_ids = load_compound_ids("test")

    if feature_name == "rdkit_desc_full":
        train_full = load_rdkit_full(train_ids)
        test_full = load_rdkit_full(test_ids)
        missing_tr = set(train_ids) - set(train_full.index)
        missing_te = set(test_ids) - set(test_full.index)
        if missing_tr or missing_te:
            raise ValueError(
                f"rdkit_desc_full missing: train={len(missing_tr)}, test={len(missing_te)}"
            )
        common_cols = train_full.columns.intersection(test_full.columns)
        X_train = train_full.loc[train_ids, common_cols].to_numpy(dtype=np.float32)
        X_test = test_full.loc[test_ids, common_cols].to_numpy(dtype=np.float32)
        print(
            f"  rdkit_desc_full loaded: {X_train.shape[1]} descriptor cols "
            f"(NaN handled natively by LightGBM/XGBoost; CatBoost requires impute)"
        )
        return X_train, X_test

    if feature_name == "jazzy":
        jazzy_train = load_jazzy(train_ids).reindex(index=train_ids)
        jazzy_test = load_jazzy(test_ids).reindex(index=test_ids)
        X_train = jazzy_train[list(JAZZY_FEATURE_COLS)].to_numpy(dtype=np.float32)
        X_test = jazzy_test[list(JAZZY_FEATURE_COLS)].to_numpy(dtype=np.float32)
        return X_train, X_test

    if feature_name in ("mordred", "mordred_singleconc", "mordred_jazzy"):
        mordred_train, _ = load_train_mordred()
        mordred_test = load_mordred(test_ids)
        missing_train = set(train_ids) - set(mordred_train.index)
        missing_test = set(test_ids) - set(mordred_test.index)
        if missing_train:
            raise ValueError(
                f"Mordred missing for {len(missing_train)} train compounds"
            )
        if missing_test:
            raise ValueError(f"Mordred missing for {len(missing_test)} test compounds")
        common_cols = mordred_train.columns.intersection(mordred_test.columns)
        X_train = mordred_train.loc[train_ids, common_cols].values.astype(np.float32)
        X_test = mordred_test.loc[test_ids, common_cols].values.astype(np.float32)
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

        if feature_name == "mordred_jazzy":
            jazzy_train = load_jazzy(train_ids).reindex(index=train_ids)
            jazzy_test = load_jazzy(test_ids).reindex(index=test_ids)
            missing_j_tr = int(jazzy_train.isna().any(axis=1).sum())
            missing_j_te = int(jazzy_test.isna().any(axis=1).sum())
            if missing_j_tr or missing_j_te:
                raise ValueError(
                    f"Jazzy missing rows: train={missing_j_tr}, test={missing_j_te}"
                )
            jt_tr = jazzy_train[list(JAZZY_FEATURE_COLS)].to_numpy(dtype=np.float32)
            jt_te = jazzy_test[list(JAZZY_FEATURE_COLS)].to_numpy(dtype=np.float32)
            X_train = np.concatenate([X_train, jt_tr], axis=1)
            X_test = np.concatenate([X_test, jt_te], axis=1)
            print(
                f"  jazzy concat: +{jt_tr.shape[1]} cols (train+test both fully populated)"
            )

        if feature_name == "mordred_singleconc":
            from data import load_singleconc_features

            sc_train = (
                load_singleconc_features(train_ids)
                .loc[train_ids]
                .to_numpy(dtype=np.float32)
            )
            sc_test = (
                load_singleconc_features(test_ids)
                .loc[test_ids]
                .to_numpy(dtype=np.float32)
            )
            # NaN preserved (LightGBM handles missing natively)
            X_train = np.concatenate([X_train, sc_train], axis=1)
            X_test = np.concatenate([X_test, sc_test], axis=1)
            n_sc_train_with_data = (~np.isnan(sc_train).all(axis=1)).sum()
            print(
                f"  single_conc concat: +{sc_train.shape[1]} cols, "
                f"{n_sc_train_with_data}/{len(train_ids)} train compounds have data"
            )

        return X_train, X_test

    if feature_name == "pooled_boltz_allpairs":
        # Same shape as `pooled_boltz` (1024 features) but the z pool
        # spans ALL protein x ligand pairs (434 PXR residues x L ligand
        # atoms) rather than just 13 core pocket residues. Matches the
        # cross_pair_mask used by Boltz-2's affinity head (affinity.py).
        ap_path = REPO_ROOT.joinpath("data", "boltz_affhead", "pooled_allpairs.parquet")
        if not ap_path.exists():
            raise SystemExit(
                f"Missing {ap_path}. Run "
                f"track1_activity/scripts/boltz_affhead/01b_pool_allpairs.py"
            )
        ap_df = pd.read_parquet(ap_path).set_index("compound_id")

        def _allpairs_matrix(ids):
            X = ap_df.reindex(index=ids).to_numpy(dtype=np.float32).copy()
            col_mean = np.nanmean(X, axis=0)
            if np.isnan(col_mean).any():
                col_mean_global = ap_df.to_numpy(dtype=np.float32).mean(axis=0)
                col_mean = np.where(np.isnan(col_mean), col_mean_global, col_mean)
            inds = np.where(np.isnan(X))
            X[inds] = np.take(col_mean, inds[1])
            return X

        X_train = _allpairs_matrix(train_ids)
        X_test = _allpairs_matrix(test_ids)
        print(
            f"  pooled_boltz_allpairs: s_prot 384 + s_lig 384 + z_xp_mean 128 "
            f"+ z_xp_max 128 = {X_train.shape[1]} features "
            f"(z pool over all 434 PXR residues x ligand atoms)"
        )
        return X_train, X_test

    _seed_match = re.match(r"^2d_full_boltz_log2fc_pred_seed(\d+)ens$", feature_name)
    if (
        feature_name
        in (
            "2d_full_boltz_log2fc_pred",
            "2d_full_boltz_log2fc_pred_ens4",
        )
        or _seed_match is not None
    ):
        # 2d_full_boltz (1817d) + 2 predicted log2_fc scalars.
        # Buterez 2024 strategy-2: predicted LF labels as side feature.
        # - "2d_full_boltz_log2fc_pred": chemprop-only log2fc predictor
        #   (baseline, single seed=42). See run_chemprop_predict_log2fc.py.
        # - "2d_full_boltz_log2fc_pred_ens4": inverse-val-loss weighted
        #   mean of 4 encoders (chemprop + molformer_c3 + attentivefp +
        #   gatedgcn). See build_ensemble_log2fc.py (#115).
        # - "2d_full_boltz_log2fc_pred_seed5ens": same chemprop arch
        #   trained with seeds [42,43,44,45,46] then averaged. Variance
        #   reduction trick (Plan A 2026-04-25). Multi-arch ens4 was null
        #   per #116 because weak encoders diluted chemprop; same-arch
        #   multi-seed only averages noise so should be at least as good
        #   as single seed.
        X_train_base, X_test_base = load_features("2d_full_boltz", train_df, test_df)
        if feature_name == "2d_full_boltz_log2fc_pred":
            lf_filename = "chemprop_pretrain_log2fc_predictions.parquet"
            rebuild_hint = "track1_activity/scripts/run_chemprop_predict_log2fc.py"
        elif _seed_match is not None:
            n_seeds = _seed_match.group(1)
            lf_filename = (
                f"chemprop_pretrain_log2fc_predictions_seed{n_seeds}ens.parquet"
            )
            rebuild_hint = "track1_activity/scripts/build_log2fc_seed_ensemble.py"
        else:
            lf_filename = "ensemble4_log2fc_predictions.parquet"
            rebuild_hint = "track1_activity/scripts/build_ensemble_log2fc.py"
        lf_path = REPO_ROOT.joinpath("data", lf_filename)
        if not lf_path.exists():
            raise SystemExit(f"Missing {lf_path}. Run {rebuild_hint}")
        lf_df = pd.read_parquet(lf_path)
        cols = ["log2fc_8p25_pred", "log2fc_33_pred"]
        Xl_tr = lf_df.reindex(index=train_ids)[cols].to_numpy(dtype=np.float32).copy()
        Xl_te = lf_df.reindex(index=test_ids)[cols].to_numpy(dtype=np.float32).copy()
        Xl_tr = np.nan_to_num(Xl_tr, nan=0.0, posinf=0.0, neginf=0.0)
        Xl_te = np.nan_to_num(Xl_te, nan=0.0, posinf=0.0, neginf=0.0)
        X_train = np.concatenate([X_train_base, Xl_tr], axis=1)
        X_test = np.concatenate([X_test_base, Xl_te], axis=1)
        print(
            f"  {feature_name}: {X_train_base.shape[1]} base + "
            f"{Xl_tr.shape[1]} LF preds = {X_train.shape[1]} features"
        )
        return X_train, X_test

    if feature_name == "cconcat_2d_full_boltz_log2fc_pred":
        # SWAP: replace chemeleon (300d) with Boltz C-concat pretrain embed
        # (1024d) in the cheme_2df recipe. Tests whether TabPFN benefits
        # mainly from Boltz-trunk-pretrained dims vs chemeleon's foundation
        # FP dims. Expected ambiguous: chemeleon is ZINC-scale decorrelating
        # signal, C-concat is Boltz-family (high redundancy with 2d_full_boltz
        # tier-0/1). Quick ablation per 2026-04-22 evening.
        X_train_base, X_test_base = load_features(
            "2d_full_boltz_log2fc_pred", train_df, test_df
        )
        embed_path = REPO_ROOT.joinpath(
            "data", "boltz_trunk_pretrain_embed_c_concat.parquet"
        )
        emb_df = pd.read_parquet(embed_path).set_index("compound_id")
        c_tr = emb_df.reindex(index=train_ids).to_numpy(dtype=np.float32).copy()
        c_te = emb_df.reindex(index=test_ids).to_numpy(dtype=np.float32).copy()
        for X in (c_tr, c_te):
            cm = np.nanmean(X, axis=0)
            cm = np.where(np.isfinite(cm), cm, 0.0)
            X[~np.isfinite(X)] = np.broadcast_to(cm, X.shape)[~np.isfinite(X)]
        X_train = np.concatenate([c_tr, X_train_base], axis=1)
        X_test = np.concatenate([c_te, X_test_base], axis=1)
        print(
            f"  cconcat_2d_full_boltz_log2fc_pred: {c_tr.shape[1]} c_concat + "
            f"{X_train_base.shape[1]} 2df_lf = {X_train.shape[1]} features"
        )
        return X_train, X_test

    if feature_name == "cheme_cconcat_2d_full_boltz_log2fc_pred":
        # ADD: chemeleon (300d) + C-concat (1024d) + 2d_full_boltz_log2fc_pred
        # (1803d) = 3127d. Tests if both foundation FPs stack (they're from
        # different pretrain corpora: ZINC vs PXR single_conc log2_fc).
        X_train_base, X_test_base = load_features(
            "cheme_2d_full_boltz_log2fc_pred", train_df, test_df
        )
        embed_path = REPO_ROOT.joinpath(
            "data", "boltz_trunk_pretrain_embed_c_concat.parquet"
        )
        emb_df = pd.read_parquet(embed_path).set_index("compound_id")
        c_tr = emb_df.reindex(index=train_ids).to_numpy(dtype=np.float32).copy()
        c_te = emb_df.reindex(index=test_ids).to_numpy(dtype=np.float32).copy()
        for X in (c_tr, c_te):
            cm = np.nanmean(X, axis=0)
            cm = np.where(np.isfinite(cm), cm, 0.0)
            X[~np.isfinite(X)] = np.broadcast_to(cm, X.shape)[~np.isfinite(X)]
        X_train = np.concatenate([c_tr, X_train_base], axis=1)
        X_test = np.concatenate([c_te, X_test_base], axis=1)
        print(
            f"  cheme_cconcat_2d_full_boltz_log2fc_pred: {c_tr.shape[1]} c_concat + "
            f"{X_train_base.shape[1]} cheme_2df_lf = {X_train.shape[1]} features"
        )
        return X_train, X_test

    _cheme_seed_match = re.match(
        r"^cheme_2d_full_boltz_log2fc_pred_seed(\d+)ens$", feature_name
    )
    if (
        feature_name
        in (
            "cheme_2d_full_boltz_log2fc_pred",
            "cheme_2d_full_boltz_log2fc_pred_ens4",
        )
        or _cheme_seed_match is not None
    ):
        # Chemeleon (300d) + 2d_full_boltz_log2fc_pred (1803d) = 2103d.
        # Empirically discovered via mix-feature bakeoff 2026-04-21: this
        # concat is a strict supersede of 2d_full_boltz_log2fc_pred alone
        # (single-model OOF MAE 0.4234 vs 0.4427). Chemeleon foundation FP
        # on its own was weak with TabPFN (user-confirmed "iffy"), but
        # mixed with the 2D/Boltz descriptors it adds complementary signal
        # that caruana strongly rewards (post-swap 8-pool MAE 0.4181,
        # cheme_2df alone carries weight 0.388).
        # "_ens4" swaps chemprop-only log2fc_pred for 4-encoder inverse-
        # val-loss weighted ensemble (#115 Phase 3).
        if feature_name.endswith("_ens4"):
            base_name = "2d_full_boltz_log2fc_pred_ens4"
        elif _cheme_seed_match is not None:
            base_name = f"2d_full_boltz_log2fc_pred_seed{_cheme_seed_match.group(1)}ens"
        else:
            base_name = "2d_full_boltz_log2fc_pred"
        X_train_base, X_test_base = load_features(base_name, train_df, test_df)
        # Load chemeleon from DB
        import psycopg2 as _pg

        with _pg.connect(**DB_PARAMS) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT compound_id, embedding FROM compound_chemeleon ORDER BY compound_id"
            )
            rows = cur.fetchall()
        cheme_map = {int(r[0]): np.asarray(r[1], dtype=np.float32) for r in rows}
        cheme_tr = np.stack(
            [cheme_map.get(cid, np.zeros(300, dtype=np.float32)) for cid in train_ids]
        )
        cheme_te = np.stack(
            [cheme_map.get(cid, np.zeros(300, dtype=np.float32)) for cid in test_ids]
        )
        cheme_tr = np.nan_to_num(cheme_tr, nan=0.0, posinf=0.0, neginf=0.0)
        cheme_te = np.nan_to_num(cheme_te, nan=0.0, posinf=0.0, neginf=0.0)
        X_train = np.concatenate([cheme_tr, X_train_base], axis=1)
        X_test = np.concatenate([cheme_te, X_test_base], axis=1)
        print(
            f"  cheme_2d_full_boltz_log2fc_pred: {cheme_tr.shape[1]} chemeleon + "
            f"{X_train_base.shape[1]} 2df_lf = {X_train.shape[1]} features"
        )
        return X_train, X_test

    if feature_name == "cheme_2d_full_boltz_log2fc_emax_pred":
        # cheme_2d_full_boltz_log2fc_pred (2103d) + 2 emax_pred scalars
        # (emax_estimate_pred + emax_vs_pos_ctrl_pred) = 2105d.
        # emax labels are 100% in train_activity (4140/4140) and orthogonal
        # to pec50 (corr ~ -0.13), so emax_pred captures efficacy signal
        # not directly encoded by pool members today. Generated via
        # run_emax_predict.py (LGBM cross-fit on rdkit_desc_full).
        # Same Buterez 2024 strategy-2 pattern as log2fc_pred.
        X_train_base, X_test_base = load_features(
            "cheme_2d_full_boltz_log2fc_pred", train_df, test_df
        )
        emax_path = REPO_ROOT.joinpath("data", "emax_predictions.parquet")
        if not emax_path.exists():
            raise SystemExit(
                f"Missing {emax_path}. Run track1_activity/scripts/run_emax_predict.py"
            )
        emax_df = pd.read_parquet(emax_path)
        cols = ["emax_estimate_pred", "emax_vs_pos_ctrl_pred"]
        Xe_tr = emax_df.reindex(index=train_ids)[cols].to_numpy(dtype=np.float32).copy()
        Xe_te = emax_df.reindex(index=test_ids)[cols].to_numpy(dtype=np.float32).copy()
        Xe_tr = np.nan_to_num(Xe_tr, nan=0.0, posinf=0.0, neginf=0.0)
        Xe_te = np.nan_to_num(Xe_te, nan=0.0, posinf=0.0, neginf=0.0)
        X_train = np.concatenate([X_train_base, Xe_tr], axis=1)
        X_test = np.concatenate([X_test_base, Xe_te], axis=1)
        print(
            f"  cheme_2d_full_boltz_log2fc_emax_pred: {X_train_base.shape[1]} cheme_2df_lf + "
            f"{Xe_tr.shape[1]} emax_pred = {X_train.shape[1]} features"
        )
        return X_train, X_test

    if feature_name == "oe_cheme_2d_full_boltz_log2fc_pred":
        # OpenEye full bundle (23) + chemeleon 300 + 2d_full_boltz_log2fc_pred
        # 1803 = 2126. Intended as the base for top-K LGBM gain selection;
        # per-fold selection naturally drops redundant OE scalars (xlogp is
        # redundant with log2fc_pred at 0.09% gain share, quacpac at 0.01%,
        # most oemedchem at <0.05%) while keeping ROCS 6 features (all
        # rank in overall top 30 in eda_oe_vs_2dfull_log2fc.py, 2026-04-23).
        # Constant anionic_carbon_count is the only explicitly excluded
        # oemedchem scalar.
        X_train_base, X_test_base = load_features(
            "cheme_2d_full_boltz_log2fc_pred", train_df, test_df
        )
        oemedchem_cols = [
            "xlogp",
            "psa_2d",
            "mw",
            "hba",
            "hbd",
            "lipinski_hba",
            "lipinski_hbd",
            "aromatic_ring_count",
            "rotatable_bond_count",
            "fraction_csp3",
            "halide_fraction",
            "longest_unbranched_c_chain",
            "longest_unbranched_heavy_chain",
            "num_unspecified_atom_stereo",
            "num_unspecified_bond_stereo",
        ]  # anionic_carbon_count dropped (constant 0 on PXR)
        rocs_cols = [
            "max_shape_tanimoto",
            "max_color_tanimoto",
            "max_combo_tanimoto",
            "mean_shape_tanimoto",
            "mean_color_tanimoto",
            "mean_combo_tanimoto",
        ]
        import psycopg2 as _pg

        with _pg.connect(**DB_PARAMS) as conn:
            oe_df = pd.read_sql(
                f"SELECT compound_id, {','.join(oemedchem_cols)} FROM compound_oemedchem",
                conn,
            ).set_index("compound_id")
            rocs_df = pd.read_sql(
                f"SELECT compound_id, {','.join(rocs_cols)} FROM compound_rocs",
                conn,
            ).set_index("compound_id")
            quacpac_df = pd.read_sql(
                "SELECT q.compound_id, q.formal_charge, t.n_tautomers "
                "FROM compound_quacpac q "
                "JOIN compound_tautomers t ON t.compound_id = q.compound_id",
                conn,
            ).set_index("compound_id")
        quacpac_cols = ["formal_charge", "n_tautomers"]

        def _stack(df, cols, ids):
            X = df.reindex(ids)[cols].to_numpy(dtype=np.float32)
            cm = np.nanmean(X, axis=0)
            cm = np.where(np.isfinite(cm), cm, 0.0)
            return np.where(np.isfinite(X), X, cm).astype(np.float32)

        oe_tr = _stack(oe_df, oemedchem_cols, train_ids)
        oe_te = _stack(oe_df, oemedchem_cols, test_ids)
        rocs_tr = _stack(rocs_df, rocs_cols, train_ids)
        rocs_te = _stack(rocs_df, rocs_cols, test_ids)
        qp_tr = _stack(quacpac_df, quacpac_cols, train_ids)
        qp_te = _stack(quacpac_df, quacpac_cols, test_ids)

        X_train = np.concatenate([oe_tr, rocs_tr, qp_tr, X_train_base], axis=1)
        X_test = np.concatenate([oe_te, rocs_te, qp_te, X_test_base], axis=1)
        print(
            f"  oe_cheme_2d_full_boltz_log2fc_pred: "
            f"oemedchem {oe_tr.shape[1]} + rocs {rocs_tr.shape[1]} + "
            f"quacpac {qp_tr.shape[1]} + cheme_2df_lf {X_train_base.shape[1]} "
            f"= {X_train.shape[1]} features"
        )
        return X_train, X_test

    if feature_name == "3d_ligand_omega":
        # Same descriptor set as 3d_ligand (RDKit scalar 11 + vector 973)
        # but computed on OpenEye Omega conformers (lowest-energy per mol
        # from structures/omega/<id>.sdf) instead of Boltz-2 docked poses.
        # Omega = solution-state statistical sampling, Boltz = docked
        # bound-state. Different conformer source -> different descriptor
        # values = complementary information axis. skfp3d + mordred3d
        # not included (need separate venv / additional compute).
        # Total: 11 + 973 = 984 dims.
        scalar_cols = [
            "asphericity",
            "eccentricity",
            "inertial_shape_factor",
            "npr1",
            "npr2",
            "pmi1",
            "pmi2",
            "pmi3",
            "radius_of_gyration",
            "spherocity_index",
            "pbf",
        ]
        with psycopg2.connect(**DB_PARAMS) as conn:
            scalar_df = pd.read_sql(
                f"SELECT compound_id, {', '.join(scalar_cols)} "
                f"FROM compound_omega_desc3d",
                conn,
            ).set_index("compound_id")
            vec_df = pd.read_sql(
                "SELECT compound_id, autocorr3d, getaway, morse, rdf, "
                "whim, usr, usrcat FROM compound_omega_desc3d_vector",
                conn,
            ).set_index("compound_id")

        def _fill_vec(series, dim):
            out = []
            for v in series:
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    out.append(np.zeros(dim, dtype=np.float32))
                else:
                    a = np.asarray(v, dtype=np.float64)
                    a = np.where(np.isnan(a) | np.isinf(a), 0.0, a)
                    out.append(a.astype(np.float32))
            return np.stack(out, axis=0)

        def _build(ids):
            s = scalar_df.reindex(ids)[scalar_cols].to_numpy(dtype=np.float32)
            s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
            v = vec_df.reindex(ids)
            autocorr = _fill_vec(v["autocorr3d"], 80)
            getaway = _fill_vec(v["getaway"], 273)
            morse = _fill_vec(v["morse"], 224)
            rdf = _fill_vec(v["rdf"], 210)
            whim = _fill_vec(v["whim"], 114)
            usr = _fill_vec(v["usr"], 12)
            usrcat = _fill_vec(v["usrcat"], 60)
            return np.concatenate(
                [s, autocorr, getaway, morse, rdf, whim, usr, usrcat], axis=1
            )

        X_train = _build(train_ids)
        X_test = _build(test_ids)
        print(f"  3d_ligand_omega: scalar 11 + vector 973 = {X_train.shape[1]} dims")
        return X_train, X_test

    if feature_name == "unimol_v2_3d_omega":
        # Finetuned Uni-Mol v2 CLS/mean/max concat (2304d) +
        # 3d_ligand_omega (984d, Omega conformer 3D desc) = 3288d.
        # Base for per-fold top-K LGBM-gain selection. Omega conformers
        # are solution-state; complements Uni-Mol's internal 3D attention.
        X_uni_tr, X_uni_te = load_features(
            "unimol_v2_pretrain_embed", train_df, test_df
        )
        X_3d_tr, X_3d_te = load_features("3d_ligand_omega", train_df, test_df)
        X_train = np.concatenate([X_uni_tr, X_3d_tr], axis=1).astype(np.float32)
        X_test = np.concatenate([X_uni_te, X_3d_te], axis=1).astype(np.float32)
        print(
            f"  unimol_v2_3d_omega: unimol_v2 {X_uni_tr.shape[1]} + "
            f"3d_ligand_omega {X_3d_tr.shape[1]} = {X_train.shape[1]} dims"
        )
        return X_train, X_test

    if feature_name == "unimol_v2_3ddesc":
        # unimol_v2_pretrain_embed (2304d concat cls/mean/max) +
        # 3d_ligand (1212d Boltz-2 pose descriptors) = 3516d. Intended
        # as base for top-K LGBM-gain selection; per-fold selection
        # keeps the signal-bearing dims and drops redundant ones.
        X_uni_tr, X_uni_te = load_features(
            "unimol_v2_pretrain_embed", train_df, test_df
        )
        X_3d_tr, X_3d_te = load_features("3d_ligand", train_df, test_df)
        X_train = np.concatenate([X_uni_tr, X_3d_tr], axis=1).astype(np.float32)
        X_test = np.concatenate([X_uni_te, X_3d_te], axis=1).astype(np.float32)
        print(
            f"  unimol_v2_3ddesc: unimol_v2 {X_uni_tr.shape[1]} + "
            f"3d_ligand {X_3d_tr.shape[1]} = {X_train.shape[1]} dims"
        )
        return X_train, X_test

    if feature_name == "unimol_v2_pretrain_embed":
        # 768d CLS representation from Uni-Mol v2 (84M). Extracted via
        # unimol_tools.UniMolRepr.get_repr() on all 13,136 std_smiles;
        # see track1_activity/scripts/unimol/03_extract_repr.sh and
        # 04_npz_to_parquet.py (PR feature/unimol-etkdg-pretrain-embed).
        #
        # CAVEAT: our Task 3 log2_fc finetune (kfold=2, 30 epochs) hit
        # Pearson 0.09 / R^2 -0.003 on the pretrain task, and
        # unimol_tools UniMolRepr auto-downloaded the PUBLIC Uni-Mol v2
        # checkpoint (ignoring our finetuned .pth files which have a
        # different serialization format). So this embedding is
        # effectively the BASE Uni-Mol v2 representation, NOT our
        # log2_fc finetune. That is probably fine given the finetune
        # was weak; a separate finetuned variant can be added later
        # by bridging the checkpoint format.
        embed_path = REPO_ROOT.joinpath("data", "unimol_v2_pretrain_embed.parquet")
        if not embed_path.exists():
            raise SystemExit(
                f"Missing {embed_path}. Run "
                f"track1_activity/scripts/unimol/03_extract_repr.sh then "
                f"04_npz_to_parquet.py"
            )
        emb_df = pd.read_parquet(embed_path)
        X_train = emb_df.reindex(index=train_ids).to_numpy(dtype=np.float32).copy()
        X_test = emb_df.reindex(index=test_ids).to_numpy(dtype=np.float32).copy()
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
        print(
            f"  unimol_v2_pretrain_embed: {X_train.shape[1]} dims "
            f"(train {X_train.shape[0]} / test {X_test.shape[0]})"
        )
        return X_train, X_test

    if feature_name == "chemprop_pretrain_embed":
        # 256d per-compound fingerprints from MPNN.fingerprint() of the
        # chemprop pretrain checkpoint (track1_activity/checkpoints/
        # chemprop_pretrain/pretrain.pt). See
        # track1_activity/scripts/run_chemprop_embed_extract.py.
        # Buterez 2024 strategy-3: use the LF model's molecule-level
        # embedding as a side feature for a HF tabular regressor
        # (TabPFN / LGBM).
        embed_path = REPO_ROOT.joinpath("data", "chemprop_pretrain_embed.parquet")
        if not embed_path.exists():
            raise SystemExit(
                f"Missing {embed_path}. Run "
                f"track1_activity/scripts/run_chemprop_embed_extract.py"
            )
        emb_df = pd.read_parquet(embed_path)
        X_train = emb_df.reindex(index=train_ids).to_numpy(dtype=np.float32).copy()
        X_test = emb_df.reindex(index=test_ids).to_numpy(dtype=np.float32).copy()
        # All 4653 compounds are covered, so no NaN expected -- but be
        # defensive in case of future partial coverage.
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
        print(
            f"  chemprop_pretrain_embed: {X_train.shape[1]} dims "
            f"(train {X_train.shape[0]} / test {X_test.shape[0]})"
        )
        return X_train, X_test

    if feature_name == "molformer_c3_pretrain_embed":
        # 768d per-compound [CLS] embeddings from MoLFormer-c3 pretrain
        # LoRA checkpoint. See
        # track1_activity/scripts/run_molformer_c3_pretrain.py and
        # track1_activity/scripts/run_molformer_c3_embed_extract.py.
        # Buterez 2024 strategy-3 with a transformer-family backbone
        # (parallel to chemprop_pretrain_embed which uses a GNN).
        embed_path = REPO_ROOT.joinpath("data", "molformer_c3_pretrain_embed.parquet")
        if not embed_path.exists():
            raise SystemExit(
                f"Missing {embed_path}. Run "
                f"track1_activity/scripts/run_molformer_c3_embed_extract.py"
            )
        emb_df = pd.read_parquet(embed_path)
        X_train = emb_df.reindex(index=train_ids).to_numpy(dtype=np.float32).copy()
        X_test = emb_df.reindex(index=test_ids).to_numpy(dtype=np.float32).copy()
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
        print(
            f"  molformer_c3_pretrain_embed: {X_train.shape[1]} dims "
            f"(train {X_train.shape[0]} / test {X_test.shape[0]})"
        )
        return X_train, X_test

    if feature_name == "chemberta_5m_mtr_pretrain_embed":
        # 384d per-compound [CLS] embeddings from ChemBERTa-5M-MTR (RoBERTa
        # 3L, ~5M params) continued-pretrained on 2-head log2_fc via LoRA.
        # See run_chemberta_5m_mtr_pretrain.py + _embed_extract.py.
        # Phase B of #100 BERT-family audit: Phase B1 raw audit found
        # 5m_mtr was the best raw (MAE 0.5287 / min-r 0.77) but caruana
        # ADD was only -0.002. This continued-pretrain should tighten the
        # single-model gap to pool weakest.
        embed_path = REPO_ROOT.joinpath(
            "data", "chemberta_5m_mtr_pretrain_embed.parquet"
        )
        if not embed_path.exists():
            raise SystemExit(
                f"Missing {embed_path}. Run "
                f"track1_activity/scripts/run_chemberta_5m_mtr_embed_extract.py"
            )
        emb_df = pd.read_parquet(embed_path)
        X_train = emb_df.reindex(index=train_ids).to_numpy(dtype=np.float32).copy()
        X_test = emb_df.reindex(index=test_ids).to_numpy(dtype=np.float32).copy()
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
        print(
            f"  chemberta_5m_mtr_pretrain_embed: {X_train.shape[1]} dims "
            f"(train {X_train.shape[0]} / test {X_test.shape[0]})"
        )
        return X_train, X_test

    if feature_name == "kermt_pretrain_embed":
        # 3200d per-compound graph embedding from KERMT (GROVER_base)
        # after continued-pretrain on single_concentration log2_fc.
        # Dims = 4 GROVER heads (atom_from_atom, atom_from_bond,
        # bond_from_atom, bond_from_bond) x hidden=800 with
        # --fingerprint_source both.
        # See:
        #   track1_activity/scripts/run_kermt_pretrain.sh
        #   track1_activity/scripts/run_kermt_embed_extract.sh
        #   track1_activity/scripts/kermt_embed_npz_to_parquet.py
        # Buterez 2024 strategy-3 with a graph-transformer backbone
        # (parallel to chemprop_pretrain_embed = GNN,
        # molformer_c3_pretrain_embed = transformer).
        embed_path = REPO_ROOT.joinpath("data", "kermt_pretrain_embed.parquet")
        if not embed_path.exists():
            raise SystemExit(
                f"Missing {embed_path}. Run "
                f"track1_activity/scripts/kermt_embed_npz_to_parquet.py"
            )
        emb_df = pd.read_parquet(embed_path)
        X_train = emb_df.reindex(index=train_ids).to_numpy(dtype=np.float32).copy()
        X_test = emb_df.reindex(index=test_ids).to_numpy(dtype=np.float32).copy()
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
        print(
            f"  kermt_pretrain_embed: {X_train.shape[1]} dims "
            f"(train {X_train.shape[0]} / test {X_test.shape[0]})"
        )
        return X_train, X_test

    if feature_name == "attentivefp_pretrain_embed":
        # 512d graph-readout embedding from PyG AttentiveFP pretrained on
        # single_concentration log2_fc (2-head, 90/10 random split, z-scored
        # targets). Extracted by replacing model.lin2 with nn.Identity() so
        # forward returns post-GRU pre-projection representation. See:
        #   track1_activity/scripts/run_attentivefp_embed_extract.py
        # Buterez 2024 strategy-3 with graph-attention backbone
        # (parallel to chemprop_pretrain_embed = D-MPNN,
        # molformer_c3_pretrain_embed = transformer,
        # kermt_pretrain_embed = graph-transformer).
        embed_path = REPO_ROOT.joinpath("data", "attentivefp_pretrain_embed.parquet")
        if not embed_path.exists():
            raise SystemExit(
                f"Missing {embed_path}. Run "
                f"track1_activity/scripts/run_attentivefp_embed_extract.py"
            )
        emb_df = pd.read_parquet(embed_path)
        X_train = emb_df.reindex(index=train_ids).to_numpy(dtype=np.float32).copy()
        X_test = emb_df.reindex(index=test_ids).to_numpy(dtype=np.float32).copy()
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
        print(
            f"  attentivefp_pretrain_embed: {X_train.shape[1]} dims "
            f"(train {X_train.shape[0]} / test {X_test.shape[0]})"
        )
        return X_train, X_test

    if feature_name == "gatedgcn_pretrain_embed":
        # 512d graph-pooled embedding from PyG ResGatedGraphConv stack
        # pretrained on single_concentration log2_fc (2-head, z-scored
        # targets, hidden_dim=512, batch=64). Extracted by replacing
        # GatedGCNModel.ffn with nn.Identity() so forward returns the
        # global_mean_pool output.
        # See: track1_activity/scripts/run_gatedgcn_embed_extract.py
        # Buterez 2024 strategy-3 with gated edge-conditioned message
        # passing backbone (fifth pretrain-embed family member).
        # Note: PR #79 h128 checkpoint (val_loss 0.7394) was upgraded to
        # h512 (val_loss 0.7478) to bring single-model OOF MAE from
        # 0.4902 to 0.4740. h128 backup retained at
        # checkpoints/gatedgcn_pretrain/pretrain_h128.pt.
        embed_path = REPO_ROOT.joinpath("data", "gatedgcn_pretrain_embed.parquet")
        if not embed_path.exists():
            raise SystemExit(
                f"Missing {embed_path}. Run "
                f"track1_activity/scripts/run_gatedgcn_embed_extract.py"
            )
        emb_df = pd.read_parquet(embed_path)
        X_train = emb_df.reindex(index=train_ids).to_numpy(dtype=np.float32).copy()
        X_test = emb_df.reindex(index=test_ids).to_numpy(dtype=np.float32).copy()
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
        print(
            f"  gatedgcn_pretrain_embed: {X_train.shape[1]} dims "
            f"(train {X_train.shape[0]} / test {X_test.shape[0]})"
        )
        return X_train, X_test

    if feature_name.startswith("boltz_trunk_pretrain_embed_"):
        # 256d embedding from the Boltz-2 trunk MLP pretrain head
        # (Buterez 2024 strategy-3 on the Boltz backbone, issue #109).
        # Variant suffix: _a simple MLP, _b wider MLP, _c PairFormer-inspired
        # transformer over 4 pooled tokens (s_prot / s_lig / z_mean / z_max).
        # Input = 13134-compound 1024d all-pairs pooled trunk vectors
        # (4652 rcycle=3 from the original full run + 8482 rcycle=1 from the
        # fast embeddings-only run) stored in compound_boltz2_trunk_fast.
        # See:
        #   track1_activity/scripts/boltz_affhead/09_mlp_pretrain.py
        #   track1_activity/scripts/boltz_affhead/09b_extract_embed.py
        # Sixth pretrain-embed family member (parallel to chemprop D-MPNN,
        # molformer transformer, kermt graph-transformer, attentivefp
        # graph-attention, gatedgcn gated-MPNN) and the first with explicit
        # protein-ligand structural context.
        # Suffix after "boltz_trunk_pretrain_embed_" maps directly to the
        # parquet filename: "a", "b", "c", or "b_first" (B's 512d
        # intermediate). Variant letter is the first token.
        suffix = feature_name[len("boltz_trunk_pretrain_embed_") :]
        variant = suffix.split("_", 1)[0]
        embed_path = REPO_ROOT.joinpath(
            "data", f"boltz_trunk_pretrain_embed_{suffix}.parquet"
        )
        if not embed_path.exists():
            raise SystemExit(
                f"Missing {embed_path}. Run "
                f"track1_activity/scripts/boltz_affhead/09b_extract_embed.py "
                f"--variant {variant}"
            )
        emb_df = pd.read_parquet(embed_path).set_index("compound_id")
        X_train = emb_df.reindex(index=train_ids).to_numpy(dtype=np.float32).copy()
        X_test = emb_df.reindex(index=test_ids).to_numpy(dtype=np.float32).copy()
        # Coverage caveat: 01576 / 01657 / 03840 are Boltz preprocessing
        # failures absent from compound_boltz2_trunk_fast and will show up
        # here as NaN rows. Fill with column means so TabPFN has well-
        # defined rows (same convention as pooled_boltz below).
        col_means_train = np.nanmean(X_train, axis=0)
        col_means_train = np.where(np.isfinite(col_means_train), col_means_train, 0.0)
        X_train = np.where(np.isfinite(X_train), X_train, col_means_train)
        X_test = np.where(np.isfinite(X_test), X_test, col_means_train)
        print(
            f"  {feature_name}: {X_train.shape[1]} dims "
            f"(train {X_train.shape[0]} / test {X_test.shape[0]})"
        )
        return X_train, X_test

    if feature_name.startswith("pooled_boltz_ab_"):
        # Ablation subsets of the raw all-pairs pooled trunk. Suffix:
        #   sonly  = s_prot_mean + s_lig_mean      (768d)
        #   zonly  = z_xp_mean + z_xp_max          (256d)
        #   sprot  = s_prot_mean                   (384d)
        #   slig   = s_lig_mean                    (384d)
        #   zmean  = z_xp_mean                     (128d)
        #   zmax   = z_xp_max                      (128d)
        # Used to diagnose which component(s) of the 1024d trunk carry
        # the downstream signal. s/z ablation (option 7) per issue #109.
        subset = feature_name[len("pooled_boltz_ab_") :]
        prefix_sets = {
            "sonly": ["s_prot_mean", "s_lig_mean"],
            "zonly": ["z_xp_mean", "z_xp_max"],
            "sprot": ["s_prot_mean"],
            "slig": ["s_lig_mean"],
            "zmean": ["z_xp_mean"],
            "zmax": ["z_xp_max"],
            # 896d: full 1024d minus z_xp_max (ablation showed z_xp_max is
            # ~noise -- solo MAE 0.5755, the worst single component). Drop
            # it as a quick de-noising experiment.
            "nozmax": ["s_prot_mean", "s_lig_mean", "z_xp_mean"],
        }
        if subset not in prefix_sets:
            raise SystemExit(
                f"Unknown pooled_boltz_ab subset '{subset}'. Choices: {list(prefix_sets)}"
            )
        raw_path = REPO_ROOT.joinpath(
            "data", "boltz_affhead", "pooled_allpairs.parquet"
        )
        if not raw_path.exists():
            raise SystemExit(f"Missing {raw_path}")
        raw_df = pd.read_parquet(raw_path).set_index("compound_id")
        keep_cols = [
            c
            for c in raw_df.columns
            if any(c.startswith(p) for p in prefix_sets[subset])
        ]
        sub_df = raw_df[keep_cols]

        def _sub_matrix(ids):
            X = sub_df.reindex(index=ids).to_numpy(dtype=np.float32).copy()
            col_mean = np.nanmean(X, axis=0)
            col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
            X[~np.isfinite(X)] = np.broadcast_to(col_mean, X.shape)[~np.isfinite(X)]
            return X

        X_train = _sub_matrix(train_ids)
        X_test = _sub_matrix(test_ids)
        print(
            f"  {feature_name}: {X_train.shape[1]} dims "
            f"(train {X_train.shape[0]} / test {X_test.shape[0]})"
        )
        return X_train, X_test

    if feature_name == "boltz_raw_plus_pretrain_concat":
        # Hybrid: concat raw all-pairs pooled trunk (1024d) with the
        # pretrained concat-pool embedding (1024d) -> 2048d for TabPFN.
        # Rationale: C-concat beat raw by MAE -0.0009, so each side has
        # marginal unique signal; join them rather than swap.
        raw_path = REPO_ROOT.joinpath(
            "data", "boltz_affhead", "pooled_allpairs.parquet"
        )
        pre_path = REPO_ROOT.joinpath(
            "data", "boltz_trunk_pretrain_embed_c_concat.parquet"
        )
        if not raw_path.exists():
            raise SystemExit(f"Missing {raw_path}")
        if not pre_path.exists():
            raise SystemExit(f"Missing {pre_path}")
        raw_df = pd.read_parquet(raw_path).set_index("compound_id")
        pre_df = pd.read_parquet(pre_path).set_index("compound_id")

        def _concat_matrix(ids):
            Xraw = raw_df.reindex(index=ids).to_numpy(dtype=np.float32).copy()
            Xpre = pre_df.reindex(index=ids).to_numpy(dtype=np.float32).copy()
            # Column-mean fill for NaN rows (same convention as pooled_boltz)
            for X in (Xraw, Xpre):
                col_mean = np.nanmean(X, axis=0)
                col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
                X[~np.isfinite(X)] = np.broadcast_to(col_mean, X.shape)[~np.isfinite(X)]
            return np.concatenate([Xraw, Xpre], axis=1)

        X_train = _concat_matrix(train_ids)
        X_test = _concat_matrix(test_ids)
        print(
            f"  boltz_raw_plus_pretrain_concat: {X_train.shape[1]} dims "
            f"(train {X_train.shape[0]} / test {X_test.shape[0]})"
        )
        return X_train, X_test

    if feature_name == "boltz2_mordred3d":
        # Boltz-2 pose-derived Mordred 3D descriptors (213 features per
        # compound). Source: compound_boltz2_mordred3d.descriptors JSONB.
        # Unlike compound_mordred (2D), these use the actual Boltz-2
        # predicted conformation. Family: 3D tabular (pose-conditioned
        # scalars). Complementary to tier-0 (post-trunk heads).
        sql = "SELECT compound_id, descriptors FROM compound_boltz2_mordred3d"
        with psycopg2.connect(**DB_PARAMS) as conn:
            rows = pd.read_sql(sql, conn)
        rows["compound_id"] = rows["compound_id"].astype(int)
        all_keys = sorted({k for d in rows["descriptors"] for k in d.keys()})
        mat = np.zeros((len(rows), len(all_keys)), dtype=np.float32)
        for i, d in enumerate(rows["descriptors"]):
            for j, k in enumerate(all_keys):
                v = d.get(k)
                if v is None:
                    mat[i, j] = np.nan
                else:
                    try:
                        mat[i, j] = float(v)
                    except (TypeError, ValueError):
                        mat[i, j] = np.nan
        m3d = pd.DataFrame(mat, index=rows["compound_id"].values, columns=all_keys)

        def _m3d_matrix(ids):
            X = m3d.reindex(index=ids).to_numpy(dtype=np.float32).copy()
            col_mean = np.nanmean(X, axis=0)
            if np.isnan(col_mean).any():
                col_mean_global = np.nanmean(m3d.to_numpy(dtype=np.float32), axis=0)
                col_mean = np.where(np.isnan(col_mean), col_mean_global, col_mean)
            col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
            inds = np.where(np.isnan(X))
            X[inds] = np.take(col_mean, inds[1])
            return X

        X_train = _m3d_matrix(train_ids)
        X_test = _m3d_matrix(test_ids)
        print(
            f"  boltz2_mordred3d: {X_train.shape[1]} features "
            f"(train {X_train.shape[0]} / test {X_test.shape[0]})"
        )
        return X_train, X_test

    if feature_name == "boltz2_tabular_tier0":
        # Boltz-2 tier-0 scalar bundle (17 features):
        #   6 affinity heads: mean ensemble + 2 members, each (pred_value,
        #     probability_binary)
        #   9 confidence: confidence_score, ptm, iptm, ligand_iptm,
        #     protein_iptm, complex_plddt, complex_iplddt, complex_pde,
        #     complex_ipde
        #   2 geometry: ligand_atom_count, ligand_to_pocket_distance_a
        # Source: compound_boltz2 (populated by boltz2_postprocess.py).
        # Motivation (#116 Codex): currently no pose-derived tabular
        # feature in the pool — the 2 pooled_boltz members use trunk
        # embeddings only. tier-0 scalars are from a different level of
        # the Boltz-2 stack (post-trunk inference heads) and add a new
        # "axis" the pool doesn't cover.
        sql = """
        SELECT compound_id,
               affinity_pred_value, affinity_probability_binary,
               affinity_pred_value_1, affinity_probability_binary_1,
               affinity_pred_value_2, affinity_probability_binary_2,
               confidence_score, ptm, iptm, ligand_iptm, protein_iptm,
               complex_plddt, complex_iplddt, complex_pde, complex_ipde,
               ligand_atom_count, ligand_to_pocket_distance_a
          FROM compound_boltz2
        """
        with psycopg2.connect(**DB_PARAMS) as conn:
            tier0_df = pd.read_sql(sql, conn).set_index("compound_id")
        tier0_cols = tier0_df.columns.tolist()

        def _tier0_matrix(ids):
            X = tier0_df.reindex(index=ids).to_numpy(dtype=np.float32).copy()
            col_mean = np.nanmean(X, axis=0)
            if np.isnan(col_mean).any():
                col_mean_global = tier0_df.to_numpy(dtype=np.float32)
                col_mean_global = np.nanmean(col_mean_global, axis=0)
                col_mean = np.where(np.isnan(col_mean), col_mean_global, col_mean)
            col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
            inds = np.where(np.isnan(X))
            X[inds] = np.take(col_mean, inds[1])
            return X

        X_train = _tier0_matrix(train_ids)
        X_test = _tier0_matrix(test_ids)
        print(
            f"  boltz2_tabular_tier0: {X_train.shape[1]} features "
            f"(cols: {tier0_cols[:3]}...)"
        )
        return X_train, X_test

    if feature_name == "pooled_boltz":
        # Pooled Boltz-2 trunk embeddings (1024 features):
        #   s_prot_mean (384) -- global mean of s over 434 PXR residue tokens
        #   s_lig_mean  (384) -- global mean of s over ligand atom tokens
        #   z_if_mean   (128) -- mean of z over (core_pocket x ligand_atoms)
        #   z_if_max    (128) -- max  of z over (core_pocket x ligand_atoms)
        #
        # Source: data/boltz_affhead/pooled.parquet, produced by
        # track1_activity/scripts/boltz_affhead/01_pool_embeddings.py.
        # Auranofin (cid 1657) has no embedding; NaN rows are filled with
        # column means so LightGBM has a well-defined row.
        pooled_path = REPO_ROOT.joinpath("data", "boltz_affhead", "pooled.parquet")
        if not pooled_path.exists():
            raise SystemExit(
                f"Missing {pooled_path}. Run "
                f"track1_activity/scripts/boltz_affhead/01_pool_embeddings.py"
            )
        pool_df = pd.read_parquet(pooled_path).set_index("compound_id")

        def _pooled_matrix(ids):
            X = pool_df.reindex(index=ids).to_numpy(dtype=np.float32).copy()
            col_mean = np.nanmean(X, axis=0)
            # Safety: if any column is all-NaN on the requested subset,
            # fall back to global mean from the full parquet.
            if np.isnan(col_mean).any():
                col_mean_global = pool_df.to_numpy(dtype=np.float32).mean(axis=0)
                col_mean = np.where(np.isnan(col_mean), col_mean_global, col_mean)
            inds = np.where(np.isnan(X))
            X[inds] = np.take(col_mean, inds[1])
            return X

        X_train = _pooled_matrix(train_ids)
        X_test = _pooled_matrix(test_ids)
        print(
            f"  pooled_boltz: s_prot 384 + s_lig 384 + z_if_mean 128 + "
            f"z_if_max 128 = {X_train.shape[1]} features"
        )
        return X_train, X_test

    if feature_name == "3d_ligand":
        # 3D ligand-only bundle (1212 features):
        #   scalar3d       11  (compound_boltz2_desc3d)
        #   autocorr3d     80  (compound_boltz2_desc3d_vector.autocorr3d)
        #   getaway       273  (compound_boltz2_desc3d_vector.getaway)
        #   morse         224  (compound_boltz2_desc3d_vector.morse)
        #   rdf           210  (compound_boltz2_desc3d_vector.rdf)
        #   whim          114  (compound_boltz2_desc3d_vector.whim)
        #   usr            12  (compound_boltz2_desc3d_vector.usr)
        #   usrcat         60  (compound_boltz2_desc3d_vector.usrcat)
        #   electroshape   15  (compound_boltz2_skfp3d.electroshape)
        #   mordred3d     213  (compound_boltz2_mordred3d)
        #
        # Excluded:
        #   pose_jazzy (6) -- in 2d_full_boltz; skipped here for orthogonality
        #   e3fp (1024)   -- 4% gain / dim is very inefficient per today's EDA
        #   pharmacophore3d (2048) -- 8% gain / dim likewise inefficient
        #
        # Compound 1657 (Auranofin) is missing from all three tables
        # (Boltz-2 preprocessing refused the Au complex). NaN cells filled
        # with zeros -- same treatment as 2d_full_boltz.
        scalar_cols = [
            "asphericity",
            "eccentricity",
            "inertial_shape_factor",
            "npr1",
            "npr2",
            "pmi1",
            "pmi2",
            "pmi3",
            "radius_of_gyration",
            "spherocity_index",
            "pbf",
        ]
        with psycopg2.connect(**DB_PARAMS) as conn:
            scalar_df = pd.read_sql(
                f"SELECT compound_id, {', '.join(scalar_cols)} "
                f"FROM compound_boltz2_desc3d",
                conn,
            ).set_index("compound_id")
            vec_df = pd.read_sql(
                "SELECT compound_id, autocorr3d, getaway, morse, rdf, "
                "whim, usr, usrcat FROM compound_boltz2_desc3d_vector",
                conn,
            ).set_index("compound_id")
            skfp_df = pd.read_sql(
                "SELECT compound_id, electroshape FROM compound_boltz2_skfp3d",
                conn,
            ).set_index("compound_id")
            mord_df = pd.read_sql(
                "SELECT compound_id, descriptors FROM compound_boltz2_mordred3d",
                conn,
            ).set_index("compound_id")

        def _fill_vec(series, dim):
            out = []
            for v in series:
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    out.append(np.zeros(dim, dtype=np.float32))
                else:
                    a = np.asarray(v, dtype=np.float64)
                    a = np.where(np.isnan(a) | np.isinf(a), 0.0, a)
                    out.append(a.astype(np.float32))
            return np.stack(out, axis=0)

        def _mord_matrix(series, train_idx):
            # Build stable column ordering from first non-null row
            mord_cols = None
            for v in series:
                if v is not None:
                    mord_cols = sorted(v.keys())
                    break
            if mord_cols is None:
                raise RuntimeError("compound_boltz2_mordred3d is empty")
            rows = []
            for v in series:
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    rows.append(np.zeros(len(mord_cols), dtype=np.float32))
                else:
                    vals = [v.get(c) for c in mord_cols]
                    arr = np.asarray(
                        [float(x) if x is not None else 0.0 for x in vals],
                        dtype=np.float64,
                    )
                    arr = np.where(np.isnan(arr) | np.isinf(arr), 0.0, arr)
                    rows.append(arr.astype(np.float32))
            return np.stack(rows, axis=0), mord_cols

        def _build(ids):
            s = scalar_df.reindex(ids)[scalar_cols].to_numpy(dtype=np.float32)
            s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
            v = vec_df.reindex(ids)
            autocorr = _fill_vec(v["autocorr3d"], 80)
            getaway = _fill_vec(v["getaway"], 273)
            morse = _fill_vec(v["morse"], 224)
            rdf = _fill_vec(v["rdf"], 210)
            whim = _fill_vec(v["whim"], 114)
            usr = _fill_vec(v["usr"], 12)
            usrcat = _fill_vec(v["usrcat"], 60)
            es = _fill_vec(skfp_df.reindex(ids)["electroshape"], 15)
            mord, _ = _mord_matrix(mord_df.reindex(ids)["descriptors"], ids)
            return np.concatenate(
                [s, autocorr, getaway, morse, rdf, whim, usr, usrcat, es, mord],
                axis=1,
            )

        X_train = _build(train_ids)
        X_test = _build(test_ids)
        print(
            f"  3d_ligand: scalar 11 + autocorr 80 + getaway 273 + morse 224 "
            f"+ rdf 210 + whim 114 + usr 12 + usrcat 60 + electroshape 15 "
            f"+ mordred3d 213 = {X_train.shape[1]} features"
        )
        return X_train, X_test

    if feature_name == "2d_full_boltz":
        # Mordred (1531) + pose-Jazzy (6, from compound_boltz2_jazzy) +
        # RDKit_desc_full (217) + Boltz-2 Tier-0 (17 cols + 2 derived = 19) +
        # Tier-1 (44 confidence aggregates) = 1817 features.
        #
        # Uses pose-specific Jazzy instead of Jazzy's self-relaxed conformer
        # (the default 2d_full does the latter). Rationale: all pose-derived
        # Boltz features are already in the bundle, so keeping Jazzy
        # consistent with the binding pose is more principled. Pose-Jazzy
        # correlates 0.98-0.999 with self-Jazzy per prior A/B, so signal
        # loss is minimal but interpretability gains.
        #
        # Excluded: Tier-2 (IFP 114, 0.05x Mordred gain/feature) and
        # PoseBusters (19 bool, near-constant across PXR drug-like train,
        # variance ~= 0). See memory:project_posebusters_low_variance and
        # PR #72.

        # Mordred 1531
        mordred_train, _ = load_train_mordred()
        mordred_test = load_mordred(test_ids)
        common_m = mordred_train.columns.intersection(mordred_test.columns)
        Xm_tr = mordred_train.loc[train_ids, common_m].values.astype(np.float32)
        Xm_te = mordred_test.loc[test_ids, common_m].values.astype(np.float32)
        Xm_tr = np.nan_to_num(Xm_tr, nan=0.0, posinf=0.0, neginf=0.0)
        Xm_te = np.nan_to_num(Xm_te, nan=0.0, posinf=0.0, neginf=0.0)

        # Pose-Jazzy 6 (from compound_boltz2_jazzy)
        with psycopg2.connect(**DB_PARAMS) as conn:
            pose_jazzy_df = pd.read_sql(
                """
                SELECT compound_id, sdc, sdx, sa, dga, dgp, dgtot
                FROM compound_boltz2_jazzy
                """,
                conn,
            ).set_index("compound_id")
        jz_cols = list(JAZZY_FEATURE_COLS)
        Xj_tr = pose_jazzy_df.reindex(train_ids)[jz_cols].to_numpy(dtype=np.float32)
        Xj_te = pose_jazzy_df.reindex(test_ids)[jz_cols].to_numpy(dtype=np.float32)
        Xj_tr = np.nan_to_num(Xj_tr, nan=0.0, posinf=0.0, neginf=0.0)
        Xj_te = np.nan_to_num(Xj_te, nan=0.0, posinf=0.0, neginf=0.0)

        # RDKit 217
        rdkit_train = load_rdkit_full(train_ids)
        rdkit_test = load_rdkit_full(test_ids)
        common_r = rdkit_train.columns.intersection(rdkit_test.columns)
        Xr_tr = rdkit_train.loc[train_ids, common_r].to_numpy(dtype=np.float32)
        Xr_te = rdkit_test.loc[test_ids, common_r].to_numpy(dtype=np.float32)
        Xr_tr = np.nan_to_num(Xr_tr, nan=0.0, posinf=0.0, neginf=0.0)
        Xr_te = np.nan_to_num(Xr_te, nan=0.0, posinf=0.0, neginf=0.0)

        # Tier-0: scalar + derived from compound_boltz2
        boltz2_cols = [
            "affinity_pred_value",
            "affinity_pred_value_1",
            "affinity_pred_value_2",
            "affinity_probability_binary",
            "affinity_probability_binary_1",
            "affinity_probability_binary_2",
            "confidence_score",
            "ptm",
            "iptm",
            "ligand_iptm",
            "protein_iptm",
            "complex_plddt",
            "complex_iplddt",
            "complex_pde",
            "complex_ipde",
            "ligand_atom_count",
            "ligand_to_pocket_distance_a",
        ]
        col_sql = ", ".join(f"b.{c}" for c in boltz2_cols)
        with psycopg2.connect(**DB_PARAMS) as conn:
            boltz_df = pd.read_sql(
                f"""
                SELECT c.id AS compound_id, {col_sql},
                       (b.affinity_pred_value_1 - b.affinity_pred_value_2)
                           AS ensemble_diff_affinity,
                       (b.affinity_probability_binary_1
                          - b.affinity_probability_binary_2) AS ensemble_diff_prob
                FROM compounds c
                LEFT JOIN compound_boltz2 b ON b.compound_id = c.id
                """,
                conn,
            ).set_index("compound_id")
        tier0_cols = boltz2_cols + ["ensemble_diff_affinity", "ensemble_diff_prob"]
        Xt0_tr = boltz_df.reindex(train_ids)[tier0_cols].to_numpy(dtype=np.float32)
        Xt0_te = boltz_df.reindex(test_ids)[tier0_cols].to_numpy(dtype=np.float32)
        Xt0_tr = np.nan_to_num(Xt0_tr, nan=0.0, posinf=0.0, neginf=0.0)
        Xt0_te = np.nan_to_num(Xt0_te, nan=0.0, posinf=0.0, neginf=0.0)

        # Tier-1: confidence aggregates from parquet
        tier1_path = REPO_ROOT.joinpath("data", "boltz2_confidence_features.parquet")
        tier1_df = pd.read_parquet(tier1_path)
        Xt1_tr = tier1_df.reindex(train_ids).to_numpy(dtype=np.float32)
        Xt1_te = tier1_df.reindex(test_ids).to_numpy(dtype=np.float32)
        Xt1_tr = np.nan_to_num(Xt1_tr, nan=0.0, posinf=0.0, neginf=0.0)
        Xt1_te = np.nan_to_num(Xt1_te, nan=0.0, posinf=0.0, neginf=0.0)

        X_train = np.concatenate([Xm_tr, Xj_tr, Xr_tr, Xt0_tr, Xt1_tr], axis=1)
        X_test = np.concatenate([Xm_te, Xj_te, Xr_te, Xt0_te, Xt1_te], axis=1)
        print(
            f"  2d_full_boltz: mordred {Xm_tr.shape[1]} + "
            f"pose-jazzy {Xj_tr.shape[1]} + rdkit {Xr_tr.shape[1]} + "
            f"tier0 {Xt0_tr.shape[1]} + tier1 {Xt1_tr.shape[1]} "
            f"= {X_train.shape[1]} features"
        )
        return X_train, X_test

    if feature_name == "2d_full":
        # Mordred (1531) + Jazzy (6) + RDKit_desc_full (217) = 1754 features.
        # Sized to fit comfortably inside TabPFN's ~2000-feature soft limit
        # while exposing every 2D descriptor family we have computed.
        # NaNs in Mordred/RDKit are zeroed; LightGBM's native NaN handling
        # is not relevant here since TabPFN does not accept NaN.
        mordred_train, _ = load_train_mordred()
        mordred_test = load_mordred(test_ids)
        common_m = mordred_train.columns.intersection(mordred_test.columns)
        Xm_tr = mordred_train.loc[train_ids, common_m].values.astype(np.float32)
        Xm_te = mordred_test.loc[test_ids, common_m].values.astype(np.float32)
        Xm_tr = np.nan_to_num(Xm_tr, nan=0.0, posinf=0.0, neginf=0.0)
        Xm_te = np.nan_to_num(Xm_te, nan=0.0, posinf=0.0, neginf=0.0)

        jazzy_train = load_jazzy(train_ids).reindex(index=train_ids)
        jazzy_test = load_jazzy(test_ids).reindex(index=test_ids)
        Xj_tr = jazzy_train[list(JAZZY_FEATURE_COLS)].to_numpy(dtype=np.float32)
        Xj_te = jazzy_test[list(JAZZY_FEATURE_COLS)].to_numpy(dtype=np.float32)

        rdkit_train = load_rdkit_full(train_ids)
        rdkit_test = load_rdkit_full(test_ids)
        common_r = rdkit_train.columns.intersection(rdkit_test.columns)
        Xr_tr = rdkit_train.loc[train_ids, common_r].to_numpy(dtype=np.float32)
        Xr_te = rdkit_test.loc[test_ids, common_r].to_numpy(dtype=np.float32)
        Xr_tr = np.nan_to_num(Xr_tr, nan=0.0, posinf=0.0, neginf=0.0)
        Xr_te = np.nan_to_num(Xr_te, nan=0.0, posinf=0.0, neginf=0.0)

        X_train = np.concatenate([Xm_tr, Xj_tr, Xr_tr], axis=1)
        X_test = np.concatenate([Xm_te, Xj_te, Xr_te], axis=1)
        print(
            f"  2d_full: mordred {Xm_tr.shape[1]} + jazzy {Xj_tr.shape[1]} "
            f"+ rdkit {Xr_tr.shape[1]} = {X_train.shape[1]} features"
        )
        return X_train, X_test

    if feature_name in EMBEDDING_TABLES:
        table = EMBEDDING_TABLES[feature_name]
        return load_embeddings(table, train_ids), load_embeddings(table, test_ids)

    if feature_name in FP_REGISTRY:
        train_mols = smiles_to_mols(train_df["smiles"])
        test_mols = smiles_to_mols(test_df["smiles"])
        return (
            FP_REGISTRY[feature_name](train_mols).astype(np.float32),
            FP_REGISTRY[feature_name](test_mols).astype(np.float32),
        )

    raise ValueError(
        f"Unknown feature: {feature_name}. "
        f"Available: rdkit_desc_full, mordred, {list(EMBEDDING_TABLES)}, {list(FP_REGISTRY)}"
    )


# ---------------------------------------------------------------------------
# Optuna search spaces
# ---------------------------------------------------------------------------


def optuna_search_space(model_type: str, trial):
    """Generate hyperparameter candidates for Optuna trial."""
    if model_type == "lgbm":
        return {
            "objective": "regression",
            "metric": "mae",
            "boosting_type": "gbdt",
            "verbose": -1,
            "seed": 42,
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        }
    if model_type == "xgboost":
        return {
            "objective": "reg:absoluteerror",
            "eval_metric": "mae",
            "tree_method": "hist",
            "seed": 42,
            "verbosity": 0,
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 50),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
        }
    if model_type == "catboost":
        return {
            "iterations": 2000,
            "loss_function": "MAE",
            "eval_metric": "MAE",
            "verbose": 0,
            "random_seed": 42,
            "allow_writing_files": False,
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "random_strength": trial.suggest_float(
                "random_strength", 1e-3, 10.0, log=True
            ),
            "border_count": trial.suggest_int("border_count", 32, 255),
        }
    if model_type == "tabpfn":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 2, 32),
            "device": "cuda",
            "softmax_temperature": trial.suggest_float(
                "softmax_temperature", 0.1, 2.0, log=True
            ),
            "random_state": 42,
        }
        if "model_path" in DEFAULT_PARAMS["tabpfn"]:
            params["model_path"] = DEFAULT_PARAMS["tabpfn"]["model_path"]
        return params
    raise ValueError(f"Unknown model: {model_type}")


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------


def train_predict(model_type: str, params: dict, X_tr, y_tr, X_va, y_va, w_tr=None):
    """Train a model and return (val_preds, best_iteration)."""
    if model_type == "lgbm":
        import lightgbm as lgb

        dt = lgb.Dataset(X_tr, label=y_tr, weight=w_tr)
        dv = lgb.Dataset(X_va, label=y_va, reference=dt)
        model = lgb.train(
            params,
            dt,
            num_boost_round=2000,
            valid_sets=[dv],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )
        return model.predict(X_va), model.best_iteration, model

    if w_tr is not None and model_type != "lgbm":
        raise NotImplementedError(
            f"Sample weights only supported for lgbm, got {model_type}"
        )

    if model_type == "xgboost":
        import xgboost as xgb

        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dval = xgb.DMatrix(X_va, label=y_va)
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=2000,
            evals=[(dval, "val")],
            early_stopping_rounds=50,
            verbose_eval=False,
        )
        return model.predict(dval), model.best_iteration, model

    if model_type == "catboost":
        import catboost as cb

        model = cb.CatBoostRegressor(**params)
        model.fit(
            cb.Pool(X_tr, label=y_tr),
            eval_set=cb.Pool(X_va, label=y_va),
            early_stopping_rounds=50,
        )
        return model.predict(X_va), model.best_iteration_, model

    if model_type == "tabpfn":
        from tabpfn import TabPFNRegressor

        model = TabPFNRegressor(**params)
        model.fit(X_tr, y_tr)
        return model.predict(X_va), params["n_estimators"], model

    raise ValueError(f"Unknown model: {model_type}")


def train_final(
    model_type: str,
    params: dict,
    X_train,
    y_train,
    num_rounds: int,
    sample_weight=None,
):
    """Train final model on all training data.

    ``sample_weight`` is supported for LightGBM only; XGBoost/CatBoost raise
    NotImplementedError when a non-None weight is supplied (matching
    ``train_predict``).
    """
    if model_type == "lgbm":
        import lightgbm as lgb

        return lgb.train(
            params,
            lgb.Dataset(X_train, label=y_train, weight=sample_weight),
            num_boost_round=num_rounds,
        )

    if sample_weight is not None:
        raise NotImplementedError(
            f"Sample weights only supported for lgbm, got {model_type}"
        )

    if model_type == "xgboost":
        import xgboost as xgb

        return xgb.train(
            params, xgb.DMatrix(X_train, label=y_train), num_boost_round=num_rounds
        )

    if model_type == "catboost":
        import catboost as cb

        p = {**params, "iterations": num_rounds}
        model = cb.CatBoostRegressor(**p)
        model.fit(cb.Pool(X_train, label=y_train))
        return model

    if model_type == "tabpfn":
        from tabpfn import TabPFNRegressor

        model = TabPFNRegressor(**params)
        model.fit(X_train, y_train)
        return model

    raise ValueError(f"Unknown model: {model_type}")


def predict_final(model_type: str, model, X_test):
    """Predict with final model."""
    if model_type == "xgboost":
        import xgboost as xgb

        return model.predict(xgb.DMatrix(X_test))
    return model.predict(X_test)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def run(args):
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if args.model == "tabpfn" and args.tabpfn_version != "v2_6":
        from tabpfn import TabPFNRegressor
        from tabpfn.constants import ModelVersion

        version_enum = {"v2_5": ModelVersion.V2_5, "v2": ModelVersion.V2}[
            args.tabpfn_version
        ]
        ref = TabPFNRegressor.create_default_for_version(version_enum)
        DEFAULT_PARAMS["tabpfn"]["model_path"] = ref.model_path

    print(
        f"Model: {args.model}, Feature: {args.feature}, Split: {args.split}, Trials: {args.trials}"
    )

    # Load data
    print("Loading data...")
    train_df = load_train_smiles_target()
    test_df = load_test_smiles()
    y_train = train_df["pec50"].values

    X_train, X_test = load_features(args.feature, train_df, test_df)
    print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")

    if args.model == "tabpfn" and X_train.shape[1] > 500:
        print(
            f"  NOTE: TabPFN with {X_train.shape[1]} features — GPU strongly "
            f"recommended. Consider rdkit_desc_full (217d) or chemeleon (300d)."
        )

    # Optional pseudo-labels
    pseudo_X = None
    pseudo_y = None
    pseudo_w = None
    pseudo_count = 0
    if args.pseudo:
        if args.trials > 0:
            raise ValueError(
                "--pseudo requires --trials 0 (Optuna+pseudo not supported)"
            )
        print(f"Loading pseudo-labels from {args.pseudo}...")
        pseudo_df = load_pseudo_labels(Path(args.pseudo))
        if args.pseudo_min_confidence > 0:
            n_before = len(pseudo_df)
            pseudo_df = pseudo_df[
                pseudo_df["confidence"] >= args.pseudo_min_confidence
            ].reset_index(drop=True)
            print(
                f"  confidence filter ≥{args.pseudo_min_confidence}: "
                f"{n_before} → {len(pseudo_df)} pseudo compounds"
            )
        train_ids = set(load_compound_ids("train"))
        pseudo_ids = set(pseudo_df["compound_id"].astype(int))
        overlap = train_ids & pseudo_ids
        assert not overlap, (
            f"Pseudo compound_ids overlap with train ({len(overlap)} ids); "
            f"fold-safety violated"
        )
        if args.feature != "mordred":
            raise NotImplementedError(
                f"--pseudo currently supports --feature mordred only, "
                f"got {args.feature!r}"
            )
        # Canonical column order: same intersection load_features built above.
        mordred_train_df, _ = load_train_mordred()
        mordred_test_df = load_mordred(load_compound_ids("test"))
        feature_columns = list(
            mordred_train_df.columns.intersection(mordred_test_df.columns)
        )
        assert X_train.shape[1] == len(feature_columns), (
            f"Train feature column count mismatch: X_train has {X_train.shape[1]} "
            f"cols, expected {len(feature_columns)}"
        )
        pseudo_X = build_pseudo_feature_matrix(
            args.feature, pseudo_df, feature_columns=feature_columns
        )
        assert pseudo_X.shape[1] == X_train.shape[1], (
            f"Pseudo/train column mismatch: pseudo={pseudo_X.shape[1]}, "
            f"train={X_train.shape[1]}"
        )
        pseudo_y = pseudo_df["pseudo_pec50"].values.astype(np.float32)
        pseudo_w = (
            pseudo_df["confidence"].values.astype(np.float32) * args.pseudo_weight
        )
        pseudo_count = len(pseudo_df)
        print(
            f"  pseudo_X: {pseudo_X.shape}, weight={args.pseudo_weight}, "
            f"effective sum={pseudo_w.sum():.1f}"
        )

    # CV splits
    smiles_list = train_df["smiles"].tolist()
    if args.split == "scaffold":
        outer_splits = scaffold_split_indices(smiles_list, n_splits=5, seed=42)
    elif args.split == "umap":
        train_ids_ordered = load_compound_ids("train")
        split_features, split_metric = _build_umap_split_features(
            args.embedding_space, train_ids_ordered
        )
        outer_splits = umap_split_indices(
            smiles_list,
            n_splits=5,
            n_clusters=args.umap_clusters,
            seed=args.umap_seed,
            features=split_features,
            metric=split_metric,
        )
    elif args.split == "analog":
        counter_df = load_train_smiles_with_counter()
        assert len(counter_df) == len(train_df), (
            f"Row count mismatch: train_df={len(train_df)}, "
            f"counter_df={len(counter_df)}; both must ORDER BY t.id"
        )
        selectivity = (counter_df["pec50"] - counter_df["counter_pec50"]).to_numpy()
        outer_splits = analog_aware_split_indices(
            smiles_list=smiles_list,
            pec50=y_train,
            selectivity=selectivity,
            n_splits=5,
            analog_tanimoto_threshold=args.analog_threshold,
            seed=42,
            verbose=True,
        )
    elif args.split == "mixed":
        counter_df = load_train_smiles_with_counter()
        assert len(counter_df) == len(train_df), (
            f"Row count mismatch: train_df={len(train_df)}, "
            f"counter_df={len(counter_df)}; both must ORDER BY t.id"
        )
        selectivity = (counter_df["pec50"] - counter_df["counter_pec50"]).to_numpy()
        outer_splits = mixed_analog_diversity_split_indices(
            smiles_list=smiles_list,
            pec50=y_train,
            selectivity=selectivity,
            n_splits=5,
            analog_tanimoto_threshold=args.analog_threshold,
            seed=args.mixed_seed,
            verbose=True,
        )
    elif args.split == "test_nn":
        test_smiles = load_test_smiles()["smiles"].tolist()
        outer_splits = test_nn_split_indices(
            smiles_list=smiles_list,
            test_smiles=test_smiles,
            n_splits=5,
            test_nn_threshold=args.test_nn_threshold,
            seed=args.test_nn_seed,
            verbose=True,
        )
    elif args.split == "adv":
        test_smiles = load_test_smiles()["smiles"].tolist()
        outer_splits, _ = adversarial_split_indices(
            smiles_list=smiles_list,
            test_smiles=test_smiles,
            n_splits=5,
            n_top=args.adv_n_top,
            seed=args.adv_seed,
            verbose=True,
        )
    else:
        raise ValueError(f"Unknown split: {args.split}")

    # Experiment name
    exp_name = f"{args.model}_{args.feature}_{args.split}"
    if args.split == "analog" and args.analog_threshold != 0.25:
        exp_name += f"{args.analog_threshold}"
    if args.split == "umap" and args.embedding_space != "morgan":
        exp_name += f"_{args.embedding_space}"
    if args.split == "umap" and args.umap_seed != 42:
        exp_name += f"_s{args.umap_seed}"
    if args.split == "umap" and args.umap_clusters != 50:
        exp_name += f"_k{args.umap_clusters}"
    if args.split == "mixed" and args.mixed_seed != 42:
        exp_name += f"_s{args.mixed_seed}"
    if args.split == "test_nn" and args.test_nn_seed != 42:
        exp_name += f"_s{args.test_nn_seed}"
    if args.split == "test_nn" and args.test_nn_threshold != 0.25:
        exp_name += f"_t{args.test_nn_threshold}"
    if args.split == "adv" and args.adv_seed != 42:
        exp_name += f"_s{args.adv_seed}"
    if args.split == "adv" and args.adv_n_top != 849:
        exp_name += f"_n{args.adv_n_top}"
    if args.trials == 0:
        exp_name += "_default"
    if args.gap_lambda > 0:
        exp_name += f"_gap{args.gap_lambda}"
    if args.pseudo:
        exp_name += f"_pseudo{args.pseudo_weight}"
        if args.pseudo_min_confidence > 0:
            exp_name += f"_minc{args.pseudo_min_confidence}"
    if args.model == "tabpfn" and args.tabpfn_version != "v2_6":
        exp_name += f"_{args.tabpfn_version}"
    print(f"  Experiment: {exp_name}")

    # Training loop
    oof_preds = np.zeros(len(y_train))
    # Tracks which train indices were covered by at least one val fold.
    # Standard UMAP/scaffold K-fold covers every index exactly once, but
    # partial-coverage splits (e.g. analog-aware) only predict a subset.
    # Overall OOF metrics must ignore uncovered positions.
    oof_covered = np.zeros(len(y_train), dtype=bool)
    fold_metrics = []
    fold_best_params = []
    num_boost_rounds = []

    for fold, (tr_idx, va_idx) in enumerate(outer_splits):
        X_tr, X_va = X_train[tr_idx], X_train[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]
        w_tr = None

        if pseudo_X is not None:
            real_n = len(X_tr)
            X_tr, y_tr, w_tr = augment_fold(
                X_tr, y_tr, pseudo_X, pseudo_y, pseudo_w, base_weight=1.0
            )
            print(
                f"  [Fold {fold}] augmented: real={real_n} + pseudo={pseudo_count} "
                f"= {len(X_tr)} train, val={len(X_va)} (real only)"
            )

        if args.trials > 0:
            # Optuna tuning directly on outer fold val (no inner CV)
            gap_lambda = args.gap_lambda
            print(
                f"  [Fold {fold}] Tuning ({args.trials} trials, gap_λ={gap_lambda})..."
            )

            def objective(trial):
                params = optuna_search_space(args.model, trial)
                val_preds, _, model = train_predict(
                    args.model, params, X_tr, y_tr, X_va, y_va
                )
                val_mae = np.mean(np.abs(y_va - val_preds))
                if gap_lambda > 0:
                    train_preds = predict_final(args.model, model, X_tr)
                    train_mae = np.mean(np.abs(y_tr - train_preds))
                    gap = abs(train_mae - val_mae)
                    return val_mae + gap_lambda * gap
                return val_mae

            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=args.trials)
            best_params = optuna_search_space(args.model, study.best_trial)
        else:
            print(f"  [Fold {fold}] Using default params...")
            best_params = DEFAULT_PARAMS[args.model].copy()

        fold_best_params.append(best_params)

        # Train with best params
        vp, best_iter, _ = train_predict(
            args.model, best_params, X_tr, y_tr, X_va, y_va, w_tr=w_tr
        )
        oof_preds[va_idx] = vp
        oof_covered[va_idx] = True
        metrics = compute_metrics(y_va, vp)
        fold_metrics.append(metrics)
        num_boost_rounds.append(best_iter)
        print_metrics(metrics, label=f"Fold {fold}")

    # OOF summary (covered subset only; identical to all-rows for
    # standard K-fold, but required for partial-coverage splits)
    covered_count = int(oof_covered.sum())
    if covered_count < len(y_train):
        print(
            f"\n  Partial-coverage split: {covered_count}/{len(y_train)} "
            f"({100 * covered_count / len(y_train):.1f}%) train rows appeared in val."
        )
    oof_metrics = compute_metrics(y_train[oof_covered], oof_preds[oof_covered])
    print("\n  Overall OOF (covered subset):")
    print_metrics(oof_metrics)
    print_fold_summary(fold_metrics)

    # Final model: use median best_iteration, best fold's params
    avg_rounds = int(np.median(num_boost_rounds))
    best_fold = int(np.argmin([m["RAE"] for m in fold_metrics]))
    final_params = fold_best_params[best_fold]
    if args.model == "tabpfn":
        print(
            f"\n  Final model: n_estimators={final_params['n_estimators']}, "
            f"params from fold {best_fold} "
            f"(RAE={fold_metrics[best_fold]['RAE']:.4f})"
        )
    else:
        print(
            f"\n  Final model: {avg_rounds} rounds, params from fold {best_fold} "
            f"(RAE={fold_metrics[best_fold]['RAE']:.4f})"
        )

    if pseudo_X is not None:
        X_final = np.concatenate([X_train, pseudo_X], axis=0)
        y_final = np.concatenate([y_train, pseudo_y], axis=0)
        w_final = np.concatenate(
            [np.ones(len(X_train), dtype=np.float32), pseudo_w.astype(np.float32)],
            axis=0,
        )
        final_model = train_final(
            args.model,
            final_params,
            X_final,
            y_final,
            avg_rounds,
            sample_weight=w_final,
        )
    else:
        final_model = train_final(
            args.model, final_params, X_train, y_train, avg_rounds
        )
    test_preds = predict_final(args.model, final_model, X_test)
    print(f"  Test preds: mean={test_preds.mean():.3f}, std={test_preds.std():.3f}")

    # Save submission
    sub = pd.DataFrame(
        {
            "SMILES": test_df["smiles"],
            "Molecule Name": test_df["molecule_name"],
            "pEC50": test_preds,
        }
    )
    sub_path = SUBMISSION_DIR.joinpath(f"{exp_name}.csv")
    sub.to_csv(sub_path, index=False)

    # Record to DB
    exp_id = record_experiment(
        name=exp_name,
        description=f"{args.model} + {args.feature} ({args.split} split)",
        model_type=args.model,
        feature_set=args.feature,
        hyperparameters=final_params,
        fold_metrics=fold_metrics,
        submission_path=f"track1_activity/submissions/{exp_name}.csv",
        num_boost_rounds=num_boost_rounds,
        notes=(
            f"OOF RAE={oof_metrics['RAE']:.4f}, {args.split}_split, "
            f"{args.trials}trials, gap_lambda={args.gap_lambda}"
            + (
                f", pseudo={args.pseudo} weight={args.pseudo_weight} n={pseudo_count}"
                if args.pseudo
                else ""
            )
        ),
    )
    save_oof_predictions(exp_id, oof_preds, covered_mask=oof_covered)

    print(f"\n  Done: {exp_name} -> RAE={oof_metrics['RAE']:.4f}")
    return oof_metrics


def main():
    all_features = (
        [
            "rdkit_desc_full",
            "mordred",
            "mordred_singleconc",
            "mordred_jazzy",
            "2d_full",
            "2d_full_boltz",
            "pooled_boltz",
            "pooled_boltz_allpairs",
            "chemprop_pretrain_embed",
            "molformer_c3_pretrain_embed",
            "chemberta_5m_mtr_pretrain_embed",
            "kermt_pretrain_embed",
            "attentivefp_pretrain_embed",
            "gatedgcn_pretrain_embed",
            "unimol_v2_pretrain_embed",
            "boltz_trunk_pretrain_embed_a",
            "boltz_trunk_pretrain_embed_b",
            "boltz_trunk_pretrain_embed_b_first",
            "boltz_trunk_pretrain_embed_c",
            "boltz_trunk_pretrain_embed_c_h512",
            "boltz_trunk_pretrain_embed_c_h1024",
            "boltz_trunk_pretrain_embed_c_concat",
            "boltz_trunk_pretrain_embed_c_concat_t8p25",
            "boltz_trunk_pretrain_embed_c_cls",
            "boltz_raw_plus_pretrain_concat",
            "pooled_boltz_ab_sonly",
            "pooled_boltz_ab_zonly",
            "pooled_boltz_ab_sprot",
            "pooled_boltz_ab_slig",
            "pooled_boltz_ab_zmean",
            "pooled_boltz_ab_zmax",
            "pooled_boltz_ab_nozmax",
            "2d_full_boltz_log2fc_pred",
            "2d_full_boltz_log2fc_pred_ens4",
            "cheme_2d_full_boltz_log2fc_pred",
            "cheme_2d_full_boltz_log2fc_pred_ens4",
            "cheme_2d_full_boltz_log2fc_pred_seed5ens",
            "cheme_2d_full_boltz_log2fc_pred_seed10ens",
            "cheme_2d_full_boltz_log2fc_pred_seed15ens",
            "cheme_2d_full_boltz_log2fc_pred_seed20ens",
            "2d_full_boltz_log2fc_pred_seed5ens",
            "2d_full_boltz_log2fc_pred_seed10ens",
            "2d_full_boltz_log2fc_pred_seed15ens",
            "2d_full_boltz_log2fc_pred_seed20ens",
            "cheme_2d_full_boltz_log2fc_emax_pred",
            "cconcat_2d_full_boltz_log2fc_pred",
            "cheme_cconcat_2d_full_boltz_log2fc_pred",
            "3d_ligand",
            "jazzy",
            "boltz2_tabular_tier0",
            "boltz2_mordred3d",
        ]
        + list(FP_REGISTRY.keys())
        + list(EMBEDDING_TABLES.keys())
    )

    parser = argparse.ArgumentParser(description="Unified model training")
    parser.add_argument(
        "--model", choices=["lgbm", "xgboost", "catboost", "tabpfn"], required=True
    )
    parser.add_argument("--feature", choices=all_features, required=True)
    parser.add_argument(
        "--split",
        choices=["scaffold", "umap", "analog", "mixed", "test_nn", "adv"],
        default="scaffold",
    )
    parser.add_argument(
        "--analog-threshold",
        type=float,
        default=0.25,
        help="Tanimoto NN threshold for analog-aware split (default 0.25)",
    )
    parser.add_argument(
        "--mixed-seed",
        type=int,
        default=42,
        help="Seed for mixed analog+diversity split bucket assignment "
        "(default 42). Use to estimate split-variance across seeds.",
    )
    parser.add_argument(
        "--test-nn-threshold",
        type=float,
        default=0.25,
        help="Tanimoto threshold for test_nn split (default 0.25)",
    )
    parser.add_argument(
        "--test-nn-seed",
        type=int,
        default=42,
        help="Seed for test_nn bucket assignment (default 42)",
    )
    parser.add_argument(
        "--adv-n-top",
        type=int,
        default=849,
        help="Top-N train compounds by classifier P(test) go to val pool "
        "(default 849, matches mixed_t025 analog stratum size).",
    )
    parser.add_argument(
        "--adv-seed",
        type=int,
        default=42,
        help="Seed for adversarial split classifier folding + bucketing",
    )
    parser.add_argument(
        "--umap-seed",
        type=int,
        default=42,
        help="Seed for UMAP+KMeans clustering (default 42). Use to estimate "
        "split-variance across seeds.",
    )
    parser.add_argument(
        "--umap-clusters",
        type=int,
        default=50,
        help="Number of KMeans clusters before 5-fold group assignment (default 50).",
    )
    parser.add_argument(
        "--embedding-space",
        default="morgan",
        choices=["morgan", "mordred", "morgan_mordred"] + list(EMBEDDING_TABLES.keys()),
        help="Feature space used by UMAP split for clustering. Default 'morgan' "
        "matches the canonical split. Other choices re-cluster in an alternative "
        "representation (e.g. chemberta_5m_mtr, molformer_xl, mordred, "
        "morgan_mordred=zscored Mordred concat Morgan FP) to test "
        "whether split-shape is similarity-space dependent.",
    )
    parser.add_argument(
        "--trials", type=int, default=20, help="Optuna trials (0=default params)"
    )
    parser.add_argument(
        "--gap-lambda",
        type=float,
        default=0.0,
        help="Gap regularization lambda (0=off, 0.5-2.0 recommended)",
    )
    parser.add_argument(
        "--pseudo",
        type=str,
        default=None,
        help="Path to pseudo-label parquet (e.g. data/pseudo_labels.parquet)",
    )
    parser.add_argument(
        "--pseudo-weight",
        type=float,
        default=0.3,
        help="Weight multiplier for pseudo samples (multiplied by per-row confidence)",
    )
    parser.add_argument(
        "--pseudo-min-confidence",
        type=float,
        default=0.0,
        help="Drop pseudo samples with confidence below this threshold (0 = keep all)",
    )
    parser.add_argument(
        "--tabpfn-version",
        choices=["v2_6", "v2_5", "v2"],
        default="v2_6",
        help="TabPFN pretrained checkpoint (v2_6=synthetic latest, "
        "v2_5=real-tuned, v2=legacy). Only used when --model tabpfn.",
    )
    args = parser.parse_args()

    run(args)


if __name__ == "__main__":
    main()

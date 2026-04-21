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
    "chemberta_5m_mtr": "compound_chemberta_5m_mtr",
    "chemberta_zinc_v1": "compound_chemberta_zinc_v1",
    "bert_base_smiles": "compound_bert_smiles",
    "molformer_xl": "compound_molformer",
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

    if feature_name == "2d_full_boltz_log2fc_pred":
        # 2d_full_boltz (1817d) + 2 predicted log2_fc scalars from the
        # chemprop pretrain model (LF head forward, un-z-scored).
        # Buterez 2024 strategy-2: predicted LF labels as side feature.
        # See track1_activity/scripts/run_chemprop_predict_log2fc.py.
        X_train_base, X_test_base = load_features("2d_full_boltz", train_df, test_df)
        lf_path = REPO_ROOT.joinpath(
            "data", "chemprop_pretrain_log2fc_predictions.parquet"
        )
        if not lf_path.exists():
            raise SystemExit(
                f"Missing {lf_path}. Run "
                f"track1_activity/scripts/run_chemprop_predict_log2fc.py"
            )
        lf_df = pd.read_parquet(lf_path)
        cols = ["log2fc_8p25_pred", "log2fc_33_pred"]
        Xl_tr = lf_df.reindex(index=train_ids)[cols].to_numpy(dtype=np.float32).copy()
        Xl_te = lf_df.reindex(index=test_ids)[cols].to_numpy(dtype=np.float32).copy()
        Xl_tr = np.nan_to_num(Xl_tr, nan=0.0, posinf=0.0, neginf=0.0)
        Xl_te = np.nan_to_num(Xl_te, nan=0.0, posinf=0.0, neginf=0.0)
        X_train = np.concatenate([X_train_base, Xl_tr], axis=1)
        X_test = np.concatenate([X_test_base, Xl_te], axis=1)
        print(
            f"  2d_full_boltz_log2fc_pred: {X_train_base.shape[1]} base + "
            f"{Xl_tr.shape[1]} LF preds = {X_train.shape[1]} features"
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
        # 128d graph-pooled embedding from PyG ResGatedGraphConv stack
        # pretrained on single_concentration log2_fc (2-head, z-scored
        # targets). Extracted by replacing GatedGCNModel.ffn with
        # nn.Identity() so forward returns the global_mean_pool output.
        # See: track1_activity/scripts/run_gatedgcn_embed_extract.py
        # Buterez 2024 strategy-3 with gated edge-conditioned message
        # passing backbone (fifth pretrain-embed family member).
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
            "kermt_pretrain_embed",
            "attentivefp_pretrain_embed",
            "gatedgcn_pretrain_embed",
            "2d_full_boltz_log2fc_pred",
            "3d_ligand",
            "jazzy",
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

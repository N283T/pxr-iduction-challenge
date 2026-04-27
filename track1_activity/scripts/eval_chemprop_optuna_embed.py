#!/usr/bin/env python3
"""Quick gate 1 + gate 2 evaluation for chemprop Optuna pretrain embeddings.

Reuses the gate framework from eval_kano_embed.py. Tests whether a 384d
embedding from a Trial 10/11 (Optuna-tuned log2fc pretrain) ckpt is
worth a caruana ADD test on top of the current 9-pool.

Gate 1: TabPFN UMAP 5-fold CV on optuna_trial_embed alone -> OOF MAE <= 0.45
        (chemprop_pretrain_embed reference: 0.4373)
Gate 2: residual r vs current pool members <= 0.85 (per
        feedback_new_family_threshold_min_r_085)

Usage:
    pixi run python track1_activity/scripts/eval_chemprop_optuna_embed.py \\
        --parquet data/chemprop_pretrain_optuna_trial10_embed.parquet \\
        --tag trial10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))

from data import (  # noqa: E402
    DB_PARAMS,
    load_test_smiles,
    load_train_smiles_target,
)
from splits import umap_split_indices  # noqa: E402


def load_compound_ids(split: str) -> list[int]:
    table = "train_activity" if split == "train" else "test_activity"
    with psycopg2.connect(**DB_PARAMS) as conn, conn.cursor() as cur:
        cur.execute(f"SELECT compound_id FROM {table} ORDER BY id")
        return [r[0] for r in cur.fetchall()]


def load_embed_features(parquet_path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_parquet(parquet_path)
    train_ids = load_compound_ids("train")
    test_ids = load_compound_ids("test")
    X_train = df.reindex(index=train_ids).to_numpy(dtype=np.float32)
    X_test = df.reindex(index=test_ids).to_numpy(dtype=np.float32)
    if np.isnan(X_train).any() or np.isnan(X_test).any():
        n_nan_tr = int(np.isnan(X_train).any(axis=1).sum())
        n_nan_te = int(np.isnan(X_test).any(axis=1).sum())
        raise ValueError(f"NaN in features: train={n_nan_tr} test={n_nan_te}")
    n_dead = int(np.sum((X_train == 0).all(axis=0)))
    n_low_var = int(np.sum(X_train.std(axis=0) < 1e-4))
    print(
        f"  features: train {X_train.shape}, test {X_test.shape}, "
        f"all-zero cols {n_dead}/{X_train.shape[1]}, "
        f"low-var cols {n_low_var}/{X_train.shape[1]}"
    )
    return X_train, X_test


def evaluate_umap_oof(
    X_train: np.ndarray, y_train: np.ndarray
) -> tuple[np.ndarray, dict]:
    from scipy.stats import spearmanr
    from tabpfn import TabPFNRegressor

    smiles_train = load_train_smiles_target()["smiles"].tolist()
    folds = umap_split_indices(smiles_train, n_splits=5, n_clusters=50, seed=42)

    oof_pred = np.zeros(len(y_train), dtype=np.float32)
    tabpfn_params = dict(
        n_estimators=8,
        device="cuda" if torch.cuda.is_available() else "cpu",
        softmax_temperature=0.9,
        random_state=42,
        ignore_pretraining_limits=True,
    )
    for fi, (tr_idx, va_idx) in enumerate(folds):
        model = TabPFNRegressor(**tabpfn_params)
        model.fit(X_train[tr_idx], y_train[tr_idx])
        oof_pred[va_idx] = model.predict(X_train[va_idx])
        fold_mae = float(np.mean(np.abs(oof_pred[va_idx] - y_train[va_idx])))
        print(f"    fold {fi}: |va|={len(va_idx)} MAE={fold_mae:.4f}")

    mae = float(np.mean(np.abs(oof_pred - y_train)))
    sp = float(spearmanr(oof_pred, y_train).correlation)
    return oof_pred, {"MAE": mae, "Spearman": sp}


def gate2_check(oof_pred: np.ndarray, y_train: np.ndarray) -> dict:
    """Compute residual r vs each pool member's OOF predictions."""
    pool_members = [
        "tabpfn_cheme_2d_full_boltz_log2fc_pred_optuna_trial10_seed5ens_umap_default",
        "tabpfn_cheme_2d_full_boltz_log2fc_pred_seed10ens_top500_umap",
        "tabpfn_chemprop_pretrain_embed_umap_default",
        "tabpfn_kermt_pretrain_embed_umap_default",
        "tabpfn_pooled_boltz_umap_default",
        "tabpfn_pooled_boltz_allpairs_umap_default",
        "tabpfn_molformer_c3_pretrain_embed_umap",
        "tabpfn_attentivefp_pretrain_embed_umap_default",
        "tabpfn_gatedgcn_pretrain_embed_umap_default",
    ]
    print("\n  Pool residual r:")
    res_new = oof_pred - y_train
    rs = {}
    with psycopg2.connect(**DB_PARAMS) as conn, conn.cursor() as cur:
        for name in pool_members:
            cur.execute(
                """SELECT op.train_idx, op.oof_prediction
                FROM experiment_oof_predictions op
                JOIN experiments e ON e.id = op.experiment_id
                WHERE e.name = %s ORDER BY op.train_idx""",
                (name,),
            )
            rows = cur.fetchall()
            if not rows:
                print(f"    {name}: NO OOF in DB")
                continue
            preds = np.full(len(y_train), np.nan, dtype=np.float32)
            for idx, val in rows:
                preds[idx] = val
            if np.isnan(preds).any():
                print(
                    f"    {name}: incomplete OOF ({int(np.isnan(preds).sum())} missing)"
                )
                continue
            res_member = preds - y_train
            r = float(np.corrcoef(res_new, res_member)[0, 1])
            rs[name] = r
            tag = " <-- HIGH (gate2 risk)" if r > 0.85 else ""
            short = (
                name.replace("tabpfn_", "")
                .replace("_umap_default", "")
                .replace("_umap", "")
            )
            print(f"    r={r:.3f}  {short}{tag}")
    return rs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parquet", type=Path, required=True, help="embed parquet path"
    )
    parser.add_argument(
        "--tag", type=str, required=True, help="short tag (e.g. trial10)"
    )
    args = parser.parse_args()

    print(f"=== chemprop optuna {args.tag} embed gate 1 + gate 2 ===\n")

    print("Loading data + features")
    train_df = load_train_smiles_target()
    test_df = load_test_smiles()
    y_train = train_df["pec50"].to_numpy(dtype=np.float32)
    print(f"  train n={len(train_df)}, test n={len(test_df)}")

    X_train, _ = load_embed_features(args.parquet)

    print("\n=== Gate 1: TabPFN UMAP 5-fold CV ===")
    oof_pred, metrics = evaluate_umap_oof(X_train, y_train)
    print(f"\n  OOF MAE={metrics['MAE']:.4f}  Spearman={metrics['Spearman']:.4f}")
    print("  Reference: chemprop_pretrain_embed OOF MAE 0.4373")
    print("  Reference: kermt OOF MAE 0.4485")
    print("  Reference: attentivefp OOF MAE 0.4844")
    print("  Gate 1 PASS if MAE <= 0.45")
    if metrics["MAE"] > 0.45:
        print(f"  GATE 1 FAIL (MAE {metrics['MAE']:.4f} > 0.45). Stop.")
        return
    print(f"  GATE 1 PASS (MAE {metrics['MAE']:.4f} <= 0.45)")

    print("\n=== Gate 2: residual r vs pool ===")
    rs = gate2_check(oof_pred, y_train)
    if rs:
        max_r = max(rs.values())
        max_name = max(rs, key=rs.get)
        short = (
            max_name.replace("tabpfn_", "")
            .replace("_umap_default", "")
            .replace("_umap", "")
        )
        print(f"\n  max r = {max_r:.3f} ({short})")
        print("  Gate 2 PASS if max r <= 0.85")
        if max_r > 0.85:
            print(
                f"  GATE 2 FAIL (max r {max_r:.3f} > 0.85). Caruana would not weight."
            )
        else:
            print("  GATE 2 PASS. Worth caruana ADD test.")


if __name__ == "__main__":
    main()

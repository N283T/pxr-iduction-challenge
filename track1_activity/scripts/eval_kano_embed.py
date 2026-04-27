#!/usr/bin/env python3
"""Quick gate 1 + gate 2 evaluation of KANO embedding.

Gate 1: TabPFN UMAP 5-fold CV on kano_embed alone → OOF MAE ≤ 0.45
Gate 2: residual r vs current pool members ≤ 0.85 (per
`feedback_new_family_threshold_min_r_085`)

If both pass, run_kano_embed_extract.py output is worth a caruana ADD test.

Usage:
    pixi run python track1_activity/scripts/eval_kano_embed.py
"""

from __future__ import annotations

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

KANO_PARQUET = REPO_ROOT.joinpath("data", "kano_embed.parquet")


def load_compound_ids(split: str) -> list[int]:
    table = "train_activity" if split == "train" else "test_activity"
    with psycopg2.connect(**DB_PARAMS) as conn, conn.cursor() as cur:
        cur.execute(f"SELECT compound_id FROM {table} ORDER BY id")
        return [r[0] for r in cur.fetchall()]


def load_kano_features() -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_parquet(KANO_PARQUET)
    train_ids = load_compound_ids("train")
    test_ids = load_compound_ids("test")
    X_train = df.reindex(index=train_ids).to_numpy(dtype=np.float32)
    X_test = df.reindex(index=test_ids).to_numpy(dtype=np.float32)
    if np.isnan(X_train).any() or np.isnan(X_test).any():
        n_nan_tr = int(np.isnan(X_train).any(axis=1).sum())
        n_nan_te = int(np.isnan(X_test).any(axis=1).sum())
        raise ValueError(f"NaN in KANO features: train={n_nan_tr} test={n_nan_te}")
    print(
        f"  KANO features: train {X_train.shape}, test {X_test.shape}, "
        f"all-zero cols {int(np.sum((X_train == 0).all(axis=0)))}/300"
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
    res_kano = oof_pred - y_train
    rs = {}
    with psycopg2.connect(**DB_PARAMS) as conn, conn.cursor() as cur:
        for name in pool_members:
            cur.execute(
                """SELECT op.compound_id, op.predicted
                FROM experiment_oof_predictions op
                JOIN experiments e ON e.id = op.experiment_id
                WHERE e.name = %s ORDER BY op.compound_id""",
                (name,),
            )
            rows = cur.fetchall()
            if not rows:
                print(f"    {name}: NO OOF in DB")
                continue
            df = pd.DataFrame(rows, columns=["compound_id", "predicted"]).set_index(
                "compound_id"
            )
            train_ids = load_compound_ids("train")
            preds = df.reindex(index=train_ids)["predicted"].to_numpy(dtype=np.float32)
            res_member = preds - y_train
            r = float(np.corrcoef(res_kano, res_member)[0, 1])
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
    print("=== KANO embed gate 1 + gate 2 ===\n")

    print("Loading data + KANO features")
    train_df = load_train_smiles_target()
    test_df = load_test_smiles()
    y_train = train_df["pec50"].to_numpy(dtype=np.float32)
    print(f"  train n={len(train_df)}, test n={len(test_df)}")

    X_train, X_test = load_kano_features()

    print("\n=== Gate 1: TabPFN UMAP 5-fold CV ===")
    oof_pred, metrics = evaluate_umap_oof(X_train, y_train)
    print(f"\n  OOF MAE={metrics['MAE']:.4f}  Spearman={metrics['Spearman']:.4f}")
    print("  Reference: chemprop_pretrain_embed OOF MAE 0.4373")
    print("  Reference: kermt OOF MAE 0.4485")
    print("  Reference: attentivefp OOF MAE 0.4844")
    print("  Gate 1 PASS if MAE ≤ 0.45")
    if metrics["MAE"] > 0.45:
        print(f"  GATE 1 FAIL (MAE {metrics['MAE']:.4f} > 0.45). Stop.")
        return
    print(f"  GATE 1 PASS (MAE {metrics['MAE']:.4f} ≤ 0.45)")

    print("\n=== Gate 2: residual r vs pool ===")
    rs = gate2_check(oof_pred, y_train)
    if rs:
        max_r = max(rs.values())
        max_name = max(rs, key=rs.get)
        print(
            f"\n  max r = {max_r:.3f} ({max_name.replace('tabpfn_', '').replace('_umap_default', '').replace('_umap', '')})"
        )
        print("  Gate 2 PASS if max r ≤ 0.85")
        if max_r > 0.85:
            print(
                f"  GATE 2 FAIL (max r {max_r:.3f} > 0.85). Caruana would not weight."
            )
        else:
            print("  GATE 2 PASS. Worth caruana ADD test.")


if __name__ == "__main__":
    main()

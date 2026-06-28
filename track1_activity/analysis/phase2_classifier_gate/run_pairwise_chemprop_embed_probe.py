#!/usr/bin/env -S pixi run python
"""Probe PXR readouts on ChEMBL-pairwise ChemProp embeddings.

This keeps the ChEMBL pairwise-pretrained ChemProp encoder fixed, extracts
256d molecule embeddings, then tests small PXR-side readouts on train+AS1 folds:

* pEC50 regression
* 5-bin activity classification
* high/low tail binary classifiers

It is experiment-only and writes outputs under this analysis directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from chemprop import data as chemprop_data
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "scripts")))
sys.path.insert(
    0, str(REPO_ROOT.joinpath("track1_activity", "analysis", "phase2_classifier_gate"))
)

from data import get_engine  # noqa: E402
from run_multiclass_gate_probe import BIN_LABELS, label_arrays, load_pool_with_folds  # noqa: E402
from score_chemprop_pairwise_pretrain import load_pairwise_model  # noqa: E402

OUT_ROOT = Path(__file__).resolve().parent / "outputs" / "pairwise_chemprop_embed"
DEFAULT_CKPT = REPO_ROOT.joinpath(
    "track1_activity",
    "checkpoints",
    "chemprop_pairwise_chembl_binding_random250k_100kp5",
    "pairwise_pretrain.pt",
)


def load_test_frame() -> pd.DataFrame:
    return pd.read_sql(
        """
        SELECT
            t.id AS test_id,
            c.id AS compound_id,
            c.molecule_name,
            c.std_smiles AS smiles,
            l.pec50 AS as1_pec50
        FROM test_activity t
        JOIN compounds c ON c.id = t.compound_id
        LEFT JOIN test_activity_phase1_labels l ON l.compound_id = t.compound_id
        ORDER BY t.id
        """,
        get_engine(),
    )


def extract_embeddings(model, smiles: list[str], batch_size: int) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    if len(smiles) % batch_size == 1 and batch_size > 2:
        batch_size -= 1
    pts = [
        chemprop_data.MoleculeDatapoint.from_smi(
            smi, np.asarray([0.0], dtype=np.float32)
        )
        for smi in smiles
    ]
    loader = chemprop_data.build_dataloader(
        chemprop_data.MoleculeDataset(pts),
        batch_size=batch_size,
        shuffle=False,
    )
    chunks = []
    with torch.no_grad():
        for batch in loader:
            bmg = batch.bmg
            bmg.to(device)
            emb = model.base.fingerprint(bmg)
            chunks.append(emb.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0)


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(stats.spearmanr(x, y).statistic)


def regression_metrics(pool: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    rows = []
    y = pool["pec50"].to_numpy(dtype=float)
    masks = {
        "all": np.ones(len(pool), dtype=bool),
        "source_train": pool["source"].eq("train").to_numpy(),
        "source_as1": pool["source"].eq("as1").to_numpy(),
        "true_lt3": y < 3.0,
        "true_gte6": y >= 6.0,
    }
    for name, mask in masks.items():
        if int(mask.sum()) == 0:
            continue
        yy = y[mask]
        pp = pred[mask]
        rows.append(
            {
                "slice": name,
                "n": int(mask.sum()),
                "mae": float(mean_absolute_error(yy, pp)),
                "bias_pred_minus_true": float(np.mean(pp - yy)),
                "spearman": safe_spearman(pp, yy),
                "pred_mean": float(np.mean(pp)),
                "true_mean": float(np.mean(yy)),
            }
        )
    return pd.DataFrame(rows)


def tail_auc_rows(
    pool: pd.DataFrame, score: np.ndarray, score_name: str
) -> list[dict[str, object]]:
    y = pool["pec50"].to_numpy(dtype=float)
    out = []
    for label, target, oriented in [
        ("lt3", y < 3.0, -score),
        ("gte6", y >= 6.0, score),
    ]:
        target_i = target.astype(int)
        row: dict[str, object] = {
            "score": score_name,
            "class": label,
            "n_pos": int(target_i.sum()),
        }
        if target_i.min() == target_i.max():
            row["roc_auc"] = np.nan
            row["average_precision"] = np.nan
        else:
            row["roc_auc"] = float(roc_auc_score(target_i, oriented))
            row["average_precision"] = float(
                average_precision_score(target_i, oriented)
            )
        out.append(row)
    return out


def make_lgbm_regressor(seed: int):
    return lgb.LGBMRegressor(
        objective="regression_l1",
        n_estimators=700,
        learning_rate=0.025,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        random_state=seed,
        verbose=-1,
    )


def make_lgbm_classifier(binary: bool, seed: int):
    return lgb.LGBMClassifier(
        objective="binary" if binary else "multiclass",
        class_weight="balanced",
        n_estimators=700,
        learning_rate=0.025,
        num_leaves=31,
        min_child_samples=15,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        random_state=seed,
        verbose=-1,
    )


def make_tabpfn_regressor(args: argparse.Namespace, seed: int):
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion

    version_enum = {"v3": ModelVersion.V3, "v2_6": ModelVersion.V2_6}[
        args.tabpfn_version
    ]
    model_path = TabPFNRegressor.create_default_for_version(version_enum).model_path
    return TabPFNRegressor(
        device=args.device,
        n_estimators=args.n_estimators,
        random_state=seed,
        model_path=model_path,
        ignore_pretraining_limits=True,
        show_progress_bar=False,
    )


def make_tabpfn_classifier(args: argparse.Namespace, seed: int):
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion

    version_enum = {"v3": ModelVersion.V3, "v2_6": ModelVersion.V2_6}[
        args.tabpfn_version
    ]
    model_path = TabPFNClassifier.create_default_for_version(version_enum).model_path
    return TabPFNClassifier(
        device=args.device,
        n_estimators=args.n_estimators,
        random_state=seed,
        model_path=model_path,
        ignore_pretraining_limits=True,
        show_progress_bar=False,
    )


def aligned_proba(clf, x: np.ndarray, n_classes: int) -> np.ndarray:
    raw = clf.predict_proba(x)
    classes = getattr(clf, "classes_", np.arange(raw.shape[1]))
    out = np.zeros((x.shape[0], n_classes), dtype=np.float64)
    for src_idx, cls in enumerate(classes):
        out[:, int(cls)] = raw[:, src_idx]
    row_sum = out.sum(axis=1, keepdims=True)
    return np.divide(
        out,
        row_sum,
        out=np.full_like(out, 1.0 / n_classes),
        where=row_sum > 0,
    )


def run_oof(
    args: argparse.Namespace, pool: pd.DataFrame, x: np.ndarray
) -> dict[str, object]:
    y_reg = pool["pec50"].to_numpy(dtype=float)
    y_cls, label_to_id, _id_to_label = label_arrays(pool)
    folds = sorted(pool["fold"].unique())
    if args.fold_limit:
        folds = folds[: args.fold_limit]

    pred_reg = np.full(len(pool), np.nan, dtype=np.float64)
    proba_multi = np.zeros((len(pool), len(BIN_LABELS)), dtype=np.float64)
    proba_high = np.full(len(pool), np.nan, dtype=np.float64)
    proba_low = np.full(len(pool), np.nan, dtype=np.float64)
    fold_rows = []

    for fold in folds:
        val_idx = pool.index[pool["fold"].eq(fold)].to_numpy(dtype=np.int64)
        tr_idx = pool.index[~pool["fold"].eq(fold)].to_numpy(dtype=np.int64)
        scaler = StandardScaler()
        x_tr = scaler.fit_transform(x[tr_idx])
        x_va = scaler.transform(x[val_idx])
        seed = args.seed + int(fold)

        if args.readout.startswith("tabpfn"):
            reg = make_tabpfn_regressor(args, seed)
            multi = make_tabpfn_classifier(args, seed)
            high = make_tabpfn_classifier(args, seed)
            low = make_tabpfn_classifier(args, seed)
        else:
            reg = make_lgbm_regressor(seed)
            multi = make_lgbm_classifier(binary=False, seed=seed)
            high = make_lgbm_classifier(binary=True, seed=seed)
            low = make_lgbm_classifier(binary=True, seed=seed)

        reg.fit(x_tr, y_reg[tr_idx])
        pred_reg[val_idx] = reg.predict(x_va)

        multi.fit(x_tr, y_cls[tr_idx])
        proba_multi[val_idx] = aligned_proba(multi, x_va, len(BIN_LABELS))

        y_high = (y_cls == label_to_id["gte6"]).astype(int)
        y_low = (y_cls == label_to_id["lt3"]).astype(int)
        high.fit(x_tr, y_high[tr_idx])
        low.fit(x_tr, y_low[tr_idx])
        proba_high[val_idx] = aligned_proba(high, x_va, 2)[:, 1]
        proba_low[val_idx] = aligned_proba(low, x_va, 2)[:, 1]

        fold_rows.append(
            {
                "fold": int(fold),
                "n_train": int(len(tr_idx)),
                "n_val": int(len(val_idx)),
                "reg_mae": float(
                    mean_absolute_error(y_reg[val_idx], pred_reg[val_idx])
                ),
                "multi_acc": float(
                    accuracy_score(y_cls[val_idx], proba_multi[val_idx].argmax(axis=1))
                ),
            }
        )
        print(
            f"fold={int(fold)} reg_mae={fold_rows[-1]['reg_mae']:.4f} "
            f"multi_acc={fold_rows[-1]['multi_acc']:.4f}"
        )

    pred_cls = proba_multi.argmax(axis=1)
    classifier_summary = pd.DataFrame(
        [
            {
                "readout": args.readout,
                "accuracy": float(accuracy_score(y_cls, pred_cls)),
                "balanced_accuracy": float(balanced_accuracy_score(y_cls, pred_cls)),
                "macro_f1": float(f1_score(y_cls, pred_cls, average="macro")),
                "weighted_f1": float(f1_score(y_cls, pred_cls, average="weighted")),
            }
        ]
    )
    auc_rows = []
    auc_rows.extend(tail_auc_rows(pool, pred_reg, "regression_pred"))
    auc_rows.extend(
        tail_auc_rows(pool, proba_multi[:, label_to_id["gte6"]], "multi_proba_gte6")
    )
    auc_rows.extend(
        tail_auc_rows(pool, -proba_multi[:, label_to_id["lt3"]], "neg_multi_proba_lt3")
    )
    auc_rows.append(
        {
            "score": "binary_proba_gte6",
            "class": "gte6",
            "n_pos": int((y_cls == label_to_id["gte6"]).sum()),
            "roc_auc": float(
                roc_auc_score((y_cls == label_to_id["gte6"]).astype(int), proba_high)
            ),
            "average_precision": float(
                average_precision_score(
                    (y_cls == label_to_id["gte6"]).astype(int), proba_high
                )
            ),
        }
    )
    auc_rows.append(
        {
            "score": "binary_proba_lt3",
            "class": "lt3",
            "n_pos": int((y_cls == label_to_id["lt3"]).sum()),
            "roc_auc": float(
                roc_auc_score((y_cls == label_to_id["lt3"]).astype(int), proba_low)
            ),
            "average_precision": float(
                average_precision_score(
                    (y_cls == label_to_id["lt3"]).astype(int), proba_low
                )
            ),
        }
    )

    return {
        "pred_reg": pred_reg,
        "proba_multi": proba_multi,
        "proba_high": proba_high,
        "proba_low": proba_low,
        "regression_summary": regression_metrics(pool, pred_reg),
        "classifier_summary": classifier_summary,
        "tail_auc": pd.DataFrame(auc_rows),
        "fold_summary": pd.DataFrame(fold_rows),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument(
        "--out-dir", type=Path, default=OUT_ROOT / "binding_seed42_lgbm"
    )
    parser.add_argument("--readout", choices=["lgbm", "tabpfn"], default="lgbm")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold-limit", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tabpfn-version", choices=["v3", "v2_6"], default="v3")
    parser.add_argument("--n-estimators", type=int, default=8)
    parser.add_argument(
        "--save-embeddings", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pool = load_pool_with_folds()
    test = load_test_frame()

    model = load_pairwise_model(args.ckpt)
    print(f"Extracting embeddings from {args.ckpt}")
    x_pool = extract_embeddings(
        model, pool["smiles"].astype(str).tolist(), args.batch_size
    )
    x_test = extract_embeddings(
        model, test["smiles"].astype(str).tolist(), args.batch_size
    )
    print(f"  pool embeddings: {x_pool.shape}")
    print(f"  test embeddings: {x_test.shape}")

    if args.save_embeddings:
        emb_cols = [f"pairchem_{i:03d}" for i in range(x_pool.shape[1])]
        pool_emb = pd.DataFrame(x_pool, columns=emb_cols)
        pool_emb.insert(0, "pool_idx", pool["pool_idx"].to_numpy())
        pool_emb.insert(1, "compound_id", pool["compound_id"].to_numpy())
        pool_emb.to_parquet(args.out_dir / "pool_embeddings.parquet", index=False)
        test_emb = pd.DataFrame(x_test, columns=emb_cols)
        test_emb.insert(0, "test_id", test["test_id"].to_numpy())
        test_emb.insert(1, "compound_id", test["compound_id"].to_numpy())
        test_emb.to_parquet(args.out_dir / "test_embeddings.parquet", index=False)

    result = run_oof(args, pool, x_pool)
    oof = pool[
        [
            "pool_idx",
            "compound_id",
            "molecule_name",
            "pec50",
            "source",
            "true_bin",
            "fold",
        ]
    ].copy()
    oof["pred_reg"] = result["pred_reg"]
    y_cls, label_to_id, _id_to_label = label_arrays(pool)
    for label, idx in label_to_id.items():
        oof[f"proba_{label}"] = result["proba_multi"][:, idx]
    oof["proba_high_binary"] = result["proba_high"]
    oof["proba_low_binary"] = result["proba_low"]
    oof.to_csv(args.out_dir / "oof_predictions.csv", index=False)

    for name in [
        "regression_summary",
        "classifier_summary",
        "tail_auc",
        "fold_summary",
    ]:
        result[name].to_csv(args.out_dir / f"{name}.csv", index=False)

    meta = {
        "ckpt": str(args.ckpt),
        "readout": args.readout,
        "pool_shape": list(x_pool.shape),
        "test_shape": list(x_test.shape),
        "args": {k: str(v) for k, v in vars(args).items()},
    }
    (args.out_dir / "meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print("\nRegression")
    print(result["regression_summary"].to_string(index=False))
    print("\nClassifier")
    print(result["classifier_summary"].to_string(index=False))
    print("\nTail AUC/AP")
    print(result["tail_auc"].to_string(index=False))
    print(f"\nSaved outputs to {args.out_dir}")


if __name__ == "__main__":
    main()

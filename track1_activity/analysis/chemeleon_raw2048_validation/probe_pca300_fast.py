#!/usr/bin/env -S pixi run python
"""Fast OOF and released-test probe for PCA-compressed CheMeleon raw-2048."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA


REPO_ROOT = Path(__file__).resolve().parents[3]
ANALYSIS_DIR = Path(__file__).resolve().parent
SCRIPT_DIR = REPO_ROOT.joinpath("track1_activity", "scripts")
SRC_DIR = REPO_ROOT.joinpath("track1_activity", "src")
sys.path.insert(0, str(ANALYSIS_DIR))
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SRC_DIR))

from data import DB_PARAMS, load_test_smiles, load_train_smiles_target  # noqa: E402
from evaluate import compute_metrics  # noqa: E402
from splits import umap_split_indices  # noqa: E402

import run_train  # noqa: E402
from probe_random_linear_fast import (  # noqa: E402
    COMPRESSED_DIM,
    N_ESTIMATORS,
    RAW_DIM,
    RAW_TABLE,
    impute_from_train,
    load_embedding_table,
    load_ids,
)


OUTPUT_DIR = REPO_ROOT.joinpath("data", "chemeleon_raw2048_validation")


def test_metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(np.mean(np.abs(y - prediction))),
        "spearman": float(spearmanr(y, prediction).statistic),
        "bias": float(np.mean(prediction - y)),
    }


def main() -> None:
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = OUTPUT_DIR.joinpath(f"pca300_fast_{stamp}.json")

    train_df = load_train_smiles_target()
    test_df = load_test_smiles()
    train_ids, test_ids = load_ids()
    all_ids = train_ids + test_ids
    raw_all = load_embedding_table(RAW_TABLE, all_ids, RAW_DIM)
    split_at = len(train_ids)
    raw_train, raw_test = raw_all[:split_at], raw_all[split_at:]
    base_train, base_test = run_train.load_features(
        "2d_full_boltz_log2fc_pred_seed10ens", train_df, test_df
    )
    y_train = train_df["pec50"].to_numpy(dtype=np.float32)

    with psycopg2.connect(**DB_PARAMS) as conn:
        released = pd.read_sql_query(
            """
            SELECT t.id, l.pec50
            FROM test_activity_phase1_labels l
            JOIN test_activity t ON t.compound_id = l.compound_id
            ORDER BY t.id
            """,
            conn,
        )
    if len(released) != len(test_df):
        raise ValueError(f"Expected {len(test_df)} labels, got {len(released)}")
    y_test = released["pec50"].to_numpy(dtype=np.float32)

    print("Fitting global train-only PCA(2048 -> 300)", flush=True)
    pca = PCA(
        n_components=COMPRESSED_DIM,
        svd_solver="randomized",
        random_state=42,
    )
    pca_train = pca.fit_transform(raw_train).astype(np.float32)
    pca_test = pca.transform(raw_test).astype(np.float32)
    explained_variance = float(pca.explained_variance_ratio_.sum())
    print(f"explained_variance_ratio={explained_variance:.6f}", flush=True)

    X_train = np.concatenate([pca_train, base_train], axis=1)
    X_test = np.concatenate([pca_test, base_test], axis=1)
    X_train, X_test = impute_from_train(X_train, X_test)
    splits = umap_split_indices(
        train_df["smiles"].tolist(), n_splits=5, n_clusters=50, seed=42
    )
    model_path = TabPFNRegressor.create_default_for_version(
        ModelVersion.V2_6
    ).model_path

    oof = np.zeros(len(y_train), dtype=np.float32)
    test_folds = []
    fold_metrics = []
    for fold, (fit_idx, val_idx) in enumerate(splits):
        model = TabPFNRegressor(
            device="cuda",
            n_estimators=N_ESTIMATORS,
            softmax_temperature=0.9,
            random_state=42,
            ignore_pretraining_limits=True,
            model_path=model_path,
        )
        model.fit(X_train[fit_idx], y_train[fit_idx])
        oof[val_idx] = model.predict(X_train[val_idx])
        test_folds.append(model.predict(X_test))
        fold_result = compute_metrics(y_train[val_idx], oof[val_idx])
        fold_metrics.append(fold_result)
        print(
            f"fold={fold} OOF_MAE={fold_result['MAE']:.6f} test=complete",
            flush=True,
        )
        del model
        torch.cuda.empty_cache()

    test_prediction = np.mean(np.stack(test_folds), axis=0)
    overall_oof = compute_metrics(y_train, oof)
    overall_test = test_metrics(y_test, test_prediction)
    result = {
        "label": "pca300_global_train",
        "input_dim": int(X_train.shape[1]),
        "n_estimators": N_ESTIMATORS,
        "explained_variance_ratio": explained_variance,
        "pca_scope": "all training features; no labels",
        "fold_metrics": fold_metrics,
        "oof": overall_oof,
        "released_test": overall_test,
    }
    json_path.write_text(
        json.dumps(
            result,
            indent=2,
            default=lambda value: (
                value.item() if isinstance(value, np.generic) else str(value)
            ),
        )
        + "\n"
    )
    print(
        f"OOF MAE={overall_oof['MAE']:.6f} Sp={overall_oof['Spearman_R']:.6f}",
        flush=True,
    )
    print(
        f"test MAE={overall_test['mae']:.6f} "
        f"Sp={overall_test['spearman']:.6f} bias={overall_test['bias']:+.6f}",
        flush=True,
    )
    print(f"Wrote {json_path}", flush=True)


if __name__ == "__main__":
    main()

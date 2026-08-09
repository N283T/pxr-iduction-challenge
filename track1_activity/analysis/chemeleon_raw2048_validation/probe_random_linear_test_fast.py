#!/usr/bin/env -S pixi run python
"""Released-test replay for legacy and random linear 300d projections."""

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


REPO_ROOT = Path(__file__).resolve().parents[3]
ANALYSIS_DIR = Path(__file__).resolve().parent
SCRIPT_DIR = REPO_ROOT.joinpath("track1_activity", "scripts")
SRC_DIR = REPO_ROOT.joinpath("track1_activity", "src")
sys.path.insert(0, str(ANALYSIS_DIR))
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SRC_DIR))

from data import DB_PARAMS, load_test_smiles, load_train_smiles_target  # noqa: E402
from splits import umap_split_indices  # noqa: E402

import run_train  # noqa: E402
from probe_random_linear_fast import (  # noqa: E402
    COMPRESSED_DIM,
    LEGACY_TABLE,
    N_ESTIMATORS,
    PROJECTION_SEEDS,
    RAW_DIM,
    RAW_TABLE,
    impute_from_train,
    load_embedding_table,
    load_ids,
    random_linear_projection,
)


OUTPUT_DIR = REPO_ROOT.joinpath("data", "chemeleon_raw2048_validation")


def metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(np.mean(np.abs(y - prediction))),
        "spearman": float(spearmanr(y, prediction).statistic),
        "bias": float(np.mean(prediction - y)),
    }


def save_checkpoint(path: Path, results: list[dict]) -> None:
    path.write_text(json.dumps({"results": results}, indent=2) + "\n")


def run_variant(
    label: str,
    chemeleon_train: np.ndarray,
    chemeleon_test: np.ndarray,
    base_train: np.ndarray,
    base_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    model_path: Path,
) -> dict:
    from tabpfn import TabPFNRegressor

    X_train = np.concatenate([chemeleon_train, base_train], axis=1)
    X_test = np.concatenate([chemeleon_test, base_test], axis=1)
    X_train, X_test = impute_from_train(X_train, X_test)
    test_folds = []
    print(f"\n{label}: input_dim={X_train.shape[1]}", flush=True)
    for fold, (fit_idx, _) in enumerate(splits):
        model = TabPFNRegressor(
            device="cuda",
            n_estimators=N_ESTIMATORS,
            softmax_temperature=0.9,
            random_state=42,
            ignore_pretraining_limits=True,
            model_path=model_path,
        )
        model.fit(X_train[fit_idx], y_train[fit_idx])
        test_folds.append(model.predict(X_test))
        print(f"  fold={fold} test prediction complete", flush=True)
        del model
        torch.cuda.empty_cache()

    prediction = np.mean(np.stack(test_folds), axis=0)
    result = {
        "label": label,
        "input_dim": int(X_train.shape[1]),
        "n_estimators": N_ESTIMATORS,
        "released_test": metrics(y_test, prediction),
    }
    print(
        f"  MAE={result['released_test']['mae']:.6f} "
        f"Sp={result['released_test']['spearman']:.6f} "
        f"bias={result['released_test']['bias']:+.6f}",
        flush=True,
    )
    return result


def main() -> None:
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = OUTPUT_DIR.joinpath(f"random_linear_test_fast_{stamp}.json")

    train_df = load_train_smiles_target()
    test_df = load_test_smiles()
    train_ids, test_ids = load_ids()
    all_ids = train_ids + test_ids
    raw_all = load_embedding_table(RAW_TABLE, all_ids, RAW_DIM)
    legacy_all = load_embedding_table(LEGACY_TABLE, all_ids, COMPRESSED_DIM)
    split_at = len(train_ids)
    raw_train, raw_test = raw_all[:split_at], raw_all[split_at:]
    legacy_train, legacy_test = legacy_all[:split_at], legacy_all[split_at:]
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

    splits = umap_split_indices(
        train_df["smiles"].tolist(), n_splits=5, n_clusters=50, seed=42
    )
    model_path = TabPFNRegressor.create_default_for_version(
        ModelVersion.V2_6
    ).model_path
    variants = {"legacy300": (legacy_train, legacy_test)}
    for seed in PROJECTION_SEEDS:
        variants[f"random_linear300_seed{seed}"] = (
            random_linear_projection(raw_train, seed),
            random_linear_projection(raw_test, seed),
        )

    results = []
    for label, (feature_train, feature_test) in variants.items():
        result = run_variant(
            label,
            feature_train,
            feature_test,
            base_train,
            base_test,
            y_train,
            y_test,
            splits,
            model_path,
        )
        results.append(result)
        save_checkpoint(json_path, results)
        print(f"  checkpoint={json_path}", flush=True)

    print("\nSummary", flush=True)
    for result in results:
        values = result["released_test"]
        print(
            f"{result['label']}: MAE={values['mae']:.6f} "
            f"Sp={values['spearman']:.6f} bias={values['bias']:+.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()

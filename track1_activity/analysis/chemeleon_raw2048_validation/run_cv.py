#!/usr/bin/env -S pixi run python
"""Paired TabPFN validation for corrected CheMeleon raw-2048 features."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import psycopg2


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = REPO_ROOT.joinpath("track1_activity", "scripts")
SRC_DIR = REPO_ROOT.joinpath("track1_activity", "src")
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SRC_DIR))

from data import DB_PARAMS, load_test_smiles, load_train_smiles_target  # noqa: E402
from evaluate import (  # noqa: E402
    compute_metrics,
    record_experiment,
    save_oof_predictions,
)
from splits import umap_split_indices  # noqa: E402

import run_train  # noqa: E402


TABLE = "compound_chemeleon_raw2048"
EXPECTED_DIM = 2048
OUTPUT_DIR = REPO_ROOT.joinpath("data", "chemeleon_raw2048_validation")
SUBMISSION_DIR = REPO_ROOT.joinpath("track1_activity", "submissions")

CASES = (
    "legacy300",
    "raw2048",
    "legacy300-mixed-full",
    "raw2048-mixed-full",
    "legacy300-mixed-top500",
    "raw2048-mixed-top500",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", choices=CASES, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=8)
    parser.add_argument("--softmax-temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=500)
    return parser.parse_args()


def load_embedding_table(table: str, ids: list[int], dim: int) -> np.ndarray:
    with psycopg2.connect(**DB_PARAMS) as conn, conn.cursor() as cur:
        cur.execute(f"SELECT compound_id, embedding FROM {table} ORDER BY compound_id")
        rows = cur.fetchall()
    embedding_map = {
        int(compound_id): np.asarray(embedding, dtype=np.float32)
        for compound_id, embedding in rows
    }
    missing = [compound_id for compound_id in ids if compound_id not in embedding_map]
    if missing:
        raise ValueError(
            f"{table} missing {len(missing)} requested IDs: {missing[:10]}"
        )
    values = np.stack([embedding_map[compound_id] for compound_id in ids])
    if values.shape != (len(ids), dim):
        raise ValueError(f"{table} shape {values.shape}; expected {(len(ids), dim)}")
    if not np.isfinite(values).all():
        raise ValueError(f"{table} contains non-finite values")
    return values


def load_ids() -> tuple[list[int], list[int]]:
    with psycopg2.connect(**DB_PARAMS) as conn, conn.cursor() as cur:
        cur.execute("SELECT compound_id FROM train_activity ORDER BY id")
        train_ids = [int(row[0]) for row in cur.fetchall()]
        cur.execute("SELECT compound_id FROM test_activity ORDER BY id")
        test_ids = [int(row[0]) for row in cur.fetchall()]
    return train_ids, test_ids


def build_features(
    case: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_ids: list[int],
    test_ids: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    raw = case.startswith("raw2048")
    table = TABLE if raw else "compound_chemeleon"
    dim = EXPECTED_DIM if raw else 300
    embedding_train = load_embedding_table(table, train_ids, dim)
    embedding_test = load_embedding_table(table, test_ids, dim)

    if "mixed" not in case:
        return embedding_train, embedding_test

    base_train, base_test = run_train.load_features(
        "2d_full_boltz_log2fc_pred_seed10ens", train_df, test_df
    )
    return (
        np.concatenate([embedding_train, base_train], axis=1),
        np.concatenate([embedding_test, base_test], axis=1),
    )


def impute_from_train(
    train: np.ndarray, test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    means = np.nanmean(train, axis=0)
    means = np.where(np.isfinite(means), means, 0.0)
    return (
        np.where(np.isfinite(train), train, means).astype(np.float32),
        np.where(np.isfinite(test), test, means).astype(np.float32),
    )


def run_case(
    case: str,
    args: argparse.Namespace,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_ids: list[int],
    test_ids: list[int],
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> dict:
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion

    model_path = TabPFNRegressor.create_default_for_version(
        ModelVersion.V2_6
    ).model_path
    X_train, X_test = build_features(case, train_df, test_df, train_ids, test_ids)
    X_train, X_test = impute_from_train(X_train, X_test)
    y = train_df["pec50"].to_numpy(dtype=np.float32)
    use_topk = case.endswith("top500")
    k = min(args.top_k, X_train.shape[1])
    print(f"\nCASE {case}: train={X_train.shape} test={X_test.shape} topk={use_topk}")

    oof = np.zeros(len(y), dtype=np.float32)
    test_folds: list[np.ndarray] = []
    fold_metrics: list[dict] = []
    selected_family_counts: list[dict[str, int]] = []
    embedding_dim = EXPECTED_DIM if case.startswith("raw2048") else 300

    for fold, (fit_idx, val_idx) in enumerate(splits):
        if use_topk:
            selector = lgb.LGBMRegressor(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=63,
                min_child_samples=10,
                random_state=args.seed,
                verbose=-1,
            )
            selector.fit(X_train[fit_idx], y[fit_idx])
            gain = selector.booster_.feature_importance(importance_type="gain")
            selected = np.argsort(-gain)[:k]
            selected_family_counts.append(
                {
                    "chemeleon": int(np.sum(selected < embedding_dim)),
                    "base_2d_boltz_pred": int(np.sum(selected >= embedding_dim)),
                    "zero_gain": int(np.sum(gain[selected] == 0)),
                }
            )
        else:
            selected = np.arange(X_train.shape[1])

        regressor = TabPFNRegressor(
            device="cuda",
            n_estimators=args.n_estimators,
            softmax_temperature=args.softmax_temperature,
            random_state=args.seed,
            ignore_pretraining_limits=True,
            model_path=model_path,
        )
        regressor.fit(X_train[fit_idx][:, selected], y[fit_idx])
        oof[val_idx] = regressor.predict(X_train[val_idx][:, selected])
        test_folds.append(regressor.predict(X_test[:, selected]))
        metrics = compute_metrics(y[val_idx], oof[val_idx])
        fold_metrics.append(metrics)
        print(
            f"  fold={fold} selected={len(selected)} "
            f"MAE={metrics['MAE']:.4f} RAE={metrics['RAE']:.4f} "
            f"Sp={metrics['Spearman_R']:.4f}"
        )
        del regressor

    overall = compute_metrics(y, oof)
    test_prediction = np.mean(np.stack(test_folds), axis=0)
    case_name = case
    if use_topk and args.top_k != 500:
        case_name = case.replace("top500", f"top{args.top_k}")
    exp_name = f"audit_chemeleon_{case_name.replace('-', '_')}_tabpfn_v2_6_umap"
    submission_path = SUBMISSION_DIR.joinpath(f"{exp_name}.csv")
    pd.DataFrame(
        {
            "SMILES": test_df["smiles"],
            "Molecule Name": test_df["molecule_name"],
            "pEC50": test_prediction,
        }
    ).to_csv(submission_path, index=False)

    hyperparameters = {
        "audit": "chemeleon_raw2048_validation",
        "tabpfn_version": "v2_6",
        "n_estimators": args.n_estimators,
        "softmax_temperature": args.softmax_temperature,
        "seed": args.seed,
        "input_dim": int(X_train.shape[1]),
        "top_k": k if use_topk else None,
        "topk_selector": "fold-local LightGBM gain" if use_topk else None,
        "selected_family_counts": selected_family_counts,
    }
    exp_id = record_experiment(
        name=exp_name,
        description=(
            f"CheMeleon raw-2048 audit case {case_name}; TabPFN v2.6 on canonical "
            "UMAP 5-fold CV. Legacy compound_chemeleon remains untouched."
        ),
        model_type="tabpfn",
        feature_set=case,
        hyperparameters=hyperparameters,
        fold_metrics=fold_metrics,
        submission_path=str(submission_path.relative_to(REPO_ROOT)),
        notes=(
            f"Overall OOF MAE={overall['MAE']:.4f}, RAE={overall['RAE']:.4f}, "
            f"Spearman={overall['Spearman_R']:.4f}."
        ),
    )
    save_oof_predictions(exp_id, oof)
    print(
        f"  overall MAE={overall['MAE']:.4f} RAE={overall['RAE']:.4f} "
        f"Sp={overall['Spearman_R']:.4f}; experiment_id={exp_id}"
    )
    return {
        "case": case_name,
        "experiment_id": exp_id,
        "experiment_name": exp_name,
        "input_dim": int(X_train.shape[1]),
        "fold_metrics": fold_metrics,
        "overall": overall,
        "selected_family_counts": selected_family_counts,
        "submission_path": str(submission_path.relative_to(REPO_ROOT)),
    }


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_df = load_train_smiles_target()
    test_df = load_test_smiles()
    train_ids, test_ids = load_ids()
    splits = umap_split_indices(
        train_df["smiles"].tolist(), n_splits=5, n_clusters=50, seed=args.seed
    )

    results = []
    for case in args.cases:
        results.append(
            run_case(
                case,
                args,
                train_df,
                test_df,
                train_ids,
                test_ids,
                splits,
            )
        )

    output_path = OUTPUT_DIR.joinpath(
        f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    output_path.write_text(
        json.dumps(
            {
                "args": vars(args),
                "results": results,
            },
            indent=2,
            default=lambda value: (
                value.item() if isinstance(value, np.generic) else str(value)
            ),
        )
        + "\n"
    )
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env -S pixi run python
"""Build the Phase 2 validation matrix.

This script keeps two Phase 2 validation roles separate:

1. AS1 external benchmark: score existing test predictions against released AS1
   labels without using AS1 for training.
2. Phase2 labeled OOF: treat train + AS1 as the labeled pool, build a fresh CV
   split, and generate out-of-fold predictions from models that do not see their
   own validation rows.

The first role is the LB replacement for old submissions. The second role is the
new development proxy for models trained after AS1 was released.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "scripts")))

from data import get_engine, load_test_smiles, load_train_smiles_target  # noqa: E402
from splits import _morgan_fp_matrix  # noqa: E402

import run_train  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.joinpath("outputs")
SUBMISSION_DIR = REPO_ROOT / "track1_activity" / "submissions"
DOC_PATH = REPO_ROOT / "docs" / "track1_explain" / "phase2_validation_matrix.md"

DEFAULT_FEATURE = "cheme_2d_full_boltz_log2fc_pred_seed10ens"
DEFAULT_SUBMISSIONS = [
    "ens_id51_top500_potent46_t40_soft_g35.csv",
    "ens_id51_top500_potent46_t40_soft_g50.csv",
    "ens_id55_combo_gate_rank1.csv",
    "ens_id57_high_activity_lift_rank2.csv",
    "ens_swap_optuna_t10_top500_calibrated_importance.csv",
]

TRUE_BINS = [-np.inf, 3.0, 4.0, 5.0, 6.0, np.inf]
TRUE_BIN_LABELS = ["lt3", "3to4", "4to5", "5to6", "gte6"]


def load_train_metadata() -> pd.DataFrame:
    return pd.read_sql(
        """
        SELECT
            t.id AS original_row_id,
            c.id AS compound_id,
            c.molecule_name,
            c.std_smiles AS smiles,
            t.pec50
        FROM train_activity t
        JOIN compounds c ON c.id = t.compound_id
        ORDER BY t.id
        """,
        get_engine(),
    )


def load_as1_metadata() -> pd.DataFrame:
    return pd.read_sql(
        """
        SELECT
            t.id AS original_row_id,
            c.id AS compound_id,
            c.molecule_name,
            c.std_smiles AS smiles,
            l.pec50
        FROM test_activity_phase1_labels l
        JOIN test_activity t ON t.compound_id = l.compound_id
        JOIN compounds c ON c.id = l.compound_id
        ORDER BY t.id
        """,
        get_engine(),
    )


def build_labeled_pool() -> pd.DataFrame:
    train = load_train_metadata()
    as1 = load_as1_metadata()
    train["source"] = "train"
    as1["source"] = "as1"
    pool = pd.concat([train, as1], ignore_index=True)
    pool.insert(0, "pool_idx", np.arange(len(pool), dtype=np.int64))
    pool["true_bin"] = pd.cut(pool["pec50"], TRUE_BINS, labels=TRUE_BIN_LABELS).astype(
        "object"
    )
    return pool


def metric_row(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    err = pred - y
    spearman = stats.spearmanr(y, pred).statistic if len(y) >= 2 else np.nan
    return {
        "n": int(len(y)),
        "mae": float(np.mean(np.abs(err))),
        "bias_pred_minus_true": float(np.mean(err)),
        "spearman": float(spearman),
        "pred_mean": float(np.mean(pred)),
        "true_mean": float(np.mean(y)),
    }


def load_submission(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"Molecule Name", "pEC50"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    return df.rename(columns={"Molecule Name": "molecule_name", "pEC50": "pred"})[
        ["molecule_name", "pred"]
    ]


def discover_submission_paths(extra: list[str]) -> list[Path]:
    paths: list[Path] = []
    for name in DEFAULT_SUBMISSIONS + extra:
        path = Path(name)
        if not path.is_absolute():
            path = SUBMISSION_DIR / path
        if path.exists():
            paths.append(path)

    exp_paths = pd.read_sql(
        """
        SELECT submission_path
        FROM experiments
        WHERE submission_path IS NOT NULL
        ORDER BY created_at DESC
        """,
        get_engine(),
    )
    for raw in exp_paths["submission_path"].dropna().tolist():
        path = Path(raw)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if path.exists():
            paths.append(path)

    unique: dict[str, Path] = {}
    for path in paths:
        unique[str(path.resolve())] = path
    return list(unique.values())


def build_as1_external_benchmark(extra_submissions: list[str]) -> pd.DataFrame:
    as1 = load_as1_metadata()[["molecule_name", "pec50"]].copy()
    as1["true_bin"] = pd.cut(as1["pec50"], TRUE_BINS, labels=TRUE_BIN_LABELS).astype(
        "object"
    )
    rows = []
    for path in discover_submission_paths(extra_submissions):
        try:
            pred = load_submission(path)
        except Exception as exc:
            rows.append(
                {
                    "candidate": path.stem,
                    "path": str(path.relative_to(REPO_ROOT)),
                    "error": str(exc),
                }
            )
            continue
        merged = as1.merge(pred, on="molecule_name", how="inner")
        if len(merged) != len(as1):
            rows.append(
                {
                    "candidate": path.stem,
                    "path": str(path.relative_to(REPO_ROOT)),
                    "error": f"aligned {len(merged)} of {len(as1)} AS1 rows",
                }
            )
            continue
        row = {
            "candidate": path.stem,
            "path": str(path.relative_to(REPO_ROOT)),
            "error": "",
            **metric_row(
                merged["pec50"].to_numpy(dtype=np.float64),
                merged["pred"].to_numpy(dtype=np.float64),
            ),
        }
        for label, sub in merged.groupby("true_bin", observed=True):
            row[f"mae_bin_{label}"] = float(np.mean(np.abs(sub["pred"] - sub["pec50"])))
            row[f"bias_bin_{label}"] = float(np.mean(sub["pred"] - sub["pec50"]))
        rows.append(row)

    out = pd.DataFrame(rows)
    if "mae" in out:
        out = out.sort_values(["error", "mae"], na_position="last")
    return out


def build_phase2_splits(
    pool: pd.DataFrame,
    n_splits: int,
    n_clusters: int,
    seed: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], pd.DataFrame]:
    import umap
    from sklearn.cluster import KMeans

    fps = _morgan_fp_matrix(pool["smiles"].tolist())
    embedding = umap.UMAP(
        n_components=10,
        metric="jaccard",
        random_state=seed,
        n_neighbors=30,
    ).fit_transform(fps)
    cluster_labels = KMeans(
        n_clusters=n_clusters,
        random_state=seed,
        n_init=10,
    ).fit_predict(embedding)

    cluster_to_indices: dict[int, list[int]] = defaultdict(list)
    for idx, cluster in enumerate(cluster_labels):
        cluster_to_indices[int(cluster)].append(idx)

    target_total = len(pool) / n_splits
    target_as1 = float((pool["source"] == "as1").sum()) / n_splits
    fold_indices: list[list[int]] = [[] for _ in range(n_splits)]
    fold_total = np.zeros(n_splits, dtype=np.float64)
    fold_as1 = np.zeros(n_splits, dtype=np.float64)

    cluster_groups = []
    is_as1 = pool["source"].eq("as1").to_numpy()
    for cluster, indices in cluster_to_indices.items():
        idx = np.array(indices, dtype=np.int64)
        cluster_groups.append(
            {
                "cluster": cluster,
                "idx": idx,
                "n": len(idx),
                "n_as1": int(is_as1[idx].sum()),
            }
        )
    cluster_groups = sorted(
        cluster_groups,
        key=lambda row: (row["n_as1"], row["n"]),
        reverse=True,
    )

    for group in cluster_groups:
        scores = []
        for fold in range(n_splits):
            next_total = fold_total.copy()
            next_as1 = fold_as1.copy()
            next_total[fold] += group["n"]
            next_as1[fold] += group["n_as1"]
            total_penalty = np.sum(((next_total - target_total) / target_total) ** 2)
            as1_penalty = np.sum(((next_as1 - target_as1) / max(target_as1, 1.0)) ** 2)
            scores.append(float(total_penalty + 2.0 * as1_penalty))
        fold = int(np.argmin(scores))
        fold_indices[fold].extend(group["idx"].tolist())
        fold_total[fold] += group["n"]
        fold_as1[fold] += group["n_as1"]

    rng = np.random.RandomState(seed)
    splits = []
    all_idx = np.arange(len(pool), dtype=np.int64)
    for fold in range(n_splits):
        val_idx = np.array(fold_indices[fold], dtype=np.int64)
        rng.shuffle(val_idx)
        train_mask = np.ones(len(pool), dtype=bool)
        train_mask[val_idx] = False
        train_idx = all_idx[train_mask]
        splits.append((train_idx, val_idx))

    rows = []
    for fold, (_train_idx, val_idx) in enumerate(splits):
        val = pool.iloc[val_idx]
        rows.append(
            {
                "fold": fold,
                "n_val": int(len(val)),
                "n_val_train_source": int((val["source"] == "train").sum()),
                "n_val_as1_source": int((val["source"] == "as1").sum()),
                "val_y_mean": float(val["pec50"].mean()),
                "val_y_std": float(val["pec50"].std(ddof=1)),
                "val_min": float(val["pec50"].min()),
                "val_max": float(val["pec50"].max()),
            }
        )
    return splits, pd.DataFrame(rows)


def build_phase2_feature_matrix(feature: str, pool: pd.DataFrame) -> np.ndarray:
    train_df = load_train_smiles_target()
    test_df = load_test_smiles()
    X_train, X_test = run_train.load_features(feature, train_df, test_df)

    train_lookup = {
        int(cid): i for i, cid in enumerate(load_train_metadata()["compound_id"])
    }
    as1_test_ids = load_as1_metadata()[["compound_id", "original_row_id"]]
    test_id_lookup = {
        int(row.compound_id): int(row.original_row_id) - 1
        for row in as1_test_ids.itertuples(index=False)
    }

    mats = []
    for row in pool.itertuples(index=False):
        if row.source == "train":
            mats.append(X_train[train_lookup[int(row.compound_id)]])
        else:
            mats.append(X_test[test_id_lookup[int(row.compound_id)]])
    X = np.vstack(mats).astype(np.float32)
    col_mean = np.nanmean(X, axis=0)
    col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0).astype(np.float32)
    X = np.where(np.isfinite(X), X, col_mean).astype(np.float32)
    return X


def run_lgbm_topk_oof(
    pool: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
    feature: str,
    top_k: int,
    seed: int,
    ranker_estimators: int,
    model_estimators: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    X = build_phase2_feature_matrix(feature, pool)
    y = pool["pec50"].to_numpy(dtype=np.float32)
    oof = np.full(len(pool), np.nan, dtype=np.float64)
    fold_rows = []

    for fold, (train_idx, val_idx) in enumerate(splits):
        print(
            f"fold {fold}: train={len(train_idx)} val={len(val_idx)} "
            f"as1_val={int((pool.iloc[val_idx]['source'] == 'as1').sum())}"
        )
        ranker = lgb.LGBMRegressor(
            n_estimators=ranker_estimators,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=10,
            random_state=seed + fold,
            verbose=-1,
        )
        ranker.fit(X[train_idx], y[train_idx])
        gain = ranker.booster_.feature_importance(importance_type="gain")
        selected = np.argsort(-gain)[:top_k]

        model = lgb.LGBMRegressor(
            n_estimators=model_estimators,
            learning_rate=0.02,
            num_leaves=63,
            min_child_samples=20,
            subsample=0.8,
            subsample_freq=5,
            colsample_bytree=0.7,
            reg_alpha=0.01,
            reg_lambda=1.0,
            random_state=seed + fold,
            objective="regression_l1",
            verbose=-1,
        )
        model.fit(
            X[train_idx][:, selected],
            y[train_idx],
            eval_set=[(X[val_idx][:, selected], y[val_idx])],
            eval_metric="l1",
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
        )
        pred = model.predict(X[val_idx][:, selected])
        oof[val_idx] = pred
        metrics = metric_row(y[val_idx].astype(np.float64), pred.astype(np.float64))
        fold_rows.append(
            {
                "fold": fold,
                "best_iteration": int(model.best_iteration_ or model_estimators),
                "zero_gain_selected": int((gain[selected] == 0).sum()),
                **metrics,
            }
        )

    if np.isnan(oof).any():
        raise RuntimeError("OOF contains NaN rows after CV")

    oof_df = pool.copy()
    oof_df["phase2_oof_pred"] = oof
    oof_df["phase2_oof_error"] = oof_df["phase2_oof_pred"] - oof_df["pec50"]
    oof_df["phase2_oof_abs_error"] = oof_df["phase2_oof_error"].abs()
    return oof_df, pd.DataFrame(fold_rows)


def summarize_oof(oof: pd.DataFrame) -> pd.DataFrame:
    rows = []
    masks = {
        "all": pd.Series(True, index=oof.index),
        "source_train": oof["source"].eq("train"),
        "source_as1": oof["source"].eq("as1"),
        "true_lt3": oof["pec50"] < 3.0,
        "true_gte6": oof["pec50"] >= 6.0,
    }
    for label in TRUE_BIN_LABELS:
        masks[f"bin_{label}"] = oof["true_bin"].eq(label)
    for name, mask in masks.items():
        sub = oof.loc[mask]
        rows.append(
            {
                "slice": name,
                **metric_row(
                    sub["pec50"].to_numpy(dtype=np.float64),
                    sub["phase2_oof_pred"].to_numpy(dtype=np.float64),
                ),
            }
        )
    return pd.DataFrame(rows)


def write_report(
    pool: pd.DataFrame,
    split_summary: pd.DataFrame,
    as1_benchmark: pd.DataFrame,
    oof_summary: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    best_as1 = as1_benchmark[as1_benchmark["error"].fillna("").eq("")].head(8)
    lines = [
        "# Phase 2 validation matrix",
        "",
        "This matrix separates AS1 external validation from the new `train + AS1`",
        "cross-fit OOF. The two numbers answer different questions and should not",
        "be merged into one score prematurely.",
        "",
        "## Labeled pool",
        "",
        f"- train rows: {(pool['source'] == 'train').sum()}",
        f"- AS1 rows: {(pool['source'] == 'as1').sum()}",
        f"- total labeled rows: {len(pool)}",
        "",
        "## Phase2 OOF recipe",
        "",
        f"- feature: `{args.feature}`",
        f"- model: LightGBM top-{args.top_k}",
        f"- folds: {args.n_splits}, UMAP clusters: {args.n_clusters}, seed: {args.seed}",
        "",
        "## Split summary",
        "",
        split_summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Fold metrics",
        "",
        fold_metrics.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Phase2 OOF slices",
        "",
        oof_summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## AS1 external benchmark, top rows",
        "",
        best_as1[
            ["candidate", "n", "mae", "bias_pred_minus_true", "spearman", "path"]
        ].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Interpretation",
        "",
        "- `AS1 external` remains the fixed LB replacement for already-built test predictions.",
        "- `Phase2 OOF` is the development proxy for models that train on `train + AS1`.",
        "- AS1-only wins should be treated as overfit risk unless Phase2 OOF and train-source slices also hold up.",
    ]
    DOC_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature", default=DEFAULT_FEATURE)
    parser.add_argument("--top-k", type=int, default=500)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-clusters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ranker-estimators", type=int, default=500)
    parser.add_argument("--model-estimators", type=int, default=2000)
    parser.add_argument(
        "--extra-submission",
        action="append",
        default=[],
        help="Additional submission CSV path or filename to include in AS1 benchmark.",
    )
    parser.add_argument(
        "--skip-oof",
        action="store_true",
        help="Only build pool/splits/AS1 benchmark; do not retrain OOF.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)

    pool = build_labeled_pool()
    pool.to_csv(OUT_DIR / "phase2_labeled_pool.csv", index=False)

    as1_benchmark = build_as1_external_benchmark(args.extra_submission)
    as1_benchmark.to_csv(OUT_DIR / "as1_external_benchmark.csv", index=False)

    splits, split_summary = build_phase2_splits(
        pool, args.n_splits, args.n_clusters, args.seed
    )
    split_summary.to_csv(OUT_DIR / "phase2_split_summary.csv", index=False)
    split_assign = np.full(len(pool), -1, dtype=np.int64)
    for fold, (_train_idx, val_idx) in enumerate(splits):
        split_assign[val_idx] = fold
    pool.assign(fold=split_assign).to_csv(
        OUT_DIR / "phase2_labeled_pool_with_folds.csv", index=False
    )

    if args.skip_oof:
        oof = pd.DataFrame()
        fold_metrics = pd.DataFrame()
        oof_summary = pd.DataFrame()
    else:
        oof, fold_metrics = run_lgbm_topk_oof(
            pool=pool,
            splits=splits,
            feature=args.feature,
            top_k=args.top_k,
            seed=args.seed,
            ranker_estimators=args.ranker_estimators,
            model_estimators=args.model_estimators,
        )
        oof_summary = summarize_oof(oof)
        oof.to_csv(OUT_DIR / "phase2_lgbm_topk_oof_predictions.csv", index=False)
        fold_metrics.to_csv(OUT_DIR / "phase2_lgbm_topk_fold_metrics.csv", index=False)
        oof_summary.to_csv(OUT_DIR / "phase2_lgbm_topk_oof_summary.csv", index=False)

    metadata = {
        "feature": args.feature,
        "top_k": args.top_k,
        "n_splits": args.n_splits,
        "n_clusters": args.n_clusters,
        "seed": args.seed,
        "ranker_estimators": args.ranker_estimators,
        "model_estimators": args.model_estimators,
        "skip_oof": args.skip_oof,
    }
    (OUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if not args.skip_oof:
        write_report(pool, split_summary, as1_benchmark, oof_summary, fold_metrics, args)

    print(f"wrote {OUT_DIR / 'phase2_labeled_pool.csv'}")
    print(f"wrote {OUT_DIR / 'as1_external_benchmark.csv'}")
    print(f"wrote {OUT_DIR / 'phase2_labeled_pool_with_folds.csv'}")
    if not args.skip_oof:
        print(f"wrote {OUT_DIR / 'phase2_lgbm_topk_oof_predictions.csv'}")
        print(f"wrote {OUT_DIR / 'phase2_lgbm_topk_oof_summary.csv'}")
        print(f"wrote {DOC_PATH}")


if __name__ == "__main__":
    main()

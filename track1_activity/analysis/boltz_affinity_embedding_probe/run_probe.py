#!/usr/bin/env -S pixi run python
"""Replay AS1 as a blind test for Boltz affinity-module embeddings.

This is an experiment-only Phase 1 style replay:

- fit on the original Track 1 train_activity labels only
- predict the released AS1 rows from test_activity
- compare Boltz affinity-module embeddings against existing pooled trunk views

No submission files or experiment DB rows are written.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent / "outputs"
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "scripts")))

import run_train  # noqa: E402
from data import DB_PARAMS, load_test_smiles, load_train_smiles_target  # noqa: E402


@dataclass(frozen=True)
class FeatureConfig:
    name: str
    kind: str


FEATURES = [
    FeatureConfig("boltz_affinity_g1g2", "affinity"),
    FeatureConfig("boltz_affinity_g1g2_scalars", "affinity"),
    FeatureConfig("boltz_affinity_gmean", "affinity"),
    FeatureConfig("pooled_boltz", "run_train"),
    FeatureConfig("pooled_boltz_allpairs", "run_train"),
]


def load_ids(table: str) -> list[int]:
    order_col = "id"
    sql = f"SELECT compound_id FROM {table} ORDER BY {order_col}"
    with psycopg2.connect(**DB_PARAMS) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            return [int(row[0]) for row in cur.fetchall()]


def load_as1() -> pd.DataFrame:
    sql = """
    SELECT
        (t.id - 1)::int AS test_idx,
        t.compound_id,
        c.smiles,
        c.molecule_name,
        l.pec50
    FROM test_activity_phase1_labels l
    JOIN test_activity t ON t.compound_id = l.compound_id
    JOIN compounds c ON c.id = t.compound_id
    ORDER BY t.id
    """
    with psycopg2.connect(**DB_PARAMS) as conn:
        return pd.read_sql(sql, conn)


def _affinity_rows(compound_ids: list[int]) -> pd.DataFrame:
    sql = """
    SELECT compound_id,
           affinity_pred_value,
           affinity_probability_binary,
           affinity_pred_value_1,
           affinity_probability_binary_1,
           affinity_pred_value_2,
           affinity_probability_binary_2,
           affinity_g1,
           affinity_g2
    FROM compound_boltz2_affinity_reuse
    WHERE compound_id = ANY(%s)
    """
    with psycopg2.connect(**DB_PARAMS) as conn:
        df = pd.read_sql(sql, conn, params=(compound_ids,))
    return df.set_index("compound_id")


def load_affinity_feature(
    feature: str, train_ids: list[int], as1_ids: list[int]
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    ids = train_ids + as1_ids
    df = _affinity_rows(ids)
    missing_train = [cid for cid in train_ids if cid not in df.index]
    missing_as1 = [cid for cid in as1_ids if cid not in df.index]
    if missing_as1:
        raise ValueError(
            f"{feature} missing {len(missing_as1)} AS1 compounds in "
            f"compound_boltz2_affinity_reuse: {missing_as1[:10]}"
        )
    if missing_train:
        print(
            f"  {feature}: dropping {len(missing_train)} train rows without "
            f"affinity embeddings: {missing_train[:10]}"
        )
    train_ids_used = [cid for cid in train_ids if cid in df.index]

    def one(cid: int) -> np.ndarray:
        row = df.loc[cid]
        g1 = np.asarray(row["affinity_g1"], dtype=np.float32)
        g2 = np.asarray(row["affinity_g2"], dtype=np.float32)
        if feature == "boltz_affinity_g1g2":
            return np.concatenate([g1, g2])
        if feature == "boltz_affinity_gmean":
            return ((g1 + g2) * 0.5).astype(np.float32)
        if feature == "boltz_affinity_g1g2_scalars":
            scalars = np.asarray(
                [
                    row["affinity_pred_value"],
                    row["affinity_probability_binary"],
                    row["affinity_pred_value_1"],
                    row["affinity_probability_binary_1"],
                    row["affinity_pred_value_2"],
                    row["affinity_probability_binary_2"],
                ],
                dtype=np.float32,
            )
            return np.concatenate([g1, g2, scalars])
        raise ValueError(f"Unknown affinity feature {feature!r}")

    X_train = np.stack([one(cid) for cid in train_ids_used]).astype(np.float32)
    X_as1 = np.stack([one(cid) for cid in as1_ids]).astype(np.float32)
    return X_train, X_as1, train_ids_used


def load_feature(
    feature: FeatureConfig, as1: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    train_ids = load_ids("train_activity")
    as1_ids = as1["compound_id"].astype(int).tolist()
    if feature.kind == "affinity":
        return load_affinity_feature(feature.name, train_ids, as1_ids)

    train_df = load_train_smiles_target()
    test_df = load_test_smiles()
    X_train, X_test = run_train.load_features(feature.name, train_df, test_df)
    return (
        X_train.astype(np.float32),
        X_test[as1["test_idx"].to_numpy(dtype=np.int64)].astype(np.float32),
        train_ids,
    )


def finite_impute(
    X_train: np.ndarray, X_eval: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    X_train = X_train.copy()
    X_eval = X_eval.copy()
    train_finite = np.where(np.isfinite(X_train), X_train, np.nan)
    col_mean = np.nanmean(train_finite, axis=0)
    col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0).astype(np.float32)
    X_train[~np.isfinite(X_train)] = np.broadcast_to(col_mean, X_train.shape)[
        ~np.isfinite(X_train)
    ]
    X_eval[~np.isfinite(X_eval)] = np.broadcast_to(col_mean, X_eval.shape)[
        ~np.isfinite(X_eval)
    ]
    return X_train.astype(np.float32), X_eval.astype(np.float32)


def metric_row(y: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    err = pred - y
    return {
        "n": int(len(y)),
        "mae": float(np.mean(np.abs(err))),
        "bias_pred_minus_true": float(np.mean(err)),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "spearman": float(stats.spearmanr(y, pred).statistic),
        "pearson": float(stats.pearsonr(y, pred).statistic),
        "pred_mean": float(np.mean(pred)),
        "pred_std": float(np.std(pred)),
        "true_mean": float(np.mean(y)),
        "true_std": float(np.std(y)),
    }


def summarize_slices(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    masks = {
        "all": pd.Series(True, index=pred_df.index),
        "true_lt3": pred_df["pec50"] < 3.0,
        "true_3_4": pred_df["pec50"].between(3.0, 4.0, inclusive="left"),
        "true_4_5": pred_df["pec50"].between(4.0, 5.0, inclusive="left"),
        "true_5_6": pred_df["pec50"].between(5.0, 6.0, inclusive="left"),
        "true_gte6": pred_df["pec50"] >= 6.0,
    }
    for name, mask in masks.items():
        sub = pred_df.loc[mask]
        if len(sub) < 2:
            continue
        rows.append(
            {
                "slice": name,
                **metric_row(
                    sub["pec50"].to_numpy(dtype=np.float64),
                    sub["prediction"].to_numpy(dtype=np.float64),
                ),
            }
        )
    return pd.DataFrame(rows)


def fit_predict_tabpfn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_as1: np.ndarray,
    args: argparse.Namespace,
    seed: int,
) -> np.ndarray:
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion

    version_enum = {
        "v3": ModelVersion.V3,
        "v2_6": ModelVersion.V2_6,
        "v2_5": ModelVersion.V2_5,
        "v2": ModelVersion.V2,
    }[args.tabpfn_version]
    model_path = TabPFNRegressor.create_default_for_version(version_enum).model_path
    model = TabPFNRegressor(
        device=args.device,
        n_estimators=args.n_estimators,
        softmax_temperature=args.softmax_temperature,
        average_before_softmax=args.average_before_softmax,
        random_state=seed,
        model_path=model_path,
        ignore_pretraining_limits=X_train.shape[1] > 500,
    )
    model.fit(X_train, y_train)
    return model.predict(X_as1).astype(np.float64)


def choose_features(names: list[str]) -> list[FeatureConfig]:
    if not names or names == ["all"]:
        return FEATURES
    by_name = {f.name: f for f in FEATURES}
    chosen = []
    for name in names:
        if name not in by_name:
            raise SystemExit(f"Unknown feature {name!r}. Known: {', '.join(by_name)}")
        chosen.append(by_name[name])
    return chosen


def write_report(out_dir: Path) -> None:
    summary = pd.read_csv(out_dir / "summary.csv")
    overall = summary[summary["slice"] == "all"].sort_values("mae")
    corr_path = out_dir / "prediction_correlations.csv"
    corr = pd.read_csv(corr_path, index_col=0) if corr_path.exists() else None
    lines = [
        "# Boltz Affinity Embedding AS1 Replay",
        "",
        "Phase 1 style replay: models fit on original train labels only; AS1 is held out as test.",
        "",
        "## Overall",
        "",
        overall[
            [
                "feature",
                "tabpfn_version",
                "n_features",
                "n",
                "mae",
                "bias_pred_minus_true",
                "spearman",
                "pearson",
                "pred_std",
            ]
        ].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Slices",
        "",
        summary.pivot_table(
            index=["feature", "tabpfn_version", "n_features"],
            columns="slice",
            values="mae",
            aggfunc="first",
        )
        .reset_index()
        .to_markdown(index=False, floatfmt=".4f"),
    ]
    if corr is not None:
        lines.extend(
            ["", "## Prediction Correlations", "", corr.to_markdown(floatfmt=".4f")]
        )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("feature", nargs="*", default=["all"])
    parser.add_argument(
        "--tabpfn-version", choices=["v3", "v2_6", "v2_5", "v2"], default="v2_6"
    )
    parser.add_argument("--n-estimators", type=int, default=8)
    parser.add_argument("--softmax-temperature", type=float, default=0.9)
    parser.add_argument("--average-before-softmax", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = OUT_DIR / (
        f"tabpfn_{args.tabpfn_version}_ne{args.n_estimators}_"
        f"t{str(args.softmax_temperature).replace('.', 'p')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    as1 = load_as1()
    train_df = load_train_smiles_target()
    all_train_ids = load_ids("train_activity")
    y_by_compound = dict(
        zip(all_train_ids, train_df["pec50"].to_numpy(dtype=np.float32), strict=True)
    )

    selected = choose_features(args.feature)
    summaries = []
    pred_cols = {}
    for feature in selected:
        pred_path = out_dir / f"{feature.name}__predictions.csv"
        summary_path = out_dir / f"{feature.name}__summary.csv"
        if pred_path.exists() and summary_path.exists() and not args.force:
            print(f"skip existing {feature.name}")
            pred_df = pd.read_csv(pred_path)
            pred_cols[feature.name] = pred_df["prediction"].to_numpy(dtype=np.float64)
            summaries.append(pd.read_csv(summary_path))
            continue

        print(f"\n=== {feature.name} ({args.tabpfn_version}) ===")
        X_train, X_as1, train_ids_used = load_feature(feature, as1)
        X_train, X_as1 = finite_impute(X_train, X_as1)
        y_train = np.asarray(
            [y_by_compound[int(cid)] for cid in train_ids_used], dtype=np.float32
        )
        print(f"X_train={X_train.shape} X_as1={X_as1.shape}")
        pred = fit_predict_tabpfn(X_train, y_train, X_as1, args, seed=args.seed)
        pred_cols[feature.name] = pred

        pred_df = as1.copy()
        pred_df["prediction"] = pred
        pred_df["error"] = pred_df["prediction"] - pred_df["pec50"]
        pred_df["abs_error"] = pred_df["error"].abs()
        pred_df.to_csv(pred_path, index=False)

        summary = summarize_slices(pred_df)
        summary.insert(0, "feature", feature.name)
        summary.insert(1, "tabpfn_version", args.tabpfn_version)
        summary.insert(2, "n_features", int(X_train.shape[1]))
        summary.to_csv(summary_path, index=False)
        summaries.append(summary)
        print(summary[summary["slice"] == "all"].to_string(index=False))

    combined = pd.concat(summaries, ignore_index=True)
    combined.to_csv(out_dir / "summary.csv", index=False)
    if len(pred_cols) >= 2:
        pred_mat = pd.DataFrame(pred_cols)
        pred_mat.corr(method="pearson").to_csv(out_dir / "prediction_correlations.csv")
    metadata = {
        "tabpfn_version": args.tabpfn_version,
        "n_estimators": args.n_estimators,
        "softmax_temperature": args.softmax_temperature,
        "average_before_softmax": args.average_before_softmax,
        "device": args.device,
        "seed": args.seed,
        "features": [asdict(f) for f in selected],
        "train_rows": int(len(train_df)),
        "as1_rows": int(len(as1)),
        "note": "Experiment only; no submission files or experiment DB rows written.",
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    write_report(out_dir)
    print(f"\nwrote {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()

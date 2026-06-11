#!/usr/bin/env -S pixi run python
"""Probe whether pred_htchem belongs in the LGBM-gain top500 feature pool."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "scripts")))

import run_train  # noqa: E402
from data import get_engine  # noqa: E402
from splits import umap_split_indices  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "outputs" / "top500_gain"
DOC_PATH = REPO_ROOT / "docs" / "track1_explain" / "phase2_htchem_top500_gain.md"

TRAIN_TEST_EMBED = REPO_ROOT / "data" / "chemprop_pretrain_embed.parquet"
HTCHEM_EMBED = REPO_ROOT / "data" / "chemprop_pretrain_htchem_embed.parquet"
ALPHAS = np.logspace(-2, 4, 25)


def load_train() -> pd.DataFrame:
    return pd.read_sql(
        """
        SELECT
            t.id AS compound_id,
            t.pec50,
            c.std_smiles AS smiles,
            c.molecule_name
        FROM train_activity t
        JOIN compounds c ON c.id = t.compound_id
        ORDER BY t.id
        """,
        get_engine(),
    )


def load_test() -> pd.DataFrame:
    return pd.read_sql(
        """
        SELECT
            t.id AS compound_id,
            c.std_smiles AS smiles,
            c.molecule_name
        FROM test_activity t
        JOIN compounds c ON c.id = t.compound_id
        ORDER BY t.id
        """,
        get_engine(),
    )


def load_htchem_labels() -> pd.DataFrame:
    return pd.read_sql(
        """
        SELECT h.compound_id, h.corrected_pec50
        FROM htchem_activity h
        WHERE h.corrected_pec50 IS NOT NULL
        ORDER BY h.compound_id
        """,
        get_engine(),
    )


def pred_htchem_for_challenge() -> pd.Series:
    htchem = load_htchem_labels()
    htchem_embed = pd.read_parquet(HTCHEM_EMBED)
    challenge_embed = pd.read_parquet(TRAIN_TEST_EMBED)

    x_ht = htchem_embed.loc[htchem["compound_id"].astype(int)].to_numpy(
        dtype=np.float32
    )
    y_ht = htchem["corrected_pec50"].to_numpy(dtype=np.float64)
    model = make_pipeline(
        StandardScaler(),
        RidgeCV(alphas=ALPHAS, scoring="neg_mean_absolute_error"),
    )
    model.fit(x_ht, y_ht)
    return pd.Series(
        model.predict(challenge_embed.to_numpy(dtype=np.float32)).astype(np.float32),
        index=challenge_embed.index.astype(int),
        name="pred_htchem",
    )


def feature_name(idx: int, base_dim: int) -> str:
    if idx == base_dim:
        return "pred_htchem"
    if idx == base_dim - 2:
        return "log2fc_8p25_pred"
    if idx == base_dim - 1:
        return "log2fc_33_pred"
    if idx < 300:
        return f"chemeleon_{idx:03d}"
    return f"base_{idx:04d}"


def feature_family(idx: int, base_dim: int) -> str:
    if idx == base_dim:
        return "pred_htchem"
    if idx >= base_dim - 2:
        return "log2fc_pred"
    if idx < 300:
        return "chemeleon"
    return "2d_boltz"


def sanitize_train_test(
    x_train: np.ndarray, x_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    col_mean = np.nanmean(x_train, axis=0)
    col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
    x_train = np.where(np.isfinite(x_train), x_train, col_mean).astype(np.float32)
    x_test = np.where(np.isfinite(x_test), x_test, col_mean).astype(np.float32)
    return x_train, x_test


def fit_gain(x: np.ndarray, y: np.ndarray, train_idx: np.ndarray | None, seed: int):
    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=10,
        random_state=seed,
        verbose=-1,
    )
    if train_idx is None:
        model.fit(x, y)
    else:
        model.fit(x[train_idx], y[train_idx])
    booster = model.booster_
    gain = booster.feature_importance(importance_type="gain")
    split = booster.feature_importance(importance_type="split")
    order = np.argsort(-gain)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(order) + 1)
    return gain, split, ranks, order


def target_rows(
    gain: np.ndarray,
    split: np.ndarray,
    ranks: np.ndarray,
    base_dim: int,
    k: int,
    fold: str,
) -> list[dict[str, object]]:
    total_gain = float(gain.sum())
    rows = []
    for idx in [base_dim, base_dim - 2, base_dim - 1]:
        rows.append(
            {
                "fold": fold,
                "feature": feature_name(idx, base_dim),
                "family": feature_family(idx, base_dim),
                "index": idx,
                "rank_by_gain": int(ranks[idx]),
                "selected_topk": bool(ranks[idx] <= k),
                "gain": float(gain[idx]),
                "gain_share_pct": float(gain[idx] / total_gain * 100.0)
                if total_gain > 0
                else 0.0,
                "split": int(split[idx]),
            }
        )
    return rows


def selected_feature_rows(
    gain: np.ndarray,
    split: np.ndarray,
    order: np.ndarray,
    base_dim: int,
    k: int,
    fold: str,
) -> pd.DataFrame:
    selected = order[:k]
    total_gain = float(gain.sum())
    return pd.DataFrame(
        [
            {
                "fold": fold,
                "rank_by_gain": rank,
                "feature": feature_name(int(idx), base_dim),
                "family": feature_family(int(idx), base_dim),
                "index": int(idx),
                "gain": float(gain[idx]),
                "gain_share_pct": float(gain[idx] / total_gain * 100.0)
                if total_gain > 0
                else 0.0,
                "split": int(split[idx]),
            }
            for rank, idx in enumerate(selected, start=1)
        ]
    )


def write_doc(target_summary: pd.DataFrame, family_summary: pd.DataFrame) -> None:
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    text = f"""# Phase 2 HTChem top500 gain probe

Purpose: check whether `pred_htchem` should be considered as an extra scalar in the existing LGBM-gain top500 feature selection path.

## Target Feature Ranks

{target_summary.to_markdown(index=False, floatfmt=".4f")}

## Selected Top500 Family Gain

{family_summary.to_markdown(index=False, floatfmt=".4f")}

## Read

If `pred_htchem` repeatedly lands inside top500 with non-trivial gain, the next step is a proper TabPFN top500 SWAP-style run. If it is outside top500 or only barely selected, it is better kept as a diagnostic/map axis rather than promoted into the high-weight top500 member.
"""
    DOC_PATH.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature", default="cheme_2d_full_boltz_log2fc_pred_seed10ens"
    )
    parser.add_argument("--K", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train = load_train()
    test = load_test()
    x_train_base, x_test_base = run_train.load_features(args.feature, train, test)
    pred_htchem = pred_htchem_for_challenge()

    train_ht = pred_htchem.reindex(train["compound_id"].astype(int)).to_numpy(
        dtype=np.float32
    )[:, None]
    test_ht = pred_htchem.reindex(test["compound_id"].astype(int)).to_numpy(
        dtype=np.float32
    )[:, None]
    x_train = np.concatenate([x_train_base, train_ht], axis=1)
    x_test = np.concatenate([x_test_base, test_ht], axis=1)
    x_train, x_test = sanitize_train_test(x_train, x_test)
    y = train["pec50"].to_numpy(dtype=np.float32)
    base_dim = x_train_base.shape[1]

    target_rows_all: list[dict[str, object]] = []
    selected_frames: list[pd.DataFrame] = []

    gain, split, ranks, order = fit_gain(x_train, y, train_idx=None, seed=args.seed)
    target_rows_all.extend(target_rows(gain, split, ranks, base_dim, args.K, "full"))
    selected_frames.append(
        selected_feature_rows(gain, split, order, base_dim, args.K, "full")
    )

    folds = umap_split_indices(
        train["smiles"].tolist(), n_splits=5, n_clusters=50, seed=args.seed
    )
    for fold_idx, (tr_idx, _) in enumerate(folds):
        gain, split, ranks, order = fit_gain(
            x_train, y, train_idx=tr_idx, seed=args.seed
        )
        fold = f"fold{fold_idx}"
        target_rows_all.extend(target_rows(gain, split, ranks, base_dim, args.K, fold))
        selected_frames.append(
            selected_feature_rows(gain, split, order, base_dim, args.K, fold)
        )

    target = pd.DataFrame(target_rows_all)
    selected = pd.concat(selected_frames, ignore_index=True)
    target.to_csv(OUT_DIR / "target_feature_gain_ranks.csv", index=False)
    selected.to_csv(OUT_DIR / "selected_top500_features.csv", index=False)

    fold_only = target[target["fold"].ne("full")].copy()
    target_summary = (
        fold_only.groupby(["feature", "family"], as_index=False)
        .agg(
            selected_folds=("selected_topk", "sum"),
            mean_rank=("rank_by_gain", "mean"),
            min_rank=("rank_by_gain", "min"),
            max_rank=("rank_by_gain", "max"),
            mean_gain_share_pct=("gain_share_pct", "mean"),
            mean_split=("split", "mean"),
        )
        .sort_values("mean_rank")
    )
    full_target = target[target["fold"].eq("full")][
        ["feature", "rank_by_gain", "selected_topk", "gain_share_pct", "split"]
    ].rename(
        columns={
            "rank_by_gain": "full_rank",
            "selected_topk": "full_selected",
            "gain_share_pct": "full_gain_share_pct",
            "split": "full_split",
        }
    )
    target_summary = target_summary.merge(full_target, on="feature", how="left")
    target_summary.to_csv(OUT_DIR / "target_feature_gain_summary.csv", index=False)

    family_summary = (
        selected.groupby(["fold", "family"], as_index=False)
        .agg(
            n_selected=("feature", "size"),
            gain_sum=("gain", "sum"),
            gain_share_pct=("gain_share_pct", "sum"),
            split_sum=("split", "sum"),
        )
        .sort_values(["fold", "gain_share_pct"], ascending=[True, False])
    )
    family_summary.to_csv(OUT_DIR / "selected_top500_family_gain.csv", index=False)

    write_doc(target_summary, family_summary)
    print(f"Wrote outputs to {OUT_DIR}")
    print(f"Wrote doc to {DOC_PATH}")
    print("\nTarget feature summary:")
    print(target_summary.to_string(index=False))
    print("\nFull-train target ranks:")
    print(target[target["fold"].eq("full")].to_string(index=False))


if __name__ == "__main__":
    main()

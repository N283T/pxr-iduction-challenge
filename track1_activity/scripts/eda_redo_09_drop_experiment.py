#!/usr/bin/env -S pixi run python
"""Light before/after single-model experiment to measure drop-list impact.

For each drop configuration, train LightGBM on ChemBERTa-77M-MLM
embeddings with the standard UMAP-split 5-fold CV used elsewhere in
this project. Compare OOF RAE / MAE vs the baseline (no drops).

Configurations:
  - baseline         : no drops
  - drop_big_tail    : n_out >= 5 (32 compounds)
  - drop_small_tail  : HA <= 10 (102 compounds)
  - drop_union       : big_tail OR small_tail (126 compounds)
  - drop_n_out_ge_3  : more aggressive -- 79 compounds
  - drop_n_out_ge_4  : mid threshold    -- 53 compounds

ChemBERTa was chosen because the embeddings are already materialised in
`compound_chemberta` and LightGBM on 384 features is fast enough to run
all six configs in a few minutes.

Outputs:
  - eda_redo_09_drop_experiment.png     - bar chart of RAE per config
  - 09_drop_experiment.parquet          - per-config CV metrics
"""

from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.joinpath("src")))

from data import load_train_chemberta  # noqa: E402
from evaluate import compute_metrics  # noqa: E402
from splits import umap_split_indices  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FIG_DIR = REPO_ROOT.joinpath("docs", "figures")
DATA_DIR = REPO_ROOT.joinpath("data", "eda_redo")

N_SPLITS = 5
SEED = 42

LGBM_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_data_in_leaf": 20,
    "n_estimators": 500,
    "verbose": -1,
    "seed": SEED,
}


def _run_cv(
    X: np.ndarray,
    y: np.ndarray,
    smiles: list[str],
    label: str,
) -> dict:
    """Train LightGBM across UMAP-split folds, return OOF metrics + gap."""
    splits = umap_split_indices(smiles, n_splits=N_SPLITS, seed=SEED)
    oof = np.full_like(y, np.nan, dtype=float)
    fold_train_rae = []
    fold_val_rae = []
    for fi, (tr_idx, va_idx) in enumerate(splits):
        model = lgb.LGBMRegressor(**LGBM_PARAMS)
        model.fit(
            X[tr_idx],
            y[tr_idx],
            eval_set=[(X[va_idx], y[va_idx])],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        tr_pred = model.predict(X[tr_idx])
        va_pred = model.predict(X[va_idx])
        oof[va_idx] = va_pred
        tr_m = compute_metrics(y[tr_idx], tr_pred)
        va_m = compute_metrics(y[va_idx], va_pred)
        fold_train_rae.append(tr_m["RAE"])
        fold_val_rae.append(va_m["RAE"])
        print(
            f"  [{label}] fold {fi}: train RAE {tr_m['RAE']:.4f}"
            f"  val RAE {va_m['RAE']:.4f}  val MAE {va_m['MAE']:.4f}"
        )
    oof_metrics = compute_metrics(y, oof)
    return {
        "config": label,
        "n_train": len(y),
        "oof_rae": oof_metrics["RAE"],
        "oof_mae": oof_metrics["MAE"],
        "oof_r2": oof_metrics["R2"],
        "oof_spearman": oof_metrics["Spearman_R"],
        "val_rae_mean": float(np.mean(fold_val_rae)),
        "val_rae_std": float(np.std(fold_val_rae)),
        "train_rae_mean": float(np.mean(fold_train_rae)),
        "gap": float(np.mean(fold_val_rae) - np.mean(fold_train_rae)),
    }


def main() -> None:
    # ------------------------------------------------------------------
    # Load baseline data
    # ------------------------------------------------------------------
    feats_df, y_series = load_train_chemberta()
    # load_chemberta returns DataFrame with compound_id as index
    compound_ids = feats_df.index.to_numpy()
    X = feats_df.to_numpy(dtype=np.float32)
    y = y_series.to_numpy(dtype=float)
    print(f"[09] baseline: N={len(y)}, features={X.shape[1]}")

    # We also need SMILES for the UMAP split. Pull from master.
    from eda_redo import load_master  # lazy import

    master = load_master()
    smiles_map = dict(zip(master["compound_id"], master["smiles"]))
    smiles = [smiles_map[int(c)] for c in compound_ids]

    # ------------------------------------------------------------------
    # Build drop configurations
    # ------------------------------------------------------------------
    drops = pd.read_parquet(DATA_DIR.joinpath("07_drop_candidates.parquet"))
    scorecard = pd.read_parquet(DATA_DIR.joinpath("06_outlier_scorecard.parquet"))

    big_ids = set(scorecard[scorecard["n_out"] >= 5]["compound_id"].tolist())
    small_ids = set(
        drops[drops["drop_reason"].isin(["small_tail", "both"])]["compound_id"]
    )
    union_ids = big_ids | small_ids
    n3_ids = set(scorecard[scorecard["n_out"] >= 3]["compound_id"].tolist())
    n4_ids = set(scorecard[scorecard["n_out"] >= 4]["compound_id"].tolist())

    configs = [
        ("baseline", set()),
        ("drop_big_tail", big_ids),
        ("drop_small_tail", small_ids),
        ("drop_union", union_ids),
        ("drop_n_out_ge_3", n3_ids),
        ("drop_n_out_ge_4", n4_ids),
    ]

    # ------------------------------------------------------------------
    # Run each config
    # ------------------------------------------------------------------
    results = []
    for label, drop_set in configs:
        mask = ~np.isin(compound_ids, list(drop_set))
        Xc = X[mask]
        yc = y[mask]
        sm = [s for s, m in zip(smiles, mask) if m]
        print(f"\n[09] --- {label} (dropped {len(drop_set):>4d}, N={len(yc):>5d}) ---")
        r = _run_cv(Xc, yc, sm, label)
        r["n_dropped"] = len(drop_set)
        results.append(r)

    df = pd.DataFrame(results)
    print()
    print("=" * 80)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DATA_DIR.joinpath("09_drop_experiment.parquet"), index=False)

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # OOF RAE
    ax = axes[0]
    xs = np.arange(len(df))
    base_rae = df[df["config"] == "baseline"]["oof_rae"].iloc[0]
    colors = [
        "grey" if r == base_rae else ("C2" if r < base_rae else "C3")
        for r in df["oof_rae"]
    ]
    ax.bar(xs, df["oof_rae"], color=colors, alpha=0.85)
    ax.axhline(
        base_rae,
        color="black",
        linestyle=":",
        linewidth=1,
        label=f"baseline RAE = {base_rae:.4f}",
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(df["config"], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("OOF RAE (lower = better)")
    ax.set_title("OOF RAE by drop configuration")
    for i, v in enumerate(df["oof_rae"]):
        ax.text(i, v + 0.001, f"{v:.4f}", ha="center", fontsize=8)
    ax.legend()

    # Gap (val - train)
    ax = axes[1]
    ax.bar(xs, df["gap"], color="C0", alpha=0.85)
    ax.set_xticks(xs)
    ax.set_xticklabels(df["config"], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("val RAE - train RAE  (generalisation gap)")
    ax.set_title("Train-val gap by drop configuration")
    for i, v in enumerate(df["gap"]):
        ax.text(i, v + 0.001, f"{v:.4f}", ha="center", fontsize=8)

    fig.suptitle(
        "Issue #52 09: drop experiment (ChemBERTa -> LightGBM, UMAP 5-fold)",
        fontsize=13,
    )
    fig.tight_layout()
    fig_path = FIG_DIR.joinpath("eda_redo_09_drop_experiment.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[09] wrote {fig_path}")


if __name__ == "__main__":
    main()

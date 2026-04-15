#!/usr/bin/env -S pixi run python
"""Drop-list before/after experiment across multiple feature types.

Runs 3 independent feature representations x 6 drop configurations. The
multi-feature pass matters because the drop list was defined using 11
RDKit descriptors, so a model fed Mordred (which overlaps with those
descriptors) could show an artificial bias; we want to see if the
Morgan-FP picture agrees.

Features:
  - chemberta   : ChemBERTa-77M-MLM embeddings (compound_chemberta, 384d).
                  Weak ensemble member, sanity check only.
  - morgan_r2   : Morgan fingerprints r=2, 2048 bits (computed on the fly).
                  Independent of the descriptor definitions used to build
                  the drop list - cleanest test.
  - mordred     : Mordred 2D descriptors (compound_mordred, ~1460d).
                  Overlaps with the outlier descriptors so there is some
                  "circular" risk, but the bulk of the feature space is
                  independent.

For each (feature, drop) pair:
  - LightGBM default params
  - UMAP 5-fold CV (seed=42)
  - OOF RAE / MAE / R2 / Spearman + train-val gap

Outputs:
  - eda_redo_09_drop_experiment.png     - 3-panel bar chart (one per feature)
  - 09_drop_experiment.parquet          - per-config x per-feature metrics
"""

from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.joinpath("src")))

from data import get_engine, load_train_chemberta, load_train_mordred  # noqa: E402
from evaluate import compute_metrics  # noqa: E402
from features import morgan_fp, smiles_to_mols  # noqa: E402
from splits import umap_split_indices  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
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


def _run_cv(X, y, smiles, label):
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
    oof_m = compute_metrics(y, oof)
    return {
        "config": label,
        "n_train": int(len(y)),
        "oof_rae": float(oof_m["RAE"]),
        "oof_mae": float(oof_m["MAE"]),
        "oof_r2": float(oof_m["R2"]),
        "oof_spearman": float(oof_m["Spearman_R"]),
        "val_rae_mean": float(np.mean(fold_val_rae)),
        "val_rae_std": float(np.std(fold_val_rae)),
        "train_rae_mean": float(np.mean(fold_train_rae)),
        "gap": float(np.mean(fold_val_rae) - np.mean(fold_train_rae)),
    }


def _prepare_feature(name: str):
    """Return (compound_ids: np.ndarray, X: np.ndarray, y: np.ndarray, smiles: list[str])."""
    if name == "chemberta":
        emb, y_s = load_train_chemberta()
        cids = emb.index.to_numpy()
        X = emb.to_numpy(dtype=np.float32)
        y = y_s.to_numpy(dtype=float)
    elif name == "morgan_r2":
        train_df = pd.read_sql(
            """SELECT t.compound_id, c.std_smiles AS smiles, t.pec50
               FROM train_activity t
               JOIN compounds c ON c.id = t.compound_id
               ORDER BY t.id""",
            get_engine(),
        )
        cids = train_df["compound_id"].to_numpy()
        mols = smiles_to_mols(train_df["smiles"])
        X = morgan_fp(mols, radius=2, n_bits=2048).astype(np.float32)
        y = train_df["pec50"].to_numpy(dtype=float)
    elif name == "mordred":
        mord_df, y_s = load_train_mordred()
        cids = mord_df.index.to_numpy()
        X = mord_df.to_numpy(dtype=np.float32)
        y = y_s.to_numpy(dtype=float)
    else:
        raise ValueError(name)
    # Align SMILES via master
    from eda_redo import load_master

    master = load_master()
    smap = dict(zip(master["compound_id"], master["smiles"]))
    smiles = [smap[int(c)] for c in cids]
    return cids, X, y, smiles


def _load_drop_sets() -> dict[str, set[int]]:
    drops = pd.read_parquet(DATA_DIR.joinpath("07_drop_candidates.parquet"))
    scorecard = pd.read_parquet(DATA_DIR.joinpath("06_outlier_scorecard.parquet"))
    big = set(scorecard[scorecard["n_out"] >= 5]["compound_id"].tolist())
    small = set(drops[drops["drop_reason"].isin(["small_tail", "both"])]["compound_id"])
    return {
        "baseline": set(),
        "drop_big_tail": big,
        "drop_small_tail": small,
        "drop_union": big | small,
        "drop_n_out_ge_3": set(
            scorecard[scorecard["n_out"] >= 3]["compound_id"].tolist()
        ),
        "drop_n_out_ge_4": set(
            scorecard[scorecard["n_out"] >= 4]["compound_id"].tolist()
        ),
    }


def main() -> None:
    drop_sets = _load_drop_sets()

    all_rows = []
    for feat_name in ["morgan_r2", "chemberta", "mordred"]:
        print(f"\n\n====================  feature = {feat_name}  ====================")
        cids, X, y, smiles = _prepare_feature(feat_name)
        print(f"[09] feature={feat_name}  N={len(y)}  dim={X.shape[1]}")
        for cfg_name, drop_ids in drop_sets.items():
            mask = ~np.isin(cids, list(drop_ids))
            Xc = X[mask]
            yc = y[mask]
            sm = [s for s, m in zip(smiles, mask) if m]
            print(
                f"\n[09] --- {feat_name} / {cfg_name} (dropped {len(drop_ids):>4d}, N={len(yc):>5d}) ---"
            )
            r = _run_cv(Xc, yc, sm, cfg_name)
            r["feature"] = feat_name
            r["n_dropped"] = int(len(drop_ids))
            all_rows.append(r)
            print(
                f"      OOF RAE {r['oof_rae']:.4f}  val_std {r['val_rae_std']:.4f}"
                f"  gap {r['gap']:.4f}"
            )

    df = pd.DataFrame(all_rows)
    print()
    print("=" * 90)
    pivot = df.pivot_table(index="config", columns="feature", values="oof_rae")
    print(pivot.to_string(float_format=lambda x: f"{x:.4f}"))

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DATA_DIR.joinpath("09_drop_experiment.parquet"), index=False)

    # ------------------------------------------------------------------
    # Figure: 1 row x 3 feature panels (OOF RAE bars vs baseline line)
    # ------------------------------------------------------------------
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    features_order = ["morgan_r2", "chemberta", "mordred"]
    fig, axes = plt.subplots(
        1, len(features_order), figsize=(6 * len(features_order), 5.5), sharey=False
    )
    configs = list(drop_sets.keys())
    for ax, feat in zip(axes, features_order):
        sub = df[df["feature"] == feat].set_index("config").loc[configs]
        base = sub.loc["baseline", "oof_rae"]
        colors = [
            "grey" if c == "baseline" else ("C2" if r < base else "C3")
            for c, r in zip(sub.index, sub["oof_rae"])
        ]
        xs = np.arange(len(sub))
        ax.bar(xs, sub["oof_rae"], color=colors, alpha=0.85)
        ax.axhline(
            base,
            color="black",
            linestyle=":",
            linewidth=1,
            label=f"baseline {base:.4f}",
        )
        for i, v in enumerate(sub["oof_rae"]):
            ax.text(i, v + 0.001, f"{v:.4f}", ha="center", fontsize=8)
        ax.set_xticks(xs)
        ax.set_xticklabels(sub.index, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("OOF RAE")
        ax.set_title(f"feature = {feat}")
        ax.legend(fontsize=8)
    fig.suptitle(
        "Issue #52 09: drop experiment across 3 feature types (UMAP 5-fold, LightGBM)",
        fontsize=13,
    )
    fig.tight_layout()
    fig_path = FIG_DIR.joinpath("eda_redo_09_drop_experiment.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[09] wrote {fig_path}")


if __name__ == "__main__":
    main()

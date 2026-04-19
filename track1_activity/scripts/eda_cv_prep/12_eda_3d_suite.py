"""EDA for the 4077-dim 3D feature suite computed on Boltz-2 poses.

Answers:
  1. Volume and quality per family (coverage, variance, correlation to pEC50)
  2. Intra-family multicollinearity
  3. Combined LGBM gain importance (is the 3D signal concentrated in a few
     families, or spread out?)
  4. Overlap with 2D feature importance -- i.e., does the 3D signal
     merely restate what Mordred already saw, or is it orthogonal?

Output:
  - markdown-style summary to stdout
  - data/eda_cv_prep/eda_3d_*.csv artifacts
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import psycopg2

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
from data import DB_PARAMS  # noqa: E402

OUT_DIR = REPO_ROOT.joinpath("data", "eda_cv_prep")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_train_ids_and_target() -> tuple[list[int], np.ndarray]:
    conn = psycopg2.connect(**DB_PARAMS)
    df = pd.read_sql("SELECT compound_id, pec50 FROM train_activity ORDER BY id", conn)
    conn.close()
    return df["compound_id"].tolist(), df["pec50"].to_numpy(dtype=np.float64)


def load_3d_blocks(train_ids: list[int]) -> dict[str, tuple[np.ndarray, list[str]]]:
    """Return {family_name: (feature_matrix[n_train, dim], column_names)}."""
    conn = psycopg2.connect(**DB_PARAMS)
    blocks: dict[str, tuple[np.ndarray, list[str]]] = {}

    # Scalar 3D shape (11 columns)
    scalar_cols = [
        "asphericity",
        "eccentricity",
        "inertial_shape_factor",
        "npr1",
        "npr2",
        "pmi1",
        "pmi2",
        "pmi3",
        "radius_of_gyration",
        "spherocity_index",
        "pbf",
    ]
    df_scalar = pd.read_sql(
        f"SELECT compound_id, {', '.join(scalar_cols)} FROM compound_boltz2_desc3d",
        conn,
    ).set_index("compound_id")
    X_scalar = df_scalar.reindex(train_ids).to_numpy(dtype=np.float32)
    X_scalar = np.nan_to_num(X_scalar, nan=0.0, posinf=0.0, neginf=0.0)
    blocks["scalar3d"] = (X_scalar, [f"scalar3d:{c}" for c in scalar_cols])

    # Vector 3D (7 JSONB columns)
    vec_families = [
        ("autocorr3d", 80),
        ("getaway", 273),
        ("morse", 224),
        ("rdf", 210),
        ("whim", 114),
        ("usr", 12),
        ("usrcat", 60),
    ]
    df_vec = (
        pd.read_sql(
            "SELECT compound_id, autocorr3d, getaway, morse, rdf, whim, usr, usrcat "
            "FROM compound_boltz2_desc3d_vector",
            conn,
        )
        .set_index("compound_id")
        .reindex(train_ids)
    )
    for fam, expected_dim in vec_families:
        rows = []
        for jsn in df_vec[fam]:
            if jsn is None or (isinstance(jsn, float) and np.isnan(jsn)):
                # Missing (e.g. Auranofin, id=1657, Boltz preprocess-failed)
                rows.append(np.zeros(expected_dim, dtype=np.float32))
            else:
                arr = np.asarray(jsn, dtype=np.float64)
                arr = np.where(np.isnan(arr) | np.isinf(arr), 0.0, arr)
                rows.append(arr.astype(np.float32))
        X = np.stack(rows, axis=0)
        cols = [f"{fam}:{i}" for i in range(X.shape[1])]
        blocks[fam] = (X, cols)

    # skfp (E3FP 1024 + ElectroShape 15 + Pharmacophore3D 2048)
    df_skfp = (
        pd.read_sql(
            "SELECT compound_id, e3fp, electroshape, pharmacophore3d "
            "FROM compound_boltz2_skfp3d",
            conn,
        )
        .set_index("compound_id")
        .reindex(train_ids)
    )
    for fam, expected_dim in [
        ("e3fp", 1024),
        ("electroshape", 15),
        ("pharmacophore3d", 2048),
    ]:
        rows = []
        for jsn in df_skfp[fam]:
            if jsn is None or (isinstance(jsn, float) and np.isnan(jsn)):
                rows.append(np.zeros(expected_dim, dtype=np.float32))
            else:
                arr = np.asarray(jsn, dtype=np.float64)
                arr = np.where(np.isnan(arr) | np.isinf(arr), 0.0, arr)
                rows.append(arr.astype(np.float32))
        X = np.stack(rows, axis=0)
        cols = [f"{fam}:{i}" for i in range(X.shape[1])]
        blocks[fam] = (X, cols)

    # Pose-specific Jazzy (6 columns)
    jz_cols = ["sdc", "sdx", "sa", "dga", "dgp", "dgtot"]
    df_jz = pd.read_sql(
        f"SELECT compound_id, {', '.join(jz_cols)} FROM compound_boltz2_jazzy",
        conn,
    ).set_index("compound_id")
    X_jz = df_jz.reindex(train_ids).to_numpy(dtype=np.float32)
    X_jz = np.nan_to_num(X_jz, nan=0.0, posinf=0.0, neginf=0.0)
    blocks["pose_jazzy"] = (X_jz, [f"pose_jazzy:{c}" for c in jz_cols])

    conn.close()
    return blocks


def per_family_stats(
    blocks: dict[str, tuple[np.ndarray, list[str]]],
    y: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for fam, (X, cols) in blocks.items():
        n_total = X.shape[1]
        std = X.std(axis=0)
        n_nonzero_var = int((std > 1e-9).sum())

        # Correlation to y (for non-constant columns)
        active = std > 1e-9
        if active.any():
            Xa = X[:, active]
            Xa_mean = Xa.mean(axis=0)
            Xa_std = Xa.std(axis=0)
            y_std = y.std()
            if y_std > 0 and (Xa_std > 0).all():
                corrs = ((Xa - Xa_mean) * (y - y.mean())[:, None]).mean(axis=0) / (
                    Xa_std * y_std
                )
            else:
                corrs = np.zeros(Xa.shape[1])
        else:
            corrs = np.zeros(0)

        n_corr_gt_01 = int((np.abs(corrs) > 0.1).sum())
        n_corr_gt_02 = int((np.abs(corrs) > 0.2).sum())
        max_abs_corr = float(np.abs(corrs).max()) if corrs.size else 0.0

        # Pairwise |Pearson| among active features (capped to 500 for cost)
        n_pair = min(active.sum(), 500)
        if n_pair > 1:
            idx = np.where(active)[0][:n_pair]
            Xs = X[:, idx]
            c = np.corrcoef(Xs.T)
            np.fill_diagonal(c, 0.0)
            mean_abs_inter = float(np.abs(c).mean())
        else:
            mean_abs_inter = float("nan")

        rows.append(
            {
                "family": fam,
                "dim": n_total,
                "non_zero_var": n_nonzero_var,
                "n_|corr_y|>0.1": n_corr_gt_01,
                "n_|corr_y|>0.2": n_corr_gt_02,
                "max_|corr_y|": round(max_abs_corr, 3),
                "mean_|inter_corr|": round(mean_abs_inter, 3),
            }
        )
    return pd.DataFrame(rows)


def combined_lgbm_importance(
    blocks: dict[str, tuple[np.ndarray, list[str]]],
    y: np.ndarray,
) -> pd.DataFrame:
    Xs = [X for X, _ in blocks.values()]
    names = [n for _, cols in blocks.values() for n in cols]
    X = np.concatenate(Xs, axis=1)

    rng = np.random.RandomState(42)
    perm = rng.permutation(len(y))
    split = int(0.8 * len(y))
    tr, va = perm[:split], perm[split:]

    model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=63,
        random_state=42,
        verbose=-1,
    )
    model.fit(
        X[tr],
        y[tr],
        eval_set=[(X[va], y[va])],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    gain = model.booster_.feature_importance(importance_type="gain")
    split_imp = model.booster_.feature_importance(importance_type="split")
    df = (
        pd.DataFrame({"feature": names, "gain": gain, "split": split_imp})
        .sort_values("gain", ascending=False)
        .reset_index(drop=True)
    )
    total = df["gain"].sum()
    df["gain_frac"] = df["gain"] / max(total, 1e-12)
    df["src"] = df["feature"].str.split(":", n=1).str[0]
    return df


def main() -> None:
    t0 = time.time()
    print("Loading train ids + pec50...")
    train_ids, y = load_train_ids_and_target()
    print(f"  n_train = {len(train_ids)}")

    print("Loading 3D blocks from 4 tables...")
    blocks = load_3d_blocks(train_ids)
    for fam, (X, _) in blocks.items():
        print(f"  {fam:<16} shape={X.shape}")
    print(f"  load time: {time.time() - t0:.1f}s")

    print("\n=== PER-FAMILY STATS ===")
    stats = per_family_stats(blocks, y)
    print(stats.to_string(index=False))
    stats.to_csv(OUT_DIR.joinpath("eda_3d_per_family.csv"), index=False)

    print("\n=== LGBM combined gain importance ===")
    imp = combined_lgbm_importance(blocks, y)
    imp.to_csv(OUT_DIR.joinpath("eda_3d_importance.csv"), index=False)

    total_feats = len(imp)
    used = int((imp["gain"] > 0).sum())
    print(f"  total features: {total_feats}")
    print(f"  used (gain>0):  {used}  ({100 * used / total_feats:.1f}%)")
    print(f"  zero gain:      {total_feats - used}")

    for pct in (0.5, 0.8, 0.9, 0.95):
        cum = imp["gain_frac"].cumsum()
        n = int((cum <= pct).sum()) + 1
        print(f"  features at {int(pct * 100)}% cum gain: {n}")

    print("\n  Top 15 by gain:")
    print(
        imp.head(15)[["feature", "gain_frac", "split"]].to_string(
            index=False, formatters={"gain_frac": "{:.4f}".format}
        )
    )

    print("\n  Gain share per family:")
    src_stats = (
        imp.groupby("src")
        .agg(
            total=("feature", "count"),
            used=("gain", lambda s: int((s > 0).sum())),
            gain_share=("gain_frac", "sum"),
        )
        .sort_values("gain_share", ascending=False)
    )
    print(src_stats.to_string())
    src_stats.to_csv(OUT_DIR.joinpath("eda_3d_gain_by_family.csv"))

    print(f"\nTotal elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

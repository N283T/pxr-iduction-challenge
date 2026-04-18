"""Compare Morgan-FP vs Mordred-descriptor UMAP splits on three
axes that could explain why the Mordred-space split produced a
*lower* OOF RAE than Morgan-space (PR #70):

  1. Cluster size / fold size distribution
  2. Val y (pEC50) dispersion per fold
  3. Val->train NN Tanimoto (Morgan) per fold -- measures "did we
     actually hold out substructurally novel compounds?"

Output: markdown-style table printed to stdout + saved as CSV under
data/eda_cv_prep/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
from data import DB_PARAMS, load_train_mordred  # noqa: E402
from splits import umap_split_indices  # noqa: E402

OUT_DIR = REPO_ROOT.joinpath("data", "eda_cv_prep")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_train_with_ids() -> pd.DataFrame:
    conn = psycopg2.connect(**DB_PARAMS)
    try:
        df = pd.read_sql(
            """
            SELECT t.compound_id, c.std_smiles AS smiles, t.pec50
            FROM train_activity t
            JOIN compounds c ON t.compound_id = c.id
            ORDER BY t.id
            """,
            conn,
        )
    finally:
        conn.close()
    return df


def compute_nn_tanimoto(query_fps: list, ref_fps: list) -> np.ndarray:
    out = np.empty(len(query_fps), dtype=np.float64)
    for i, qf in enumerate(query_fps):
        sims = DataStructs.BulkTanimotoSimilarity(qf, ref_fps)
        out[i] = max(sims) if sims else np.nan
    return out


def per_fold_stats(
    splits: list[tuple[np.ndarray, np.ndarray]],
    y: np.ndarray,
    all_fps: list,
    label: str,
) -> pd.DataFrame:
    rows = []
    for k, (tr, va) in enumerate(splits):
        ref_fps = [all_fps[i] for i in tr]
        q_fps = [all_fps[i] for i in va]
        nn = compute_nn_tanimoto(q_fps, ref_fps)
        va_y = y[va]
        median_y = float(np.median(va_y))
        rows.append(
            {
                "split": label,
                "fold": k,
                "train_n": len(tr),
                "val_n": len(va),
                "val_y_mean": float(va_y.mean()),
                "val_y_std": float(va_y.std()),
                "val_y_abs_dev_median": float(np.mean(np.abs(va_y - median_y))),
                "val_nn_tanimoto_mean": float(nn.mean()),
                "val_nn_tanimoto_p25": float(np.quantile(nn, 0.25)),
                "val_nn_tanimoto_median": float(np.median(nn)),
                "val_nn_tanimoto_p75": float(np.quantile(nn, 0.75)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    df = load_train_with_ids().reset_index(drop=True)
    smiles = df["smiles"].tolist()
    y = df["pec50"].to_numpy(dtype=np.float64)
    n = len(df)
    print(f"Loaded {n} train compounds")

    print("Building Morgan fingerprints (for NN Tanimoto comparison)...")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    if any(m is None for m in mols):
        raise ValueError("Invalid SMILES")
    all_fps = [gen.GetFingerprint(m) for m in mols]

    print("Morgan-space UMAP split (canonical, seed=42)...")
    morgan_splits = umap_split_indices(
        smiles_list=smiles, n_splits=5, n_clusters=50, seed=42
    )

    print("Mordred-space UMAP split (seed=42)...")
    mordred_train, _ = load_train_mordred()
    train_ids = df["compound_id"].tolist()
    missing = set(train_ids) - set(mordred_train.index)
    if missing:
        raise ValueError(f"Mordred missing for {len(missing)} train compounds")
    mordred_mat = mordred_train.loc[train_ids].to_numpy(dtype=np.float32)
    mordred_mat = np.nan_to_num(mordred_mat, nan=0.0, posinf=0.0, neginf=0.0)
    mordred_splits = umap_split_indices(
        smiles_list=smiles,
        n_splits=5,
        n_clusters=50,
        seed=42,
        features=mordred_mat,
        metric="cosine",
    )

    print("Morgan+Mordred combined UMAP split (seed=42)...")
    # Use generator's numpy helper rather than coercing ExplicitBitVect.
    morgan_bin = np.array(
        [gen.GetFingerprintAsNumPy(m) for m in mols], dtype=np.float32
    )
    mordred_mean = mordred_mat.mean(axis=0, keepdims=True)
    mordred_std = mordred_mat.std(axis=0, keepdims=True)
    mordred_std[mordred_std == 0] = 1.0
    mordred_z = (mordred_mat - mordred_mean) / mordred_std
    combined_mat = np.concatenate([morgan_bin, mordred_z], axis=1).astype(np.float32)
    combined_splits = umap_split_indices(
        smiles_list=smiles,
        n_splits=5,
        n_clusters=50,
        seed=42,
        features=combined_mat,
        metric="cosine",
    )

    print("Computing per-fold stats...")
    stats_morgan = per_fold_stats(morgan_splits, y, all_fps, "morgan")
    stats_mordred = per_fold_stats(mordred_splits, y, all_fps, "mordred")
    stats_combined = per_fold_stats(combined_splits, y, all_fps, "morgan_mordred")
    df_stats = pd.concat(
        [stats_morgan, stats_mordred, stats_combined], ignore_index=True
    )
    df_stats.to_csv(OUT_DIR.joinpath("split_space_compare.csv"), index=False)

    print("\n=== Per-fold stats ===")
    cols_display = [
        "split",
        "fold",
        "val_n",
        "val_y_mean",
        "val_y_std",
        "val_y_abs_dev_median",
        "val_nn_tanimoto_mean",
        "val_nn_tanimoto_median",
    ]
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print(df_stats[cols_display].to_string(index=False))

    print("\n=== Aggregate summary (mean across folds) ===")
    agg = df_stats.groupby("split").agg(
        {
            "val_n": ["mean", "std"],
            "val_y_mean": "mean",
            "val_y_std": "mean",
            "val_y_abs_dev_median": "mean",
            "val_nn_tanimoto_mean": "mean",
            "val_nn_tanimoto_median": "mean",
        }
    )
    print(agg.to_string())

    print("\n=== Interpretation ===")
    print(
        "If val->train NN Tanimoto is HIGHER under Mordred-space split,\n"
        "that means val compounds were substructurally CLOSER to train\n"
        "(easier to predict because similar Morgan neighbours exist in\n"
        "train). This would explain the lower OOF MAE observed in PR #70\n"
        "without invoking any 'harder target space' story.\n"
    )


if __name__ == "__main__":
    main()

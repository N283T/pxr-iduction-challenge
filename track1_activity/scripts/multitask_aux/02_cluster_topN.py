#!/usr/bin/env -S pixi run python
"""Redundancy-aware top-N feature selection for multitask aux targets.

Reads importance_all_desc.csv (from 01_importance_lgbm.py), recomputes
the same feature matrix, then greedy-clusters features whose absolute
Pearson correlation is >= threshold. Cluster representative is the
highest-gain member.

Also reports correlation between each candidate aux target and pEC50 --
features with |r(aux, pec50)| >= 0.85 are flagged as "shortcut risk"
and downweighted (still output, but separate table for user decision).

Outputs:
  reports/multitask_aux/top_aux_candidates.csv  -- ranked reps with
    family, gain, r_pec50, cluster_size, shortcut_flag.
  reports/multitask_aux/cluster_map.csv         -- feature -> cluster_id.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
sys.path.insert(
    0, str(REPO_ROOT.joinpath("track1_activity", "scripts", "multitask_aux"))
)

from data import load_train_smiles_target  # noqa: E402

# Reuse build_matrix from the importance script to avoid drift
importance_mod = __import__("01_importance_lgbm")

REPORT_DIR = REPO_ROOT.joinpath("track1_activity", "reports", "multitask_aux")

# Thresholds
CORR_CLUSTER_THRESHOLD = 0.9  # |r| >= this -> cluster together
SHORTCUT_R_THRESHOLD = 0.85  # |r(aux, pec50)| >= this -> flag as shortcut
TOP_BY_GAIN = 200  # cluster only the top-200 gain features (plenty)
FINAL_TOP_N = 50  # keep up to this many cluster reps

# Weight down families that we deliberately want fewer of
FAMILY_CAP = {
    "mordred": 25,  # Mordred top-25 reps max (still dominates naturally)
    "rdkit": 8,
    "boltz_tier0": 8,  # all 19 ok but cap gives breathing room
    "boltz_tier1": 6,  # 44 confidence aggregates, take top-6 reps
    "d3_mordred3d": 6,
    "d3_getaway": 4,
    "d3_morse": 4,
    "d3_whim": 3,
    "d3_autocorr3d": 3,
    "d3_rdf": 4,
    "d3_usr": 4,
    "d3_usrcat": 6,
    "d3_electroshape": 4,
    "d3_scalar": 4,
    "jazzy_pose": 6,
    "jazzy_self": 3,
}


def greedy_cluster(
    X: np.ndarray,
    feat_names: list[str],
    gain: np.ndarray,
    threshold: float,
) -> tuple[list[int], np.ndarray]:
    """Greedy feature clustering by |Pearson r|.

    Walks features in descending-gain order. If a feature correlates
    >= threshold with an already-selected representative, it joins that
    rep's cluster. Otherwise it becomes a new rep.

    Returns:
        rep_indices: column indices of cluster representatives (sorted
            descending by gain).
        cluster_ids: int array length=X.shape[1], maps each feature to
            the index of its rep in rep_indices. -1 for features not
            considered (outside the top-N by gain).
    """
    order = np.argsort(-gain)
    rep_idx: list[int] = []
    rep_cols: list[np.ndarray] = []
    cluster_ids = np.full(X.shape[1], -1, dtype=np.int64)

    # Standardize for speed (r = dot / n after standardize)
    def _std(col: np.ndarray) -> np.ndarray:
        c = col.astype(np.float64)
        m = c.mean()
        s = c.std()
        if s < 1e-12:
            return np.zeros_like(c)
        return (c - m) / s

    std_reps: list[np.ndarray] = []
    for idx in order:
        col_std = _std(X[:, idx])
        if np.all(col_std == 0):
            continue
        assigned = False
        for r_i, rep_std in enumerate(std_reps):
            r = float(abs(np.dot(col_std, rep_std) / len(col_std)))
            if r >= threshold:
                cluster_ids[idx] = r_i
                assigned = True
                break
        if not assigned:
            cluster_ids[idx] = len(rep_idx)
            rep_idx.append(int(idx))
            rep_cols.append(col_std)
            std_reps.append(col_std)

    return rep_idx, cluster_ids


def main() -> None:
    imp_csv = REPORT_DIR.joinpath("importance_all_desc.csv")
    imp = pd.read_csv(imp_csv)
    print(f"Loaded {len(imp)} feature rows from {imp_csv.name}")

    # Focus on top-N by gain for clustering (cheap + plenty)
    top = imp.head(TOP_BY_GAIN).reset_index(drop=True)
    print(
        f"Clustering top {len(top)} features "
        f"(gain_mean range: {top.gain_mean.min():.1f}..{top.gain_mean.max():.1f})"
    )

    # Rebuild feature matrix (only train, but column ids match importance csv)
    train_df = load_train_smiles_target()
    train_ids = importance_mod.load_compound_ids("train")
    test_ids = importance_mod.load_compound_ids("test")
    X_tr, _, feat_names = importance_mod.build_matrix(train_ids, test_ids)
    assert len(feat_names) == len(imp), (
        f"feat_names ({len(feat_names)}) != importance rows ({len(imp)})"
    )

    # Map top features to column indices
    name_to_idx = {n: i for i, n in enumerate(feat_names)}
    top_cols = [name_to_idx[n] for n in top["feature"]]
    X_top = X_tr[:, top_cols]
    gain_top = top["gain_mean"].to_numpy()

    rep_local, cluster_ids_local = greedy_cluster(
        X_top, top["feature"].tolist(), gain_top, CORR_CLUSTER_THRESHOLD
    )
    rep_cols_global = [top_cols[i] for i in rep_local]
    print(f"  Raw clusters: {len(rep_cols_global)} representatives")

    # Correlation with pEC50 (shortcut detection)
    y = train_df["pec50"].to_numpy(dtype=np.float64)
    y_std = (y - y.mean()) / y.std()
    r_pec50 = np.zeros(len(rep_cols_global))
    cluster_size = np.zeros(len(rep_cols_global), dtype=int)
    for r_i, col_idx in enumerate(rep_cols_global):
        col = X_tr[:, col_idx].astype(np.float64)
        s = col.std()
        if s < 1e-12:
            r_pec50[r_i] = 0.0
        else:
            col_std = (col - col.mean()) / s
            r_pec50[r_i] = float(np.dot(col_std, y_std) / len(col_std))
        cluster_size[r_i] = int((cluster_ids_local == r_i).sum())

    reps_df = pd.DataFrame(
        dict(
            feature=[feat_names[c] for c in rep_cols_global],
            family=[feat_names[c].split(".")[0] for c in rep_cols_global],
            gain_mean=[
                top.loc[rep_local[i], "gain_mean"] for i in range(len(rep_local))
            ],
            nonzero_folds=[
                top.loc[rep_local[i], "nonzero_folds"] for i in range(len(rep_local))
            ],
            r_pec50=r_pec50,
            cluster_size=cluster_size,
        )
    )
    reps_df["shortcut_flag"] = reps_df["r_pec50"].abs() >= SHORTCUT_R_THRESHOLD
    reps_df = reps_df.sort_values("gain_mean", ascending=False).reset_index(drop=True)
    reps_df["overall_rank"] = np.arange(1, len(reps_df) + 1)

    # Apply family caps
    kept: list[int] = []
    fam_count: dict[str, int] = {}
    for i, row in reps_df.iterrows():
        fam = row["family"]
        cap = FAMILY_CAP.get(fam, 5)
        if fam_count.get(fam, 0) >= cap:
            continue
        kept.append(i)
        fam_count[fam] = fam_count.get(fam, 0) + 1
        if len(kept) >= FINAL_TOP_N:
            break

    final_df = reps_df.loc[kept].copy()
    final_df["final_rank"] = np.arange(1, len(final_df) + 1)

    # Cluster map for transparency
    cluster_rows = []
    for local_idx, rep_global in enumerate(rep_cols_global):
        rep_name = feat_names[rep_global]
        member_local = np.where(cluster_ids_local == local_idx)[0]
        for m in member_local:
            cluster_rows.append(
                dict(
                    cluster_rep=rep_name,
                    member=top.loc[m, "feature"],
                    member_gain=top.loc[m, "gain_mean"],
                )
            )
    cluster_map = pd.DataFrame(cluster_rows)

    # Outputs
    top_path = REPORT_DIR.joinpath("top_aux_candidates.csv")
    final_df.to_csv(top_path, index=False)
    cm_path = REPORT_DIR.joinpath("cluster_map.csv")
    cluster_map.to_csv(cm_path, index=False)

    print(f"\n=== Top {len(final_df)} aux candidates (family-capped) ===")
    print(
        final_df[
            [
                "final_rank",
                "feature",
                "family",
                "gain_mean",
                "r_pec50",
                "cluster_size",
                "shortcut_flag",
            ]
        ].to_string(index=False)
    )

    print("\n=== Per-family count in final list ===")
    print(final_df["family"].value_counts().to_string())

    print("\n=== Shortcut-flagged (|r(aux, pec50)| >= 0.85) ===")
    print(
        final_df[final_df.shortcut_flag][
            ["final_rank", "feature", "family", "gain_mean", "r_pec50"]
        ].to_string(index=False)
    )

    print(f"\nSaved: {top_path}")
    print(f"Saved: {cm_path}")


if __name__ == "__main__":
    main()

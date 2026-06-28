#!/usr/bin/env python
"""Build composite pairrank + Boltz-style ChemProp scalar signals."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PAIRRANK_DIR = (
    REPO_ROOT
    / "track1_activity/analysis/phase2_classifier_gate/outputs/pairwise_assay_rank/"
    / "all_pxr_chembl_htchem_single_conc_mpa1500_top64"
)
DEFAULT_ANCHOR = (
    REPO_ROOT / "track1_activity/submissions/ens_id51_top500_potent46_t40_soft_g35.csv"
)
DEFAULT_OUT_DIR = (
    REPO_ROOT
    / "track1_activity/analysis/phase2_classifier_gate/outputs/"
    / "composite_pairrank_chemprop"
)
CHEMPROP_DEFAULTS = {
    "cp_abs005": REPO_ROOT
    / "data/chembl/pairwise_deep_binding_random250k/"
    / "pxr_pairwise_uniform_abs005_100kp5_scores.csv",
    "cp_abs01": REPO_ROOT
    / "data/chembl/pairwise_deep_binding_random250k/"
    / "pxr_pairwise_uniform_abs100kp5_scores.csv",
    "cp_abs02": REPO_ROOT
    / "data/chembl/pairwise_deep_binding_random250k/"
    / "pxr_pairwise_uniform_abs02_100kp5_scores.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairrank-dir", type=Path, default=DEFAULT_PAIRRANK_DIR)
    parser.add_argument("--anchor", type=Path, default=DEFAULT_ANCHOR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def load_chemprop_scores() -> list[pd.DataFrame]:
    frames = []
    for name, path in CHEMPROP_DEFAULTS.items():
        frame = pd.read_csv(path)[["molecule_name", "pairwise_score"]].rename(
            columns={"pairwise_score": name}
        )
        frames.append(frame)
    return frames


def merge_scores(base: pd.DataFrame) -> pd.DataFrame:
    out = base.copy()
    for frame in load_chemprop_scores():
        out = out.merge(frame, on="molecule_name", how="left")
    return out


def add_composites(
    pool: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_cols = [
        "pairrank_chembl",
        "pairrank_htchem",
        "pairrank_all",
        "cp_abs005",
        "cp_abs01",
        "cp_abs02",
    ]
    train_ref = pool[pool["source"].eq("train")]
    pool = pool.copy()
    test = test.copy()
    for col in score_cols:
        mean = float(train_ref[col].mean())
        std = float(train_ref[col].std(ddof=0))
        if std == 0:
            raise RuntimeError(f"Zero std for {col}")
        pool[f"z_{col}"] = (pool[col] - mean) / std
        test[f"z_{col}"] = (test[col] - mean) / std

    for df in [pool, test]:
        df["combo_high_chembl_cp01"] = df["z_pairrank_chembl"] + df["z_cp_abs01"]
        df["combo_high_chembl_cp02"] = df["z_pairrank_chembl"] + df["z_cp_abs02"]
        df["combo_high_htchem_cp02"] = df["z_pairrank_htchem"] + df["z_cp_abs02"]
        df["combo_low_chembl_cp01"] = -df["z_pairrank_chembl"] - df["z_cp_abs01"]
        df["combo_low_chembl_cp005"] = -df["z_pairrank_chembl"] - df["z_cp_abs005"]
    return pool, test


def load_anchor(path: Path) -> pd.DataFrame:
    return pd.read_csv(path).rename(
        columns={"Molecule Name": "molecule_name", "pEC50": "id55"}
    )[["molecule_name", "id55"]]


def metric_rows(as1: pd.DataFrame, score_cols: list[str]) -> pd.DataFrame:
    rows = []
    y = as1["pec50"].to_numpy(dtype=float)
    high = (y >= 6.0).astype(int)
    low = (y < 3.0).astype(int)
    for col in score_cols:
        score = as1[col].to_numpy(dtype=float)
        rows.append(
            {
                "score": col,
                "spearman": float(stats.spearmanr(score, y).statistic),
                "gte6_auc": float(roc_auc_score(high, score)),
                "gte6_ap": float(average_precision_score(high, score)),
                "lt3_auc": float(roc_auc_score(low, -score)),
                "lt3_ap": float(average_precision_score(low, -score)),
            }
        )
    return pd.DataFrame(rows).sort_values("gte6_ap", ascending=False)


def scan_gates(as1: pd.DataFrame, score_cols: list[str]) -> pd.DataFrame:
    y = as1["pec50"].to_numpy(dtype=float)
    anchor = as1["id55"].to_numpy(dtype=float)
    base_mae = float(np.mean(np.abs(anchor - y)))
    rows = []
    for col in score_cols:
        for mode, sign in [("high_lift", 1.0), ("low_drop", -1.0)]:
            oriented = as1[col].to_numpy(dtype=float)
            if mode == "low_drop":
                oriented = -oriented
            for q in [0.85, 0.90, 0.93, 0.95, 0.97]:
                threshold = float(np.quantile(oriented, q))
                mask = oriented >= threshold
                for mag in [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
                    pred = anchor.copy()
                    pred[mask] += sign * mag
                    mae = float(np.mean(np.abs(pred - y)))
                    rows.append(
                        {
                            "score": col,
                            "mode": mode,
                            "q": q,
                            "shift": sign * mag,
                            "threshold": threshold,
                            "mae": mae,
                            "delta": mae - base_mae,
                            "n": int(mask.sum()),
                            "high": int(((y >= 6.0) & mask).sum()),
                            "low": int(((y < 3.0) & mask).sum()),
                            "mean_id55_error": float(np.mean(anchor[mask] - y[mask]))
                            if mask.any()
                            else np.nan,
                        }
                    )
    return pd.DataFrame(rows).sort_values(["delta", "n"])


def flagged_rows(as1: pd.DataFrame, best: pd.Series) -> pd.DataFrame:
    oriented = as1[best["score"]].to_numpy(dtype=float)
    if best["mode"] == "low_drop":
        oriented = -oriented
    mask = oriented >= float(best["threshold"])
    cols = [
        "molecule_name",
        "pec50",
        "id55",
        "pairrank_chembl",
        "pairrank_htchem",
        "cp_abs01",
        "cp_abs02",
        "combo_high_htchem_cp02",
        "combo_high_chembl_cp01",
    ]
    out = as1.loc[mask, cols].copy()
    out["id55_error"] = out["id55"] - out["pec50"]
    out["best_score"] = best["score"]
    out["suggested_shift"] = float(best["shift"])
    return out.sort_values(str(best["score"]), ascending=best["mode"] == "low_drop")


def top_test_flags(test: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "molecule_name",
        "compound_id",
        "split",
        "pec50",
        "pairrank_chembl",
        "pairrank_htchem",
        "cp_abs01",
        "cp_abs02",
        "combo_high_htchem_cp02",
        "combo_high_chembl_cp01",
    ]
    pieces = []
    for score in ["combo_high_htchem_cp02", "combo_high_chembl_cp01", "cp_abs01"]:
        top = test.nlargest(32, score)[cols].copy()
        top.insert(0, "flag_score", score)
        top.insert(1, "flag_rank", np.arange(1, len(top) + 1))
        pieces.append(top)
    return pd.concat(pieces, ignore_index=True)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pool = pd.read_csv(args.pairrank_dir / "pool_pairrank_scores.csv")
    test = pd.read_csv(args.pairrank_dir / "test_pairrank_scores.csv").rename(
        columns={"as1_pec50": "pec50"}
    )
    pool = merge_scores(pool)
    test = merge_scores(test)
    pool, test = add_composites(pool, test)
    anchor = load_anchor(args.anchor)
    as1 = test[test["split"].eq("AS1") & test["pec50"].notna()].merge(
        anchor, on="molecule_name", how="inner"
    )
    score_cols = [
        "pairrank_chembl",
        "pairrank_htchem",
        "cp_abs01",
        "cp_abs02",
        "combo_high_chembl_cp01",
        "combo_high_chembl_cp02",
        "combo_high_htchem_cp02",
        "combo_low_chembl_cp01",
        "combo_low_chembl_cp005",
    ]
    metrics = metric_rows(as1, score_cols)
    gates = scan_gates(as1, score_cols)
    flags = flagged_rows(as1, gates.iloc[0])
    test_flags = top_test_flags(test)
    pool.to_csv(args.out_dir / "pool_composite_scores.csv", index=False)
    test.to_csv(args.out_dir / "test_composite_scores.csv", index=False)
    metrics.to_csv(args.out_dir / "as1_metrics.csv", index=False)
    gates.to_csv(args.out_dir / "as1_gate_scan.csv", index=False)
    flags.to_csv(args.out_dir / "as1_best_gate_flags.csv", index=False)
    test_flags.to_csv(args.out_dir / "test_top_flags.csv", index=False)
    print(metrics.to_string(index=False))
    print("\nBest gates")
    print(gates.head(20).to_string(index=False))
    print(f"\nWrote {args.out_dir}")


if __name__ == "__main__":
    main()

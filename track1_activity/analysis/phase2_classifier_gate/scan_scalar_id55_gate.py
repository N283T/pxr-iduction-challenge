#!/usr/bin/env python
"""Scan sparse id55 gates for a scalar test/train score table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ANCHOR = (
    REPO_ROOT / "track1_activity/submissions/ens_id51_top500_potent46_t40_soft_g35.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--score-col", default="pairwise_score")
    parser.add_argument("--name", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, default=DEFAULT_ANCHOR)
    return parser.parse_args()


def load_anchor(path: Path) -> pd.DataFrame:
    return pd.read_csv(path).rename(
        columns={"Molecule Name": "molecule_name", "pEC50": "id55"}
    )[["molecule_name", "id55"]]


def metric_rows(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    rows = []
    for split, sub in df[df["pec50"].notna()].groupby("split"):
        y = sub["pec50"].to_numpy(dtype=float)
        score = sub[score_col].to_numpy(dtype=float)
        high = (y >= 6.0).astype(int)
        low = (y < 3.0).astype(int)
        rows.append(
            {
                "split": split,
                "n": int(len(sub)),
                "spearman": float(stats.spearmanr(score, y).statistic),
                "pearson": float(np.corrcoef(score, y)[0, 1]),
                "gte6_auc": float(roc_auc_score(high, score))
                if high.min() != high.max()
                else np.nan,
                "gte6_ap": float(average_precision_score(high, score))
                if high.min() != high.max()
                else np.nan,
                "lt3_auc": float(roc_auc_score(low, -score))
                if low.min() != low.max()
                else np.nan,
                "lt3_ap": float(average_precision_score(low, -score))
                if low.min() != low.max()
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def gate_scan(as1: pd.DataFrame, score_col: str) -> pd.DataFrame:
    y = as1["pec50"].to_numpy(dtype=float)
    score = as1[score_col].to_numpy(dtype=float)
    anchor = as1["id55"].to_numpy(dtype=float)
    base_mae = float(np.mean(np.abs(anchor - y)))
    rows = []
    for mode, oriented, sign in [
        ("high_lift", score, 1.0),
        ("low_drop", -score, -1.0),
    ]:
        for q in [0.80, 0.85, 0.90, 0.93, 0.95, 0.97]:
            threshold = float(np.quantile(oriented, q))
            mask = oriented >= threshold
            for mag in [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
                pred = anchor.copy()
                pred[mask] += sign * mag
                mae = float(np.mean(np.abs(pred - y)))
                rows.append(
                    {
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


def flagged_rows(as1: pd.DataFrame, score_col: str, best: pd.Series) -> pd.DataFrame:
    score = as1[score_col].to_numpy(dtype=float)
    oriented = score if best["mode"] == "high_lift" else -score
    mask = oriented >= float(best["threshold"])
    flagged = as1.loc[mask, ["molecule_name", "pec50", "id55", score_col]].copy()
    flagged["id55_error"] = flagged["id55"] - flagged["pec50"]
    flagged["suggested_shift"] = float(best["shift"])
    return flagged.sort_values(score_col, ascending=best["mode"] == "low_drop")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    scores = pd.read_csv(args.scores)
    if args.score_col not in scores.columns:
        raise KeyError(f"{args.score_col} not in {args.scores}")
    anchor = load_anchor(args.anchor)
    as1 = scores[scores["split"].eq("AS1")].merge(
        anchor, on="molecule_name", how="inner"
    )
    if as1.empty:
        raise RuntimeError("No AS1 rows after joining scores to id55 anchor.")

    metrics = metric_rows(scores, args.score_col)
    scan = gate_scan(as1, args.score_col)
    flags = flagged_rows(as1, args.score_col, scan.iloc[0])
    metrics.to_csv(args.out_dir / f"{args.name}_metrics.csv", index=False)
    scan.to_csv(args.out_dir / f"{args.name}_id55_gate_scan.csv", index=False)
    flags.to_csv(args.out_dir / f"{args.name}_best_flags.csv", index=False)
    report = {
        "name": args.name,
        "scores": str(args.scores),
        "score_col": args.score_col,
        "best_gate": scan.head(10).to_dict("records"),
        "metrics": metrics.to_dict("records"),
    }
    (args.out_dir / f"{args.name}_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(metrics.to_string(index=False))
    print("\nBest gates")
    print(scan.head(20).to_string(index=False))
    print(f"\nWrote {args.out_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Build train-quantile composite gate tables for Phase 2 submissions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_POOL = (
    REPO_ROOT
    / "track1_activity/analysis/phase2_classifier_gate/outputs/"
    / "composite_pairrank_chemprop/pool_composite_scores.csv"
)
DEFAULT_TEST = (
    REPO_ROOT
    / "track1_activity/analysis/phase2_classifier_gate/outputs/"
    / "composite_pairrank_chemprop/test_composite_scores.csv"
)
DEFAULT_ANCHOR = (
    REPO_ROOT / "track1_activity/submissions/ens_id51_top500_potent46_t40_soft_g35.csv"
)
DEFAULT_OUT_DIR = (
    REPO_ROOT
    / "track1_activity/analysis/phase2_classifier_gate/outputs/"
    / "composite_pairrank_chemprop_train_quantile_gates"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--anchor", type=Path, default=DEFAULT_ANCHOR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--high-score", default="combo_high_htchem_cp02")
    parser.add_argument("--high-q", type=float, default=0.98)
    parser.add_argument("--high-shift", type=float, default=0.20)
    parser.add_argument("--low-score", default="cp_abs01")
    parser.add_argument("--low-q", type=float, default=0.98)
    parser.add_argument("--low-shift", type=float, default=-0.20)
    parser.add_argument(
        "--name",
        default="combo_q98_lift02__cp_abs01_lowq98_drop02",
        help="Output stem. The script appends _shift.csv and _summary.json.",
    )
    return parser.parse_args()


def load_anchor(path: Path) -> pd.DataFrame:
    anchor = pd.read_csv(path)
    required = {"SMILES", "Molecule Name", "pEC50"}
    missing = required.difference(anchor.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return anchor.rename(columns={"Molecule Name": "molecule_name", "pEC50": "anchor"})[
        ["molecule_name", "SMILES", "anchor"]
    ]


def oriented_low_score(df: pd.DataFrame, score_col: str) -> pd.Series:
    """Return a high-is-low oriented score for low-tail gates."""
    return -df[score_col].astype(float)


def require_columns(df: pd.DataFrame, path: Path, columns: set[str]) -> None:
    missing = columns.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")


def summarize_flags(out: pd.DataFrame) -> dict[str, object]:
    rows: dict[str, object] = {
        "n_total": int(len(out)),
        "n_high_flags": int(out["high_flag"].sum()),
        "n_low_flags": int(out["low_flag"].sum()),
        "mean_shift": float(out["shift"].mean()),
        "mean_abs_shift": float(out["shift"].abs().mean()),
        "max_abs_shift": float(out["shift"].abs().max()),
    }
    for split, sub in out.groupby("split", dropna=False):
        prefix = f"{split}_"
        rows[f"{prefix}n"] = int(len(sub))
        rows[f"{prefix}high_flags"] = int(sub["high_flag"].sum())
        rows[f"{prefix}low_flags"] = int(sub["low_flag"].sum())
        labeled = sub[sub["pec50"].notna()]
        if not labeled.empty:
            rows[f"{prefix}true_high_in_high_flags"] = int(
                (labeled["high_flag"] & labeled["pec50"].ge(6.0)).sum()
            )
            rows[f"{prefix}true_low_in_low_flags"] = int(
                (labeled["low_flag"] & labeled["pec50"].lt(3.0)).sum()
            )
    return rows


def main() -> None:
    args = parse_args()
    pool = pd.read_csv(args.pool)
    test = pd.read_csv(args.test)
    anchor = load_anchor(args.anchor)
    require_columns(
        pool,
        args.pool,
        {"source", args.high_score, args.low_score, "molecule_name", "pec50"},
    )
    require_columns(
        test,
        args.test,
        {"split", args.high_score, args.low_score, "molecule_name", "pec50"},
    )

    train_ref = pool[pool["source"].eq("train")].copy()
    if train_ref.empty:
        raise ValueError("Pool has no source=train rows for threshold fitting.")
    high_threshold = float(train_ref[args.high_score].quantile(args.high_q))
    low_threshold = float(
        oriented_low_score(train_ref, args.low_score).quantile(args.low_q)
    )

    out = test.merge(anchor, on="molecule_name", how="left", validate="one_to_one")
    if out["anchor"].isna().any():
        missing = out.loc[out["anchor"].isna(), "molecule_name"].head()
        raise ValueError(f"Anchor did not align for: {missing.tolist()}")

    high_oriented = out[args.high_score].astype(float)
    low_oriented = oriented_low_score(out, args.low_score)
    out["high_flag"] = high_oriented >= high_threshold
    out["low_flag"] = low_oriented >= low_threshold
    out["shift"] = np.select(
        [out["high_flag"], out["low_flag"]],
        [args.high_shift, args.low_shift],
        default=0.0,
    )
    both = out["high_flag"] & out["low_flag"]
    if both.any():
        raise ValueError(
            f"Overlapping high/low flags: {out.loc[both, 'molecule_name'].tolist()}"
        )
    out["candidate_pred"] = out["anchor"].astype(float) + out["shift"].astype(float)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{args.name}_shift.csv"
    summary_path = args.out_dir / f"{args.name}_summary.json"
    out.to_csv(out_path, index=False)
    summary = {
        "pool": str(args.pool.resolve().relative_to(REPO_ROOT)),
        "test": str(args.test.resolve().relative_to(REPO_ROOT)),
        "anchor": str(args.anchor.resolve().relative_to(REPO_ROOT)),
        "output": str(out_path.resolve().relative_to(REPO_ROOT)),
        "high_score": args.high_score,
        "high_q": args.high_q,
        "high_threshold": high_threshold,
        "high_shift": args.high_shift,
        "low_score": args.low_score,
        "low_q": args.low_q,
        "low_threshold": low_threshold,
        "low_shift": args.low_shift,
        **summarize_flags(out),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Wrote {out_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()

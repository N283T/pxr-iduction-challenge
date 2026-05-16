#!/usr/bin/env python
"""Compare candidate movement against historical LB submission directions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "track1_activity" / "src"
sys.path.insert(0, str(SRC_DIR))

from data import get_engine  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "outputs" / "candidate_lb_id_compare"
SUB_DIR = REPO_ROOT / "track1_activity" / "submissions"
DEFAULT_BASE = SUB_DIR / "ens_id51_top500_potent46_t40_soft_g50.csv"


def repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load_submission(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"SMILES", "Molecule Name", "pEC50"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns {sorted(missing)}")
    return df[["SMILES", "Molecule Name", "pEC50"]].copy()


def align_values(path: Path, reference: pd.DataFrame) -> np.ndarray:
    df = load_submission(path)
    if df[["SMILES", "Molecule Name"]].equals(reference[["SMILES", "Molecule Name"]]):
        return df["pEC50"].to_numpy(dtype=np.float64)
    merged = reference[["SMILES", "Molecule Name"]].merge(
        df[["SMILES", "Molecule Name", "pEC50"]],
        on=["SMILES", "Molecule Name"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(reference):
        raise RuntimeError(f"{path} aligned {len(merged)} of {len(reference)} rows")
    return merged["pEC50"].to_numpy(dtype=np.float64)


def safe_corr(a: np.ndarray, b: np.ndarray, *, spearman: bool = False) -> float:
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    if spearman:
        return float(stats.spearmanr(a, b).statistic)
    return float(np.corrcoef(a, b)[0, 1])


def projection(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.dot(b, b))
    return float(np.dot(a, b) / denom) if denom > 1e-12 else float("nan")


def load_lb_submissions() -> pd.DataFrame:
    return pd.read_sql(
        """
        SELECT id, submission_name, file_path, lb_mae::float AS lb_mae,
               lb_spearman::float AS lb_spearman, submitted_at
        FROM lb_submissions
        WHERE track='activity' AND lb_mae IS NOT NULL
        ORDER BY id
        """,
        get_engine(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--base", default=DEFAULT_BASE, type=Path)
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    name = args.name or args.candidate.stem
    ref = load_submission(args.base)
    base = ref["pEC50"].to_numpy(dtype=np.float64)
    candidate = align_values(args.candidate, ref)
    candidate_delta = candidate - base

    lb = load_lb_submissions()
    rows: list[dict[str, float | int | str]] = []
    pred_by_id: dict[int, np.ndarray] = {}
    for row in lb.itertuples(index=False):
        path = REPO_ROOT / row.file_path
        if not path.exists():
            continue
        try:
            pred = align_values(path, ref)
        except Exception as exc:
            rows.append(
                {
                    "id": int(row.id),
                    "submission_name": row.submission_name,
                    "file_path": row.file_path,
                    "error": str(exc),
                }
            )
            continue
        pred_by_id[int(row.id)] = pred
        delta_vs_base = pred - base
        rows.append(
            {
                "id": int(row.id),
                "submission_name": row.submission_name,
                "file_path": row.file_path,
                "lb_mae": float(row.lb_mae),
                "lb_delta_mae_vs_base": float(
                    row.lb_mae
                    - lb.loc[
                        lb["file_path"] == repo_relative(args.base), "lb_mae"
                    ].iloc[-1]
                )
                if (lb["file_path"] == repo_relative(args.base)).any()
                else float("nan"),
                "lb_spearman": float(row.lb_spearman),
                "mean_abs_delta_vs_base": float(np.mean(np.abs(delta_vs_base))),
                "p90_abs_delta_vs_base": float(
                    np.quantile(np.abs(delta_vs_base), 0.90)
                ),
                "max_abs_delta_vs_base": float(np.max(np.abs(delta_vs_base))),
                "pearson_delta": safe_corr(candidate_delta, delta_vs_base),
                "spearman_delta": safe_corr(
                    candidate_delta, delta_vs_base, spearman=True
                ),
                "projection_on_id_delta": projection(candidate_delta, delta_vs_base),
                "id_projection_on_candidate": projection(
                    delta_vs_base, candidate_delta
                ),
            }
        )

    direct = pd.DataFrame(rows)
    direct.to_csv(OUT_DIR / f"{name}_direct_id_comparison.csv", index=False)

    pair_rows: list[dict[str, float | int | str]] = []
    ids = sorted(pred_by_id)
    lb_by_id = lb.set_index("id")
    for from_id in ids:
        for to_id in ids:
            if from_id >= to_id:
                continue
            direction = pred_by_id[to_id] - pred_by_id[from_id]
            if np.std(direction) < 1e-12:
                continue
            pair_rows.append(
                {
                    "from_id": from_id,
                    "to_id": to_id,
                    "from_name": lb_by_id.loc[from_id, "submission_name"],
                    "to_name": lb_by_id.loc[to_id, "submission_name"],
                    "from_lb_mae": float(lb_by_id.loc[from_id, "lb_mae"]),
                    "to_lb_mae": float(lb_by_id.loc[to_id, "lb_mae"]),
                    "lb_delta_mae": float(
                        lb_by_id.loc[to_id, "lb_mae"] - lb_by_id.loc[from_id, "lb_mae"]
                    ),
                    "lb_delta_spearman": float(
                        lb_by_id.loc[to_id, "lb_spearman"]
                        - lb_by_id.loc[from_id, "lb_spearman"]
                    ),
                    "mean_abs_direction": float(np.mean(np.abs(direction))),
                    "pearson_delta": safe_corr(candidate_delta, direction),
                    "spearman_delta": safe_corr(
                        candidate_delta, direction, spearman=True
                    ),
                    "candidate_projection": projection(candidate_delta, direction),
                    "direction_projection_on_candidate": projection(
                        direction, candidate_delta
                    ),
                }
            )

    pairwise = pd.DataFrame(pair_rows).sort_values(
        ["pearson_delta", "candidate_projection"], ascending=False
    )
    pairwise.to_csv(OUT_DIR / f"{name}_pairwise_id_directions.csv", index=False)

    useful_pairs = pairwise[
        (pairwise["lb_delta_mae"] < 0)
        & (pairwise["pearson_delta"] > 0)
        & (pairwise["candidate_projection"] > 0)
    ].copy()
    harmful_pairs = pairwise[
        (pairwise["lb_delta_mae"] > 0)
        & (pairwise["pearson_delta"] > 0)
        & (pairwise["candidate_projection"] > 0)
    ].copy()

    summary = pd.DataFrame(
        [
            {
                "candidate": repo_relative(args.candidate),
                "base": repo_relative(args.base),
                "candidate_mean_shift": float(candidate_delta.mean()),
                "candidate_mean_abs_shift": float(np.mean(np.abs(candidate_delta))),
                "candidate_p90_abs_shift": float(
                    np.quantile(np.abs(candidate_delta), 0.90)
                ),
                "candidate_max_abs_shift": float(np.max(np.abs(candidate_delta))),
                "n_direct_ids": int(len(direct.dropna(subset=["pearson_delta"]))),
                "max_direct_pearson": float(direct["pearson_delta"].max()),
                "max_harmful_pair_pearson": float(harmful_pairs["pearson_delta"].max())
                if not harmful_pairs.empty
                else float("nan"),
                "max_useful_pair_pearson": float(useful_pairs["pearson_delta"].max())
                if not useful_pairs.empty
                else float("nan"),
            }
        ]
    )
    summary.to_csv(OUT_DIR / f"{name}_summary.csv", index=False)

    report = [
        f"# LB ID Direction Compare: `{name}`",
        "",
        summary.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Most Similar Direct IDs",
        "",
        direct.sort_values("pearson_delta", ascending=False)[
            [
                "id",
                "submission_name",
                "lb_mae",
                "lb_delta_mae_vs_base",
                "mean_abs_delta_vs_base",
                "pearson_delta",
                "projection_on_id_delta",
            ]
        ]
        .head(20)
        .to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Similar Helpful Historical Directions",
        "",
        useful_pairs[
            [
                "from_id",
                "to_id",
                "from_name",
                "to_name",
                "lb_delta_mae",
                "pearson_delta",
                "candidate_projection",
            ]
        ]
        .head(20)
        .to_markdown(index=False, floatfmt=".6f")
        if not useful_pairs.empty
        else "None.",
        "",
        "## Similar Harmful Historical Directions",
        "",
        harmful_pairs[
            [
                "from_id",
                "to_id",
                "from_name",
                "to_name",
                "lb_delta_mae",
                "pearson_delta",
                "candidate_projection",
            ]
        ]
        .head(20)
        .to_markdown(index=False, floatfmt=".6f")
        if not harmful_pairs.empty
        else "None.",
        "",
    ]
    (OUT_DIR / f"{name}_report.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    print(f"\nWrote {OUT_DIR}")


if __name__ == "__main__":
    main()

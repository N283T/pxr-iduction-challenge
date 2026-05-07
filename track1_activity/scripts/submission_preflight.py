#!/usr/bin/env python
"""Pre-submit risk report for Track 1 activity submission CSVs.

This is a cheap guardrail, not an LB oracle. It checks that a candidate
submission is aligned to the current test set and quantifies how far it moves
from a trusted anchor and known-bad LB directions.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT.joinpath("track1_activity", "src")
sys.path.insert(0, str(SRC_DIR))

from data import get_engine, load_test_smiles  # noqa: E402

SUB_DIR = REPO_ROOT.joinpath("track1_activity", "submissions")
DEFAULT_ANCHOR = SUB_DIR.joinpath("ens_id51_top500_potent46_t40_soft_g35.csv")
DEFAULT_OUT_DIR = REPO_ROOT.joinpath(
    "track1_activity", "analysis", "submission_preflight", "outputs"
)

KNOWN_BAD_AXES = {
    "id56_minus_id55": (
        SUB_DIR.joinpath("ens_swap_optuna_t10_top500_calibrated_importance.csv"),
        SUB_DIR.joinpath("ens_id51_top500_potent46_t40_soft_g35.csv"),
    ),
    "id56_minus_id51": (
        SUB_DIR.joinpath("ens_swap_optuna_t10_top500_calibrated_importance.csv"),
        SUB_DIR.joinpath("ens_meta_axis_reverse_id50_g10.csv"),
    ),
}


@dataclass(frozen=True)
class PreflightMetrics:
    pearson: float
    spearman: float
    mean_shift: float
    mean_abs_shift: float
    p90_abs_shift: float
    max_abs_shift: float
    n_abs_gt_005: int
    n_abs_gt_010: int
    n_abs_gt_020: int
    candidate_mean: float
    candidate_std: float
    anchor_mean: float
    anchor_std: float


@dataclass(frozen=True)
class BadAxisResult:
    label: str
    pearson: float
    spearman: float
    candidate_projection: float


@dataclass(frozen=True)
class RiskVerdict:
    level: str
    reasons: list[str]


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
        raise ValueError(f"{path} missing Track 1 columns: {sorted(missing)}")
    return df[["SMILES", "Molecule Name", "pEC50"]].copy()


def validate_current_test_order(df: pd.DataFrame) -> dict[str, object]:
    ref = load_test_smiles()
    smiles_match = df["SMILES"].tolist() == ref["smiles"].tolist()
    names_match = df["Molecule Name"].tolist() == ref["molecule_name"].tolist()
    return {
        "rows": len(df),
        "expected_rows": len(ref),
        "smiles_match": smiles_match,
        "names_match": names_match,
        "order_ok": len(df) == len(ref) and smiles_match and names_match,
    }


def align_submission(candidate: pd.DataFrame, anchor: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["SMILES", "Molecule Name"]
    if not candidate[key_cols].equals(anchor[key_cols]):
        merged = anchor[key_cols + ["pEC50"]].rename(columns={"pEC50": "anchor"})
        merged = merged.merge(
            candidate[key_cols + ["pEC50"]].rename(columns={"pEC50": "candidate"}),
            on=key_cols,
            how="inner",
        )
        if len(merged) != len(anchor):
            raise ValueError(
                f"candidate and anchor aligned {len(merged)} of {len(anchor)} rows"
            )
        return merged
    return pd.DataFrame(
        {
            "SMILES": candidate["SMILES"],
            "Molecule Name": candidate["Molecule Name"],
            "anchor": anchor["pEC50"],
            "candidate": candidate["pEC50"],
        }
    )


def compute_shift_metrics(
    anchor: np.ndarray, candidate: np.ndarray
) -> PreflightMetrics:
    if anchor.shape != candidate.shape:
        raise ValueError(
            f"shape mismatch: anchor={anchor.shape}, candidate={candidate.shape}"
        )
    delta = candidate - anchor
    abs_delta = np.abs(delta)
    return PreflightMetrics(
        pearson=float(np.corrcoef(anchor, candidate)[0, 1]),
        spearman=float(stats.spearmanr(anchor, candidate).statistic),
        mean_shift=float(delta.mean()),
        mean_abs_shift=float(abs_delta.mean()),
        p90_abs_shift=float(np.quantile(abs_delta, 0.90)),
        max_abs_shift=float(abs_delta.max()),
        n_abs_gt_005=int((abs_delta > 0.05).sum()),
        n_abs_gt_010=int((abs_delta > 0.10).sum()),
        n_abs_gt_020=int((abs_delta > 0.20).sum()),
        candidate_mean=float(candidate.mean()),
        candidate_std=float(candidate.std(ddof=1)),
        anchor_mean=float(anchor.mean()),
        anchor_std=float(anchor.std(ddof=1)),
    )


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def bad_axis_correlations(
    candidate_delta: np.ndarray,
    bad_axes: dict[str, tuple[Path, Path]] | None = None,
) -> list[BadAxisResult]:
    rows: list[BadAxisResult] = []
    for label, (bad_path, base_path) in (bad_axes or KNOWN_BAD_AXES).items():
        if not bad_path.exists() or not base_path.exists():
            continue
        bad_df = load_submission(bad_path)
        base_df = load_submission(base_path)
        aligned = align_submission(bad_df, base_df)
        bad_delta = aligned["candidate"].to_numpy(dtype=np.float64) - aligned[
            "anchor"
        ].to_numpy(dtype=np.float64)
        denom = float(np.dot(bad_delta, bad_delta))
        projection = (
            float(np.dot(candidate_delta, bad_delta) / denom)
            if denom > 0
            else float("nan")
        )
        rows.append(
            BadAxisResult(
                label=label,
                pearson=_safe_corr(candidate_delta, bad_delta),
                spearman=float(stats.spearmanr(candidate_delta, bad_delta).statistic),
                candidate_projection=projection,
            )
        )
    return rows


def classify_risk(
    metrics: PreflightMetrics, bad_axis_rows: list[BadAxisResult]
) -> RiskVerdict:
    hold_reasons: list[str] = []
    caution_reasons: list[str] = []

    if metrics.n_abs_gt_010 >= 100 or metrics.n_abs_gt_020 >= 25:
        hold_reasons.append("large_anchor_shift")
    if metrics.max_abs_shift >= 0.35:
        hold_reasons.append("extreme_single_compound_shift")
    if metrics.spearman < 0.995:
        caution_reasons.append("rank_order_changed")
    if abs(metrics.candidate_std - metrics.anchor_std) >= 0.08:
        caution_reasons.append("prediction_scale_changed")

    for row in bad_axis_rows:
        if (
            not np.isnan(row.pearson)
            and row.pearson >= 0.70
            and row.candidate_projection > 0
        ):
            caution_reasons.append("aligned_with_known_bad_axis")
            break

    if hold_reasons:
        return RiskVerdict("HOLD", hold_reasons + sorted(set(caution_reasons)))
    if caution_reasons:
        return RiskVerdict("CAUTION", sorted(set(caution_reasons)))
    return RiskVerdict("PASS", ["small_anchor_shift"])


def lookup_experiment_rows(candidate_path: Path) -> pd.DataFrame:
    rel = repo_relative(candidate_path)
    query = """
        SELECT id, name, model_type, feature_set, mae_mean::float AS mae,
               rae_mean::float AS rae, spearman_mean::float AS spearman,
               submission_path, created_at
        FROM experiment_summary
        WHERE submission_path = %(path)s
        ORDER BY created_at DESC
        LIMIT 5
    """
    return pd.read_sql(query, get_engine(), params={"path": rel})


def write_outputs(
    *,
    out_dir: Path,
    candidate_path: Path,
    anchor_path: Path,
    order_check: dict[str, object],
    metrics: PreflightMetrics,
    verdict: RiskVerdict,
    bad_rows: list[BadAxisResult],
    aligned: pd.DataFrame,
    experiment_rows: pd.DataFrame,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame(
        [
            {
                "candidate": repo_relative(candidate_path),
                "anchor": repo_relative(anchor_path),
                "verdict": verdict.level,
                "reasons": ";".join(verdict.reasons),
                **order_check,
                **metrics.__dict__,
            }
        ]
    )
    summary.to_csv(out_dir.joinpath("summary.csv"), index=False)

    bad_axis_df = pd.DataFrame([row.__dict__ for row in bad_rows])
    bad_axis_df.to_csv(out_dir.joinpath("bad_axis_correlations.csv"), index=False)

    detail = aligned.copy()
    detail["shift"] = detail["candidate"] - detail["anchor"]
    detail["abs_shift"] = detail["shift"].abs()
    detail.sort_values("abs_shift", ascending=False).head(50).to_csv(
        out_dir.joinpath("largest_shifts.csv"), index=False
    )

    if not experiment_rows.empty:
        experiment_rows.to_csv(out_dir.joinpath("experiment_rows.csv"), index=False)

    report = [
        f"# Submission Preflight: `{candidate_path.name}`",
        "",
        f"Verdict: **{verdict.level}**",
        "",
        "## Inputs",
        "",
        f"- candidate: `{repo_relative(candidate_path)}`",
        f"- anchor: `{repo_relative(anchor_path)}`",
        "",
        "## CSV Sanity",
        "",
        f"- rows: {order_check['rows']} / {order_check['expected_rows']}",
        f"- SMILES order match: {order_check['smiles_match']}",
        f"- Molecule Name order match: {order_check['names_match']}",
        "",
        "## Anchor Shift",
        "",
        f"- Pearson vs anchor: {metrics.pearson:.6f}",
        f"- Spearman vs anchor: {metrics.spearman:.6f}",
        f"- mean shift: {metrics.mean_shift:+.6f}",
        f"- mean abs shift: {metrics.mean_abs_shift:.6f}",
        f"- p90 abs shift: {metrics.p90_abs_shift:.6f}",
        f"- max abs shift: {metrics.max_abs_shift:.6f}",
        f"- |shift| > 0.05: {metrics.n_abs_gt_005}",
        f"- |shift| > 0.10: {metrics.n_abs_gt_010}",
        f"- |shift| > 0.20: {metrics.n_abs_gt_020}",
        "",
        "## Prediction Distribution",
        "",
        f"- anchor mean/std: {metrics.anchor_mean:.6f} / {metrics.anchor_std:.6f}",
        f"- candidate mean/std: {metrics.candidate_mean:.6f} / {metrics.candidate_std:.6f}",
        "",
        "## Known Bad Axis",
        "",
    ]
    if bad_rows:
        report.extend(bad_axis_df.to_markdown(index=False, floatfmt=".6f").splitlines())
    else:
        report.append("No known bad-axis files were available.")
    report.extend(["", "## Experiment Metadata", ""])
    if experiment_rows.empty:
        report.append("No matching `experiment_summary` row found for this CSV path.")
    else:
        report.extend(
            experiment_rows[["id", "name", "mae", "rae", "spearman", "created_at"]]
            .to_markdown(index=False, floatfmt=".6f")
            .splitlines()
        )
    report.extend(["", "## Reasons", ""])
    report.extend([f"- {reason}" for reason in verdict.reasons])
    report.append("")
    out_dir.joinpath("report.md").write_text("\n".join(report), encoding="utf-8")


def run_preflight(
    candidate_path: Path, anchor_path: Path, out_dir: Path
) -> RiskVerdict:
    candidate = load_submission(candidate_path)
    anchor = load_submission(anchor_path)
    order_check = validate_current_test_order(candidate)
    aligned = align_submission(candidate, anchor)
    anchor_pred = aligned["anchor"].to_numpy(dtype=np.float64)
    candidate_pred = aligned["candidate"].to_numpy(dtype=np.float64)
    metrics = compute_shift_metrics(anchor_pred, candidate_pred)
    bad_rows = bad_axis_correlations(candidate_pred - anchor_pred)
    verdict = classify_risk(metrics, bad_rows)
    experiment_rows = lookup_experiment_rows(candidate_path)
    write_outputs(
        out_dir=out_dir,
        candidate_path=candidate_path,
        anchor_path=anchor_path,
        order_check=order_check,
        metrics=metrics,
        verdict=verdict,
        bad_rows=bad_rows,
        aligned=aligned,
        experiment_rows=experiment_rows,
    )
    return verdict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--anchor", default=DEFAULT_ANCHOR, type=Path)
    parser.add_argument("--name", default=None)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    name = args.name or args.candidate.stem
    out_dir = args.out_dir.joinpath(name)
    verdict = run_preflight(args.candidate, args.anchor, out_dir)
    print(f"Verdict: {verdict.level} ({', '.join(verdict.reasons)})")
    print(f"Wrote: {repo_relative(out_dir.joinpath('report.md'))}")


if __name__ == "__main__":
    main()

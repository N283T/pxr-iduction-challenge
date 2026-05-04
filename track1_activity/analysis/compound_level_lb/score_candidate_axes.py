#!/usr/bin/env python
"""Cheap scorer for candidate prediction axes.

The scorer is intentionally conservative. It does not train or blend models;
it records whether existing artifacts are even worth a deeper blend/ensemble
test before spending GPU time or leaderboard cooldown.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))

from data import get_engine  # noqa: E402
from evaluate import load_oof_predictions  # noqa: E402

SUB_DIR = REPO_ROOT.joinpath("track1_activity", "submissions")
OUT_DIR = Path(__file__).resolve().parent / "outputs" / "candidate_axis_scorer"
REFERENCE_PATH = SUB_DIR / "ens_hybrid_meta_baseline_5050.csv"

TEST_CSV_CANDIDATES = [
    SUB_DIR / "tabpfn_drugclip_fold0_embed_umap_default.csv",
]

SUMMARY_CANDIDATES = [
    SUB_DIR / "krovex" / "summary.csv",
    SUB_DIR / "krovex_grid" / "summary.csv",
    SUB_DIR / "krovex_dimsweep" / "summary.csv",
    SUB_DIR / "krovex_multiseed" / "summary.csv",
    SUB_DIR / "krovex_foundation" / "summary.csv",
    SUB_DIR / "krovex_gcn64" / "summary.csv",
    SUB_DIR / "umsgfnet_fp_kan" / "phaseA_summary.csv",
    SUB_DIR / "umsgfnet_fp_kan" / "phaseA5_summary.csv",
    SUB_DIR / "umsgfnet_fp_kan" / "phase_b1_a5_summary.csv",
    SUB_DIR / "umsgfnet_fp_kan" / "phase_b2_a5_summary.csv",
    SUB_DIR / "umsgfnet_fp_kan" / "phaseC_aggregate.csv",
    SUB_DIR / "umsgfnet_fp_kan" / "phaseD1_summary.csv",
    SUB_DIR / "umsgfnet_fp_kan" / "phaseD2_summary.csv",
    SUB_DIR / "umsgfnet_fp_kan" / "phaseD3_summary.csv",
]

OOF_CANDIDATE_DIRS = [
    SUB_DIR / "krovex",
    SUB_DIR / "krovex_grid",
    SUB_DIR / "krovex_dimsweep",
    SUB_DIR / "umsgfnet_fp_kan",
]


def prediction_metrics(
    reference: np.ndarray, candidate: np.ndarray
) -> dict[str, float]:
    if reference.shape != candidate.shape:
        raise ValueError(
            f"shape mismatch: reference {reference.shape}, candidate {candidate.shape}"
        )
    delta = candidate - reference
    pearson = float(np.corrcoef(reference, candidate)[0, 1])
    spearman = float(stats.spearmanr(reference, candidate).statistic)
    return {
        "mean_delta": float(delta.mean()),
        "mean_abs_delta": float(np.abs(delta).mean()),
        "p90_abs_delta": float(np.quantile(np.abs(delta), 0.90)),
        "max_abs_delta": float(np.abs(delta).max()),
        "pearson": pearson,
        "spearman": spearman,
        "candidate_mean": float(candidate.mean()),
        "candidate_std": float(candidate.std(ddof=0)),
    }


def residual_correlation(
    y_true: np.ndarray, reference_pred: np.ndarray, candidate_pred: np.ndarray
) -> float:
    if y_true.shape != reference_pred.shape or y_true.shape != candidate_pred.shape:
        raise ValueError(
            "shape mismatch: "
            f"y={y_true.shape}, reference={reference_pred.shape}, "
            f"candidate={candidate_pred.shape}"
        )
    ref_resid = y_true - reference_pred
    cand_resid = y_true - candidate_pred
    return float(np.corrcoef(ref_resid, cand_resid)[0, 1])


def recommend_candidate(
    *,
    single_mae: float | None,
    pearson_vs_reference: float,
    mean_abs_shift: float,
    has_test_predictions: bool,
) -> dict[str, object]:
    reasons: list[str] = []
    if single_mae is not None and not math.isnan(single_mae) and single_mae > 0.485:
        reasons.append("weak_single")
        return {"decision": "close", "reasons": reasons}

    if not has_test_predictions:
        reasons.append("missing_test_predictions")
        return {"decision": "needs_test_predictions", "reasons": reasons}

    if pearson_vs_reference >= 0.997:
        reasons.append("high_correlation")
        return {"decision": "blend_only", "reasons": reasons}

    if mean_abs_shift > 0.06:
        reasons.append("large_unanchored_shift")
        return {"decision": "blend_only", "reasons": reasons}

    reasons.append("passes_cheap_axis_gate")
    return {"decision": "review", "reasons": reasons}


def load_submission_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Molecule Name" not in df.columns or "pEC50" not in df.columns:
        raise ValueError(f"not a Track 1 submission CSV: {path}")
    return df.rename(columns={"Molecule Name": "molecule_name"})[
        ["molecule_name", "pEC50"]
    ]


def load_experiment_metrics() -> pd.DataFrame:
    return pd.read_sql(
        """
        SELECT name, mae_mean::float AS mae, spearman_mean::float AS spearman,
               submission_path
        FROM experiment_summary
        """,
        get_engine(),
    )


def load_reference_oof() -> tuple[np.ndarray, np.ndarray]:
    row = pd.read_sql(
        """
        SELECT e.id
        FROM experiments e
        JOIN experiment_oof_predictions o ON o.experiment_id = e.id
        WHERE e.name IN (
            'ens_caruana_bag20',
            'ens_caruana_bag20_calibrated_best',
            'ens_caruana_bag20_calibrated_linear_pos',
            'ens_caruana_bag20_calibrated_importance'
        )
        GROUP BY e.id
        HAVING count(o.train_idx) = 4140
        ORDER BY e.id DESC
        LIMIT 1
        """,
        get_engine(),
    )
    if row.empty:
        raise RuntimeError("missing ensemble OOF reference in experiments")
    oof = load_oof_predictions(int(row.iloc[0]["id"]))
    y = pd.read_sql(
        "SELECT pec50 FROM train_activity ORDER BY id",
        get_engine(),
    )["pec50"].to_numpy(dtype=np.float64)
    return y, oof


def score_test_csv_candidates() -> pd.DataFrame:
    reference = load_submission_predictions(REFERENCE_PATH).rename(
        columns={"pEC50": "reference"}
    )
    exp = load_experiment_metrics()
    rows = []
    for path in TEST_CSV_CANDIDATES:
        if not path.exists():
            rows.append(
                {
                    "candidate": path.name,
                    "path": str(path.relative_to(REPO_ROOT)),
                    "exists": False,
                    "decision": "missing_artifact",
                    "reasons": "missing_artifact",
                }
            )
            continue
        cand = load_submission_predictions(path)
        merged = reference.merge(cand, on="molecule_name", how="inner")
        if len(merged) != len(reference):
            raise RuntimeError(f"{path} aligned {len(merged)} of {len(reference)} rows")
        metrics = prediction_metrics(
            merged["reference"].to_numpy(), merged["pEC50"].to_numpy()
        )
        rel_path = str(path.relative_to(REPO_ROOT))
        exp_match = exp.loc[exp["submission_path"] == rel_path]
        single_mae = float(exp_match.iloc[0]["mae"]) if len(exp_match) > 0 else math.nan
        single_sp = (
            float(exp_match.iloc[0]["spearman"]) if len(exp_match) > 0 else math.nan
        )
        rec = recommend_candidate(
            single_mae=single_mae,
            pearson_vs_reference=metrics["pearson"],
            mean_abs_shift=metrics["mean_abs_delta"],
            has_test_predictions=True,
        )
        rows.append(
            {
                "candidate": path.stem,
                "path": rel_path,
                "exists": True,
                "single_mae": single_mae,
                "single_sp": single_sp,
                **metrics,
                "decision": rec["decision"],
                "reasons": ";".join(rec["reasons"]),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "test_csv_candidate_scores.csv", index=False)
    return out


def _best_rows_from_summary(path: Path) -> list[dict[str, object]]:
    df = pd.read_csv(path)
    mae_col = next((c for c in ["mae", "mae_mean"] if c in df.columns), None)
    sp_col = next((c for c in ["sp", "spearman", "sp_mean"] if c in df.columns), None)
    if mae_col is None:
        return []
    name_cols = [
        c for c in ["tag", "variant", "model", "case", "features"] if c in df.columns
    ]
    df = df.sort_values(mae_col, ascending=True).head(5).copy()
    rows = []
    for _, row in df.iterrows():
        candidate_bits = [str(row[c]) for c in name_cols if pd.notna(row[c])]
        candidate = " / ".join(candidate_bits) if candidate_bits else path.parent.name
        single_mae = float(row[mae_col])
        rec = recommend_candidate(
            single_mae=single_mae,
            pearson_vs_reference=math.nan,
            mean_abs_shift=math.nan,
            has_test_predictions=False,
        )
        rows.append(
            {
                "source": str(path.relative_to(REPO_ROOT)),
                "candidate": candidate,
                "single_mae": single_mae,
                "single_sp": float(row[sp_col]) if sp_col is not None else math.nan,
                "decision": rec["decision"],
                "reasons": ";".join(rec["reasons"]),
            }
        )
    return rows


def score_summary_candidates() -> pd.DataFrame:
    rows = []
    for path in SUMMARY_CANDIDATES:
        if path.exists():
            rows.extend(_best_rows_from_summary(path))
        else:
            rows.append(
                {
                    "source": str(path.relative_to(REPO_ROOT)),
                    "candidate": path.parent.name,
                    "single_mae": math.nan,
                    "single_sp": math.nan,
                    "decision": "missing_artifact",
                    "reasons": "missing_artifact",
                }
            )
    out = pd.DataFrame(rows)
    out = out.sort_values(["decision", "single_mae"], na_position="last")
    out.to_csv(OUT_DIR / "summary_candidate_scores.csv", index=False)
    return out


def score_oof_candidates() -> pd.DataFrame:
    y, reference_oof = load_reference_oof()
    rows = []
    for directory in OOF_CANDIDATE_DIRS:
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.npy")):
            pred = np.load(path)
            if pred.shape != y.shape:
                continue
            mae = float(np.mean(np.abs(pred - y)))
            sp = float(stats.spearmanr(y, pred).statistic)
            resid_r = residual_correlation(y, reference_oof, pred)
            rec = recommend_candidate(
                single_mae=mae,
                pearson_vs_reference=math.nan,
                mean_abs_shift=math.nan,
                has_test_predictions=False,
            )
            rows.append(
                {
                    "candidate": path.stem,
                    "path": str(path.relative_to(REPO_ROOT)),
                    "single_mae": mae,
                    "single_sp": sp,
                    "residual_r_vs_current_ensemble": resid_r,
                    "decision": rec["decision"],
                    "reasons": ";".join(rec["reasons"]),
                }
            )
    out = pd.DataFrame(rows).sort_values("single_mae", na_position="last")
    out.to_csv(OUT_DIR / "oof_candidate_scores.csv", index=False)
    return out


def write_report(
    test_scores: pd.DataFrame, summary_scores: pd.DataFrame, oof_scores: pd.DataFrame
) -> None:
    closed = summary_scores.loc[summary_scores["decision"] == "close"].head(15)
    oof_top = oof_scores.head(20) if not oof_scores.empty else oof_scores
    report = [
        "# Candidate Axis Scorer",
        "",
        "## Test CSV Candidates",
        "",
        test_scores.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Best Summary-Only Candidates",
        "",
        summary_scores.head(25).to_markdown(index=False, floatfmt=".4f"),
        "",
        "## OOF Candidate Residual Check",
        "",
        oof_top.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Interpretation",
        "",
        "- `close`: single-model MAE is above 0.485, so do not spend LB cooldown or GPU.",
        "- `needs_test_predictions`: OOF/summary exists, but no test-side axis is available.",
        "- `blend_only`: candidate is too correlated or too large-shift for direct ADD.",
        "- `review`: passes the cheap axis gate and deserves a deeper residual/OOF check.",
        "",
        "## Closed Weak Singles",
        "",
        closed[["source", "candidate", "single_mae", "single_sp"]].to_markdown(
            index=False, floatfmt=".4f"
        ),
    ]
    (OUT_DIR / "report.md").write_text("\n".join(report) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    test_scores = score_test_csv_candidates()
    summary_scores = score_summary_candidates()
    oof_scores = score_oof_candidates()
    write_report(test_scores, summary_scores, oof_scores)
    print(f"Wrote candidate-axis scorer outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()

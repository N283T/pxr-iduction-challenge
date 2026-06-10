#!/usr/bin/env -S pixi run python
"""Score Track 1 prediction candidates with Phase 2 diagnostics.

This is a reporting tool, not a submission generator. It combines:

- AS1 replay metrics when a candidate test CSV is available.
- Test-set shift metrics against the id55 anchor, split by AS1/AS2 and AS2
  risk-map slices.
- OOF stress-slice metrics when an experiment OOF name is available.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy import text

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "analysis", "error_anatomy")))

from data import get_engine  # noqa: E402
from evaluate import load_oof_predictions  # noqa: E402

import error_anatomy as ea  # noqa: E402

SUBMISSION_DIR = REPO_ROOT / "track1_activity" / "submissions"
OUT_DIR = Path(__file__).resolve().parent.joinpath("outputs")
DOC_REPORT_PATH = REPO_ROOT / "docs" / "track1_explain" / "phase2_candidate_scorecard.md"
ANCHOR_PATH = SUBMISSION_DIR / "ens_id51_top500_potent46_t40_soft_g35.csv"
BAD_AXIS_PATH = SUBMISSION_DIR / "ens_swap_optuna_t10_top500_calibrated_importance.csv"
AS2_RISK_MAP_PATH = (
    REPO_ROOT / "track1_activity" / "analysis" / "phase2_as2_risk_map" / "outputs" / "as2_risk_map.csv"
)
TRAIN_SLICE_PATH = (
    REPO_ROOT
    / "track1_activity"
    / "analysis"
    / "phase2_validation_slices"
    / "outputs"
    / "train_oof_validation_slices.csv"
)

DEFAULT_CANDIDATES = [
    SUBMISSION_DIR / "ens_id51_top500_potent46_t40_soft_g35.csv",
    SUBMISSION_DIR / "ens_id51_top500_potent46_t40_soft_g50.csv",
    SUBMISSION_DIR / "ens_id55_combo_gate_rank1.csv",
    SUBMISSION_DIR / "ens_id57_high_activity_lift_rank2.csv",
    SUBMISSION_DIR / "ens_swap_optuna_t10_top500_calibrated_importance.csv",
]

TRUE_BINS = [-np.inf, 3.0, 4.0, 5.0, 6.0, np.inf]
TRUE_BIN_LABELS = ["lt3", "3to4", "4to5", "5to6", "gte6"]


def repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def load_submission(path: Path, column: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"Molecule Name", "pEC50"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    cols = ["Molecule Name", "pEC50"]
    if "SMILES" in df.columns:
        cols.insert(0, "SMILES")
    return df[cols].rename(
        columns={"Molecule Name": "molecule_name", "pEC50": column}
    )


def load_test_metadata() -> pd.DataFrame:
    return pd.read_sql(
        """
        SELECT
            t.id AS test_id,
            c.id AS compound_id,
            c.molecule_name,
            c.std_smiles AS smiles,
            l.pec50 AS as1_pec50
        FROM test_activity t
        JOIN compounds c ON c.id = t.compound_id
        LEFT JOIN test_activity_phase1_labels l ON l.compound_id = t.compound_id
        ORDER BY t.id
        """,
        get_engine(),
    )


def load_anchor_frame() -> pd.DataFrame:
    df = load_test_metadata()
    anchor = load_submission(ANCHOR_PATH, "anchor_pred")
    out = df.merge(anchor[["molecule_name", "anchor_pred"]], on="molecule_name")
    out["split"] = np.where(out["as1_pec50"].notna(), "AS1", "AS2")
    return out


def metric_row(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    err = pred - y
    return {
        "mae": float(np.mean(np.abs(err))),
        "bias_pred_minus_true": float(np.mean(err)),
        "spearman": float(stats.spearmanr(y, pred).statistic),
        "pred_mean": float(np.mean(pred)),
        "pred_std": float(np.std(pred, ddof=1)),
    }


def summarize_shift(delta: np.ndarray, label: str, n_total: int) -> dict[str, float | int | str]:
    abs_delta = np.abs(delta)
    return {
        "slice": label,
        "n": int(len(delta)),
        "frac": float(len(delta) / n_total),
        "mean_shift": float(delta.mean()) if len(delta) else float("nan"),
        "mean_abs_shift": float(abs_delta.mean()) if len(delta) else float("nan"),
        "p90_abs_shift": float(np.quantile(abs_delta, 0.90)) if len(delta) else float("nan"),
        "max_abs_shift": float(abs_delta.max()) if len(delta) else float("nan"),
        "n_abs_gt_005": int((abs_delta > 0.05).sum()),
        "n_abs_gt_010": int((abs_delta > 0.10).sum()),
    }


def bad_axis_projection(candidate_delta: np.ndarray, frame: pd.DataFrame) -> float:
    if not BAD_AXIS_PATH.exists():
        return float("nan")
    bad = load_submission(BAD_AXIS_PATH, "bad_pred")
    aligned = frame[["molecule_name", "anchor_pred"]].merge(
        bad[["molecule_name", "bad_pred"]], on="molecule_name"
    )
    bad_delta = aligned["bad_pred"].to_numpy(dtype=np.float64) - aligned[
        "anchor_pred"
    ].to_numpy(dtype=np.float64)
    denom = float(np.dot(bad_delta, bad_delta))
    if denom == 0.0:
        return float("nan")
    return float(np.dot(candidate_delta, bad_delta) / denom)


def as2_slice_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    masks: dict[str, pd.Series] = {
        "all_test": pd.Series(True, index=frame.index),
        "AS1": frame["split"].eq("AS1"),
        "AS2": frame["split"].eq("AS2"),
    }
    if not AS2_RISK_MAP_PATH.exists():
        return masks
    as2 = pd.read_csv(AS2_RISK_MAP_PATH)
    risk_cols = [
        "overall_risk_score",
        "tag_potent_neighbor_low_support",
        "tag_high_lf_saturated",
        "tag_high_lf_but_not_high_pred",
        "tag_member_disagreement",
    ]
    keep_cols = ["molecule_name", *[c for c in risk_cols if c in as2.columns]]
    merged = frame[["molecule_name"]].merge(as2[keep_cols], on="molecule_name", how="left")
    masks["AS2_overall_risk_ge_0p80"] = merged["overall_risk_score"].fillna(0.0) >= 0.80
    for col in keep_cols:
        if not col.startswith("tag_"):
            continue
        masks[f"AS2_{col}"] = merged[col].fillna(False).astype(bool)
    return masks


def score_submission(path: Path, frame: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    candidate = load_submission(path, "candidate_pred")
    merged = frame.merge(candidate[["molecule_name", "candidate_pred"]], on="molecule_name")
    if len(merged) != len(frame):
        raise RuntimeError(f"{path} aligned {len(merged)} of {len(frame)} test rows")
    delta = merged["candidate_pred"].to_numpy(dtype=np.float64) - merged[
        "anchor_pred"
    ].to_numpy(dtype=np.float64)

    as1 = merged[merged["as1_pec50"].notna()].copy()
    as1_metrics = metric_row(
        as1["as1_pec50"].to_numpy(dtype=np.float64),
        as1["candidate_pred"].to_numpy(dtype=np.float64),
    )
    anchor_as1_metrics = metric_row(
        as1["as1_pec50"].to_numpy(dtype=np.float64),
        as1["anchor_pred"].to_numpy(dtype=np.float64),
    )

    summary = {
        "candidate": path.stem,
        "path": repo_relative(path),
        "as1_mae": as1_metrics["mae"],
        "as1_delta_mae_vs_anchor": as1_metrics["mae"] - anchor_as1_metrics["mae"],
        "as1_bias_pred_minus_true": as1_metrics["bias_pred_minus_true"],
        "as1_spearman": as1_metrics["spearman"],
        "test_pearson_vs_anchor": safe_corr(
            merged["anchor_pred"].to_numpy(dtype=np.float64),
            merged["candidate_pred"].to_numpy(dtype=np.float64),
        ),
        "test_mean_abs_shift": float(np.abs(delta).mean()),
        "test_p90_abs_shift": float(np.quantile(np.abs(delta), 0.90)),
        "test_max_abs_shift": float(np.abs(delta).max()),
        "bad_axis_id56_projection": bad_axis_projection(delta, merged),
    }

    shift_rows = []
    for label, mask in as2_slice_masks(merged).items():
        slice_delta = delta[mask.to_numpy(dtype=bool)]
        row = summarize_shift(slice_delta, label, len(merged))
        row["candidate"] = path.stem
        shift_rows.append(row)

    as1["true_bin"] = pd.cut(as1["as1_pec50"], TRUE_BINS, labels=TRUE_BIN_LABELS)
    bin_rows = []
    for label, sub in as1.groupby("true_bin", observed=True):
        metrics = metric_row(
            sub["as1_pec50"].to_numpy(dtype=np.float64),
            sub["candidate_pred"].to_numpy(dtype=np.float64),
        )
        anchor_metrics = metric_row(
            sub["as1_pec50"].to_numpy(dtype=np.float64),
            sub["anchor_pred"].to_numpy(dtype=np.float64),
        )
        bin_rows.append(
            {
                "candidate": path.stem,
                "true_bin": str(label),
                "n": len(sub),
                **metrics,
                "delta_mae_vs_anchor": metrics["mae"] - anchor_metrics["mae"],
            }
        )

    return summary, pd.DataFrame(shift_rows), pd.DataFrame(bin_rows)


def load_train_targets() -> np.ndarray:
    return pd.read_sql("SELECT pec50 FROM train_activity ORDER BY id", get_engine())[
        "pec50"
    ].to_numpy(dtype=np.float64)


def find_experiment_id(name: str) -> int:
    row = pd.read_sql(
        text(
            """
            SELECT e.id, count(o.train_idx) AS n_oof
            FROM experiments e
            JOIN experiment_oof_predictions o ON o.experiment_id = e.id
            WHERE e.name = :name
            GROUP BY e.id
            HAVING count(o.train_idx) = 4140
            ORDER BY e.id DESC
            LIMIT 1
            """
        ),
        get_engine(),
        params={"name": name},
    )
    if row.empty:
        raise RuntimeError(f"missing full OOF experiment: {name}")
    return int(row["id"].iloc[0])


def score_oof_experiment(name: str) -> tuple[dict, pd.DataFrame]:
    exp_id: int | None = None
    try:
        exp_id = find_experiment_id(name)
        pred = load_oof_predictions(exp_id)
        if pred is None or len(pred) != 4140:
            raise RuntimeError(f"{name}: missing 4140 OOF predictions")
    except RuntimeError:
        # Ensemble rows are often represented by weights rather than saved OOF.
        # Reuse the established error_anatomy reconstruction helper for those.
        pred = ea.load_experiment_oof(name, 4140)
    y = load_train_targets()
    metrics = metric_row(y, pred)
    summary = {
        "experiment": name,
        "experiment_id": exp_id if exp_id is not None else -1,
        "oof_mae": metrics["mae"],
        "oof_bias_pred_minus_true": metrics["bias_pred_minus_true"],
        "oof_spearman": metrics["spearman"],
    }

    if not TRAIN_SLICE_PATH.exists():
        return summary, pd.DataFrame()
    slices = pd.read_csv(TRAIN_SLICE_PATH)
    slices["candidate_oof_pred"] = pred
    slices["candidate_error"] = slices["candidate_oof_pred"] - slices["pec50"]
    slices["candidate_abs_error"] = slices["candidate_error"].abs()
    mask_columns = [
        "tag_high_tail_top10",
        "tag_low_tail_top10",
        "tag_mid_ambiguity_top10",
        "tag_potent_neighbor_low_support",
        "tag_member_disagreement_top10",
        "tag_high_lf_saturated",
        "tag_high_lf_but_not_high_pred",
    ]
    rows = []
    masks = {
        "all_train": pd.Series(True, index=slices.index),
        "true_lt3": slices["pec50"] < 3.0,
        "true_gte6": slices["pec50"] >= 6.0,
    }
    for col in mask_columns:
        if col in slices:
            masks[col] = slices[col].astype(bool)
    for label, mask in masks.items():
        sub = slices[mask]
        rest = slices[~mask]
        rows.append(
            {
                "experiment": name,
                "slice": label,
                "n": int(mask.sum()),
                "mae": float(sub["candidate_abs_error"].mean()),
                "rest_mae": float(rest["candidate_abs_error"].mean()) if len(rest) else float("nan"),
                "bias_pred_minus_true": float(sub["candidate_error"].mean()),
                "pred_mean": float(sub["candidate_oof_pred"].mean()),
                "true_mean": float(sub["pec50"].mean()),
            }
        )
    return summary, pd.DataFrame(rows)


def write_report(
    submission_summary: pd.DataFrame,
    shift_summary: pd.DataFrame,
    as1_bin_summary: pd.DataFrame,
    oof_summary: pd.DataFrame,
    oof_slice_summary: pd.DataFrame,
    out_path: Path,
) -> None:
    lines = [
        "# Phase 2 candidate scorer",
        "",
        "This is a diagnostic scorecard. It does not train models, change OOF",
        "predictions, or generate a submission.",
        "",
    ]
    if not submission_summary.empty:
        lines.extend(
            [
                "## Submission CSV summary",
                "",
                submission_summary.to_markdown(index=False, floatfmt=".5f"),
                "",
                "## AS2 shift slices",
                "",
                shift_summary.to_markdown(index=False, floatfmt=".5f"),
                "",
                "## AS1 true-bin replay",
                "",
                as1_bin_summary.to_markdown(index=False, floatfmt=".5f"),
                "",
            ]
        )
    if not oof_summary.empty:
        lines.extend(
            [
                "## OOF experiment summary",
                "",
                oof_summary.to_markdown(index=False, floatfmt=".5f"),
                "",
                "## OOF stress slices",
                "",
                oof_slice_summary.to_markdown(index=False, floatfmt=".5f"),
                "",
            ]
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate",
        action="append",
        type=Path,
        help="Track 1 submission CSV to score. Can be passed multiple times.",
    )
    parser.add_argument(
        "--experiment",
        action="append",
        default=[],
        help="Experiment name with full OOF predictions to score.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help="Directory for scorecard CSV/Markdown outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = args.candidate or DEFAULT_CANDIDATES

    frame = load_anchor_frame()
    submission_rows = []
    shift_frames = []
    bin_frames = []
    for raw_path in candidates:
        path = raw_path if raw_path.is_absolute() else REPO_ROOT / raw_path
        summary, shift, bins = score_submission(path, frame)
        submission_rows.append(summary)
        shift_frames.append(shift)
        bin_frames.append(bins)

    oof_rows = []
    oof_slice_frames = []
    for name in args.experiment:
        summary, slices = score_oof_experiment(name)
        oof_rows.append(summary)
        if not slices.empty:
            oof_slice_frames.append(slices)

    submission_summary = pd.DataFrame(submission_rows).sort_values(
        ["as1_mae", "test_mean_abs_shift"]
    )
    shift_summary = pd.concat(shift_frames, ignore_index=True) if shift_frames else pd.DataFrame()
    as1_bin_summary = pd.concat(bin_frames, ignore_index=True) if bin_frames else pd.DataFrame()
    oof_summary = pd.DataFrame(oof_rows)
    oof_slice_summary = (
        pd.concat(oof_slice_frames, ignore_index=True) if oof_slice_frames else pd.DataFrame()
    )

    submission_summary.to_csv(out_dir / "submission_summary.csv", index=False)
    shift_summary.to_csv(out_dir / "shift_summary.csv", index=False)
    as1_bin_summary.to_csv(out_dir / "as1_bin_summary.csv", index=False)
    oof_summary.to_csv(out_dir / "oof_summary.csv", index=False)
    oof_slice_summary.to_csv(out_dir / "oof_slice_summary.csv", index=False)
    write_report(
        submission_summary,
        shift_summary,
        as1_bin_summary,
        oof_summary,
        oof_slice_summary,
        out_dir / "report.md",
    )
    write_report(
        submission_summary,
        shift_summary,
        as1_bin_summary,
        oof_summary,
        oof_slice_summary,
        DOC_REPORT_PATH,
    )
    print(f"Wrote Phase 2 candidate scorecard to {out_dir}")


if __name__ == "__main__":
    main()

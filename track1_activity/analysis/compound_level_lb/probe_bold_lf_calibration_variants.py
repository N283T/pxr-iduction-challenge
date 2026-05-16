#!/usr/bin/env python
"""Probe bolder log2fc-shaped calibration variants around id57.

This is intentionally exploratory. The current preferred submission remains the
small high-activity lift. Here we test larger lift/tilt/stretch corrections to
see whether any shape looks clearly better before spending the cooldown.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "track1_activity" / "src"
SCRIPT_DIR = REPO_ROOT / "track1_activity" / "scripts"
LOCAL_DIR = Path(__file__).resolve().parent
OOF_DIR = REPO_ROOT / "track1_activity" / "analysis" / "oof_reliability_audit"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(LOCAL_DIR))
sys.path.insert(0, str(OOF_DIR))

from evaluate import compute_metrics  # noqa: E402
from probe_conservative_high_activity_lift import (  # noqa: E402
    ID57_PATH,
    build_frame,
    log2fc_features,
    reconstruct_anchor_proxy,
    split_metric_delta,
    split_registry,
)
from probe_log2fc_gated_top500 import potent46_soft_gate  # noqa: E402
from data import load_test_smiles, load_train_smiles_with_counter  # noqa: E402
from submission_preflight import bad_axis_correlations, load_submission  # noqa: E402

SUB_DIR = REPO_ROOT / "track1_activity" / "submissions"
OUT_DIR = LOCAL_DIR / "outputs" / "bold_lf_calibration_variants"


@dataclass(frozen=True)
class Variant:
    name: str
    train_shape: np.ndarray
    test_shape: np.ndarray
    scale: float


def soft_above(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((values - lo) / max(hi - lo, 1e-9), 0.0, 1.0)


def soft_below(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((hi - values) / max(hi - lo, 1e-9), 0.0, 1.0)


def q_soft_high(
    train_values: np.ndarray, test_values: np.ndarray, q: float
) -> tuple[np.ndarray, np.ndarray]:
    lo = float(np.quantile(train_values, q))
    hi = float(np.quantile(train_values, 0.95))
    return soft_above(train_values, lo, hi), soft_above(test_values, lo, hi)


def q_soft_low(
    train_values: np.ndarray, test_values: np.ndarray, q: float
) -> tuple[np.ndarray, np.ndarray]:
    lo = float(np.quantile(train_values, 0.05))
    hi = float(np.quantile(train_values, q))
    return soft_below(train_values, lo, hi), soft_below(test_values, lo, hi)


def clipped_z(
    train_values: np.ndarray, test_values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mu = float(np.mean(train_values))
    sd = float(np.std(train_values, ddof=1))
    return np.clip((train_values - mu) / sd, -1.0, 1.0), np.clip(
        (test_values - mu) / sd, -1.0, 1.0
    )


def variant_shapes(
    *,
    train_lf: pd.DataFrame,
    test_lf: pd.DataFrame,
    anchor_oof: np.ndarray,
    anchor_test: np.ndarray,
    potent_train: np.ndarray,
    potent_test: np.ndarray,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    shapes: list[tuple[str, np.ndarray, np.ndarray]] = []
    lf_mean_tr = train_lf["lf_mean"].to_numpy(dtype=np.float64)
    lf_mean_te = test_lf["lf_mean"].to_numpy(dtype=np.float64)
    lf_max_tr = train_lf["lf_max"].to_numpy(dtype=np.float64)
    lf_max_te = test_lf["lf_max"].to_numpy(dtype=np.float64)

    for col_name, train_values, test_values in (
        ("lf_mean", lf_mean_tr, lf_mean_te),
        ("lf_max", lf_max_tr, lf_max_te),
    ):
        for q in (0.50, 0.60, 0.70):
            high_tr, high_te = q_soft_high(train_values, test_values, q)
            shapes.append((f"{col_name}_high_q{int(q * 100)}", high_tr, high_te))
            shapes.append(
                (
                    f"potent46_x_{col_name}_high_q{int(q * 100)}",
                    potent_train * high_tr,
                    potent_test * high_te,
                )
            )

        high_tr, high_te = q_soft_high(train_values, test_values, 0.50)
        low_tr, low_te = q_soft_low(train_values, test_values, 0.50)
        shapes.append(
            (f"{col_name}_tilt_highq50_lowq50", high_tr - low_tr, high_te - low_te)
        )

        z_tr, z_te = clipped_z(train_values, test_values)
        shapes.append((f"{col_name}_clipped_z", z_tr, z_te))

    pred_high_tr, pred_high_te = q_soft_high(anchor_oof, anchor_test, 0.60)
    high_lf_tr, high_lf_te = q_soft_high(lf_mean_tr, lf_mean_te, 0.50)
    shapes.append(
        (
            "lf_mean_highq50_x_pred_highq60",
            high_lf_tr * pred_high_tr,
            high_lf_te * pred_high_te,
        )
    )
    shapes.append(
        (
            "lf_mean_highq50_x_pred_highq60_centered",
            high_lf_tr * pred_high_tr - np.mean(high_lf_tr * pred_high_tr),
            high_lf_te * pred_high_te - np.mean(high_lf_te * pred_high_te),
        )
    )
    return shapes


def write_submission(name: str, pred: np.ndarray) -> Path:
    anchor = load_submission(ID57_PATH)
    out = anchor.copy()
    out["pEC50"] = pred
    path = SUB_DIR / f"{name}.csv"
    out.to_csv(path, index=False)
    return path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_df = load_train_smiles_with_counter()
    test_df = load_test_smiles()
    y = train_df["pec50"].to_numpy(dtype=np.float64)
    frame = build_frame()
    registry = split_registry(frame)
    anchor_oof, anchor_test, _anchor_proxy, potent_test = reconstruct_anchor_proxy(
        train_df, test_df, y
    )
    potent_train, _ = potent46_soft_gate(
        train_df, test_df["smiles"].tolist(), threshold=0.40
    )
    train_lf, test_lf = log2fc_features()
    shapes = variant_shapes(
        train_lf=train_lf,
        test_lf=test_lf,
        anchor_oof=anchor_oof,
        anchor_test=anchor_test,
        potent_train=potent_train,
        potent_test=potent_test,
    )

    base = compute_metrics(y, anchor_oof)
    rows: list[dict[str, float | str]] = []
    scales = (0.03, 0.05, 0.08, 0.12, 0.16)
    for shape_name, train_shape, test_shape in shapes:
        for scale in scales:
            cand_oof = anchor_oof + scale * train_shape
            cand_test = anchor_test + scale * test_shape
            delta = cand_test - anchor_test
            metrics = compute_metrics(y, cand_oof)
            row: dict[str, float | str] = {
                "variant": shape_name,
                "scale": scale,
                "full_mae": float(metrics["MAE"]),
                "full_delta_mae": float(metrics["MAE"] - base["MAE"]),
                "full_spearman": float(metrics["Spearman_R"]),
                "full_delta_spearman": float(
                    metrics["Spearman_R"] - base["Spearman_R"]
                ),
                "test_shift_mean": float(delta.mean()),
                "test_abs_shift_mean": float(np.abs(delta).mean()),
                "test_abs_shift_p90": float(np.quantile(np.abs(delta), 0.90)),
                "test_abs_shift_max": float(np.max(np.abs(delta))),
                "test_n_abs_gt_005": int((np.abs(delta) > 0.05).sum()),
                "test_n_abs_gt_010": int((np.abs(delta) > 0.10).sum()),
                "id57_spearman": float(
                    stats.spearmanr(anchor_test, cand_test).statistic
                ),
            }
            row.update(split_metric_delta(y, anchor_oof, cand_oof, registry))
            for axis in bad_axis_correlations(delta):
                row[f"{axis.label}_projection"] = axis.candidate_projection
                row[f"{axis.label}_pearson"] = axis.pearson
            rows.append(row)

    summary = pd.DataFrame(rows).sort_values(
        [
            "public_hybrid_with_y_top513_delta_mae",
            "full_delta_mae",
            "test_abs_shift_p90",
        ]
    )
    summary.to_csv(OUT_DIR / "bold_lf_calibration_summary.csv", index=False)

    bold_safe = summary[
        (summary["full_delta_mae"] <= 0.0005)
        & (summary["public_hybrid_with_y_top513_delta_mae"] <= -0.0040)
        & (summary["public_stress_mean_delta_mae"] <= 0.0000)
        & (summary["test_abs_shift_p90"] <= 0.070)
        & (summary["test_abs_shift_max"] <= 0.160)
        & (summary["test_n_abs_gt_010"] <= 30)
    ].copy()
    bold_safe.to_csv(OUT_DIR / "bold_lf_calibration_safeish.csv", index=False)

    written = []
    shape_by_name = {name: (tr, te) for name, tr, te in shapes}
    for idx, row in enumerate(bold_safe.head(5).itertuples(index=False), start=1):
        _tr, test_shape = shape_by_name[row.variant]
        pred = anchor_test + float(row.scale) * test_shape
        path = write_submission(f"ens_id57_bold_lf_calib_rank{idx}", pred)
        written.append(
            {
                "rank": idx,
                "path": str(path.relative_to(REPO_ROOT)),
                "variant": row.variant,
                "scale": row.scale,
                "full_delta_mae": row.full_delta_mae,
                "public_hybrid_with_y_delta_mae": row.public_hybrid_with_y_top513_delta_mae,
                "test_abs_shift_p90": row.test_abs_shift_p90,
                "test_abs_shift_max": row.test_abs_shift_max,
                "id56_minus_id55_projection": row.id56_minus_id55_projection,
            }
        )
    written_df = pd.DataFrame(written)
    written_df.to_csv(OUT_DIR / "bold_lf_calibration_candidates.csv", index=False)

    cols = [
        "variant",
        "scale",
        "full_delta_mae",
        "public_stress_mean_delta_mae",
        "public_hybrid_with_y_top513_delta_mae",
        "public_log2fc_top513_delta_mae",
        "test_shift_mean",
        "test_abs_shift_p90",
        "test_abs_shift_max",
        "test_n_abs_gt_010",
        "id56_minus_id55_projection",
    ]
    report = [
        "# Bold LF Calibration Variants",
        "",
        f"Anchor: `{ID57_PATH.relative_to(REPO_ROOT)}`",
        f"Anchor OOF proxy MAE: `{base['MAE']:.6f}`",
        "",
        "## Best By High-Y Pseudo Holdout",
        "",
        summary[cols].head(25).to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Bold Safe-Ish",
        "",
        bold_safe[cols].head(25).to_markdown(index=False, floatfmt=".6f")
        if not bold_safe.empty
        else "No bold safe-ish candidates.",
        "",
        "## Candidate CSVs",
        "",
        written_df.to_markdown(index=False, floatfmt=".6f")
        if not written_df.empty
        else "No CSVs written.",
        "",
    ]
    (OUT_DIR / "report.md").write_text("\n".join(report), encoding="utf-8")

    print(f"Anchor proxy MAE={base['MAE']:.6f}")
    print("\n=== Best by high-y pseudo holdout ===")
    print(summary[cols].head(25).to_markdown(index=False, floatfmt=".6f"))
    print("\n=== Bold safe-ish ===")
    print(
        bold_safe[cols].head(25).to_markdown(index=False, floatfmt=".6f")
        if not bold_safe.empty
        else "No bold safe-ish candidates."
    )
    print("\n=== Candidate CSVs ===")
    print(
        written_df.to_markdown(index=False, floatfmt=".6f")
        if not written_df.empty
        else "No CSVs written."
    )
    print(f"\nWrote {OUT_DIR}")


if __name__ == "__main__":
    main()

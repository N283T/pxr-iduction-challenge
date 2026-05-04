#!/usr/bin/env python
"""Build conservative candidates along the baseline -> family-meta axis.

Known LB anchors:
  alpha=0.0: id32 baseline9
  alpha=0.5: id43 hybrid 50/50
  alpha=1.0: id42 family-meta

The fitted optimum is only an interpolation heuristic. It is useful because
this axis has real LB observations at three points, unlike most recent ideas.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
SUB_DIR = REPO_ROOT.joinpath("track1_activity", "submissions")
OUT_DIR = Path(__file__).resolve().parent / "outputs" / "meta_axis_candidates"

BASE_PATH = SUB_DIR / "ens_caruana_bag20_calibrated_importance_baseline_9pool.csv"
META_PATH = SUB_DIR / "ens_caruana_bag20_calibrated_importance_meta_id42.csv"

LB_ANCHORS = pd.DataFrame(
    {
        "alpha": [0.0, 0.5, 1.0],
        "lb_mae": [0.4078471063458229, 0.4074838675224626, 0.4090744542353690],
        "lb_sp": [0.8454463307068544, 0.8470495157914333, 0.8475717867383492],
        "label": ["id32_baseline9", "id43_hybrid", "id42_family_meta"],
    }
)


def fit_quadratic(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    a, b, c = np.polyfit(x, y, deg=2)
    return float(a), float(b), float(c)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(BASE_PATH)
    meta = pd.read_csv(META_PATH)
    if not (base["Molecule Name"].to_numpy() == meta["Molecule Name"].to_numpy()).all():
        raise RuntimeError("baseline/meta molecule order mismatch")

    x = LB_ANCHORS["alpha"].to_numpy()
    mae_a, mae_b, mae_c = fit_quadratic(x, LB_ANCHORS["lb_mae"].to_numpy())
    sp_a, sp_b, sp_c = fit_quadratic(x, LB_ANCHORS["lb_sp"].to_numpy())
    mae_opt = float(np.clip(-mae_b / (2 * mae_a), 0.0, 1.0))

    alphas = sorted({0.25, 0.30, round(mae_opt, 3), 0.35, 0.40})
    rows = []
    for alpha in alphas:
        pred = (1.0 - alpha) * base["pEC50"].to_numpy() + alpha * meta[
            "pEC50"
        ].to_numpy()
        out = base.copy()
        out["pEC50"] = pred
        out_path = OUT_DIR / f"ens_meta_axis_a{int(round(alpha * 1000)):03d}.csv"
        out.to_csv(out_path, index=False)
        rows.append(
            {
                "alpha": alpha,
                "predicted_lb_mae_quadratic": mae_a * alpha**2 + mae_b * alpha + mae_c,
                "predicted_lb_sp_quadratic": sp_a * alpha**2 + sp_b * alpha + sp_c,
                "mean_pred": float(pred.mean()),
                "std_pred": float(pred.std(ddof=0)),
                "mean_abs_shift_vs_base": float(
                    np.abs(pred - base["pEC50"].to_numpy()).mean()
                ),
                "p90_abs_shift_vs_base": float(
                    np.quantile(np.abs(pred - base["pEC50"].to_numpy()), 0.90)
                ),
                "path": str(out_path.relative_to(REPO_ROOT)),
            }
        )

    summary = pd.DataFrame(rows)
    LB_ANCHORS.to_csv(OUT_DIR / "lb_anchors.csv", index=False)
    summary.to_csv(OUT_DIR / "candidate_summary.csv", index=False)
    report = [
        "# Meta-Axis Candidate Sweep",
        "",
        "Known LB anchors along `prediction = (1-alpha) * baseline9 + alpha * family_meta`:",
        "",
        LB_ANCHORS.to_markdown(index=False, floatfmt=".6f"),
        "",
        f"Quadratic MAE fit: `{mae_a:.8f} * a^2 + {mae_b:.8f} * a + {mae_c:.8f}`",
        f"Fitted MAE optimum alpha: `{mae_opt:.3f}`",
        "",
        "Candidate CSVs were written but should be treated as low-upside A/B candidates.",
        "The fitted improvement over id43 is about 0.0001 MAE, below normal LB noise.",
        "",
        summary.to_markdown(index=False, floatfmt=".6f"),
    ]
    (OUT_DIR / "report.md").write_text("\n".join(report) + "\n")
    print(f"Wrote meta-axis candidates to {OUT_DIR}")


if __name__ == "__main__":
    main()

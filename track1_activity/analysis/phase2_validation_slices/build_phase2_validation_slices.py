#!/usr/bin/env -S pixi run python
"""Validate Phase 2 AS2 risk-map slices against train OOF errors.

The AS2 map is unlabeled, so its risk tags need a train-side error check before
they can guide Phase 2 validation design. This script mirrors the main AS2 risk
families on current ensemble OOF predictions and summarizes whether those
slices are actually harder on labeled training folds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "scripts")))
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "analysis", "error_anatomy")))
sys.path.insert(
    0, str(REPO_ROOT.joinpath("track1_activity", "analysis", "phase2_as2_risk_map"))
)

from splits import _morgan_fp_matrix  # noqa: E402

import error_anatomy as ea  # noqa: E402
from build_phase2_as2_risk_map import pct_rank, tanimoto_matrix  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.joinpath("outputs")
REPORT_PATH = REPO_ROOT / "docs" / "track1_explain" / "phase2_validation_slices.md"
AS2_RISK_MAP_PATH = (
    REPO_ROOT / "track1_activity" / "analysis" / "phase2_as2_risk_map" / "outputs" / "as2_risk_map.csv"
)
LF_PATH = (
    REPO_ROOT
    / "data"
    / "chemprop_pretrain_log2fc_predictions_optuna_trial10_seed5ens.parquet"
)

TRAIN_BINS = [-np.inf, 3.0, 4.0, 5.0, 6.0, np.inf]
TRAIN_BIN_LABELS = ["lt3", "3to4", "4to5", "5to6", "gte6"]


def load_train_proxy_frame() -> pd.DataFrame:
    df, _member_names = ea.build_residual_frame()
    # error convention here follows leaderboard replay: prediction - true.
    df["error"] = df["pred"] - df["pec50"]
    df["abs_error"] = df["error"].abs()

    lf = pd.read_parquet(LF_PATH).loc[df["compound_id"].astype(int).tolist()].copy()
    lf["lf_mean"] = 0.5 * (lf["log2fc_8p25_pred"] + lf["log2fc_33_pred"])
    lf = lf.reset_index()
    df = df.merge(lf, on="compound_id", how="left")
    return add_train_support(df)


def add_train_support(df: pd.DataFrame) -> pd.DataFrame:
    fps = _morgan_fp_matrix(df["smiles"].tolist())
    sim = tanimoto_matrix(fps, fps)
    np.fill_diagonal(sim, 0.0)

    y = df["pec50"].to_numpy(dtype=np.float64)
    nn_idx = sim.argmax(axis=1)
    out = df.copy()
    out["nn_train_tanimoto"] = sim[np.arange(len(out)), nn_idx]
    out["nn_train_pec50"] = y[nn_idx]

    potent_mask = y >= 6.0
    weak_mask = y < 3.0
    out["nn_potent_tanimoto"] = sim[:, potent_mask].max(axis=1)
    out["nn_weak_tanimoto"] = sim[:, weak_mask].max(axis=1)

    pred_bins = pd.cut(out["pred"], TRAIN_BINS, labels=TRAIN_BIN_LABELS)
    true_bins = pd.cut(out["pec50"], TRAIN_BINS, labels=TRAIN_BIN_LABELS)
    train_bins = pd.cut(y, TRAIN_BINS, labels=TRAIN_BIN_LABELS)
    out["true_bin"] = true_bins.astype("object")

    for threshold in (0.40, 0.50, 0.60):
        near = sim >= threshold
        out[f"train_support_n_ge_{threshold:.2f}"] = near.sum(axis=1)
        out[f"train_support_weak_n_ge_{threshold:.2f}"] = near[:, weak_mask].sum(axis=1)
        out[f"train_support_potent_n_ge_{threshold:.2f}"] = near[:, potent_mask].sum(axis=1)
        same_counts = []
        for i, label in enumerate(pred_bins):
            same_counts.append(int((near[i] & (train_bins == label)).sum()))
        out[f"train_support_pred_bin_n_ge_{threshold:.2f}"] = same_counts
    return out


def add_slice_tags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    pred_pct = pct_rank(out["pred"])
    lf_pct = pct_rank(out["lf_mean"])
    low_lf_high_pred_gap = pred_pct - lf_pct
    low_support_shortage = 1.0 - pct_rank(out["train_support_pred_bin_n_ge_0.50"])

    out["low_tail_risk_score"] = (
        0.30 * pred_pct
        + 0.25 * pct_rank(out["nn_potent_tanimoto"])
        + 0.25 * pct_rank(low_lf_high_pred_gap)
        + 0.20 * low_support_shortage
    )
    out["high_tail_risk_score"] = (
        0.40 * pred_pct
        + 0.25 * lf_pct
        + 0.15 * pct_rank(out["member_std"])
        + 0.20 * pct_rank(out["nn_potent_tanimoto"])
    )
    out["mid_ambiguity_score"] = (
        0.35 * pct_rank(out["member_std"])
        + 0.25 * pct_rank(out["abs_family_gap"])
        + 0.20 * pct_rank(out["nn_potent_tanimoto"])
        + 0.20 * low_support_shortage
    )
    out["overall_risk_score"] = out[
        ["low_tail_risk_score", "high_tail_risk_score", "mid_ambiguity_score"]
    ].max(axis=1)

    out["tag_high_tail_top10"] = out["high_tail_risk_score"] >= out[
        "high_tail_risk_score"
    ].quantile(0.90)
    out["tag_low_tail_top10"] = out["low_tail_risk_score"] >= out[
        "low_tail_risk_score"
    ].quantile(0.90)
    out["tag_mid_ambiguity_top10"] = out["mid_ambiguity_score"] >= out[
        "mid_ambiguity_score"
    ].quantile(0.90)
    out["tag_potent_neighbor_low_support"] = (out["nn_potent_tanimoto"] >= 0.50) & (
        out["train_support_pred_bin_n_ge_0.50"] == 0
    )
    out["tag_member_disagreement_top10"] = out["member_std"] >= out[
        "member_std"
    ].quantile(0.90)
    out["tag_high_lf_saturated"] = (out["lf_mean"] >= out["lf_mean"].quantile(0.75)) & (
        out["pred"] >= out["pred"].quantile(0.85)
    )
    out["tag_high_lf_but_not_high_pred"] = (
        out["lf_mean"] >= out["lf_mean"].quantile(0.75)
    ) & (out["pred"] < out["pred"].quantile(0.85))
    out["tag_count"] = out[
        [
            "tag_high_tail_top10",
            "tag_low_tail_top10",
            "tag_mid_ambiguity_top10",
            "tag_potent_neighbor_low_support",
            "tag_member_disagreement_top10",
            "tag_high_lf_saturated",
            "tag_high_lf_but_not_high_pred",
        ]
    ].sum(axis=1)
    return out


def summarize_mask(df: pd.DataFrame, name: str, mask: pd.Series) -> dict[str, float | int | str]:
    sub = df.loc[mask]
    rest = df.loc[~mask]
    return {
        "slice": name,
        "n": int(mask.sum()),
        "frac": float(mask.mean()),
        "mae": float(sub["abs_error"].mean()),
        "rest_mae": float(rest["abs_error"].mean()),
        "delta_mae_vs_rest": float(sub["abs_error"].mean() - rest["abs_error"].mean()),
        "bias_pred_minus_true": float(sub["error"].mean()),
        "rest_bias_pred_minus_true": float(rest["error"].mean()),
        "true_mean": float(sub["pec50"].mean()),
        "pred_mean": float(sub["pred"].mean()),
        "lf_mean": float(sub["lf_mean"].mean()),
        "member_std_mean": float(sub["member_std"].mean()),
        "nn_potent_tanimoto_mean": float(sub["nn_potent_tanimoto"].mean()),
        "support_pred_bin_ge_0.50_mean": float(
            sub["train_support_pred_bin_n_ge_0.50"].mean()
        ),
    }


def build_slice_summary(df: pd.DataFrame) -> pd.DataFrame:
    masks = {
        "all_train_oof": pd.Series(True, index=df.index),
        "true_lt3": df["pec50"] < 3.0,
        "true_gte6": df["pec50"] >= 6.0,
        "tag_high_tail_top10": df["tag_high_tail_top10"],
        "tag_low_tail_top10": df["tag_low_tail_top10"],
        "tag_mid_ambiguity_top10": df["tag_mid_ambiguity_top10"],
        "tag_potent_neighbor_low_support": df["tag_potent_neighbor_low_support"],
        "tag_member_disagreement_top10": df["tag_member_disagreement_top10"],
        "tag_high_lf_saturated": df["tag_high_lf_saturated"],
        "tag_high_lf_but_not_high_pred": df["tag_high_lf_but_not_high_pred"],
        "tag_count_ge2": df["tag_count"] >= 2,
        "overall_risk_top10": df["overall_risk_score"]
        >= df["overall_risk_score"].quantile(0.90),
    }
    return pd.DataFrame(
        [summarize_mask(df, name, mask.astype(bool)) for name, mask in masks.items()]
    )


def build_true_bin_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("true_bin", observed=True)
        .agg(
            n=("abs_error", "size"),
            mae=("abs_error", "mean"),
            bias_pred_minus_true=("error", "mean"),
            true_mean=("pec50", "mean"),
            pred_mean=("pred", "mean"),
            lf_mean=("lf_mean", "mean"),
            high_tail_risk_mean=("high_tail_risk_score", "mean"),
            low_tail_risk_mean=("low_tail_risk_score", "mean"),
            mid_ambiguity_mean=("mid_ambiguity_score", "mean"),
        )
        .reset_index()
    )


def compare_as2_counts() -> pd.DataFrame:
    if not AS2_RISK_MAP_PATH.exists():
        return pd.DataFrame()
    as2 = pd.read_csv(AS2_RISK_MAP_PATH)
    rows = []
    for col in [
        "tag_potent_neighbor_low_support",
        "tag_high_lf_saturated",
        "tag_member_disagreement",
        "tag_high_lf_but_not_high_pred",
    ]:
        if col in as2:
            rows.append(
                {"as2_tag": col, "as2_n": int(as2[col].sum()), "as2_frac": float(as2[col].mean())}
            )
    rows.append(
        {
            "as2_tag": "overall_risk_score_ge_0.80",
            "as2_n": int((as2["overall_risk_score"] >= 0.80).sum()),
            "as2_frac": float((as2["overall_risk_score"] >= 0.80).mean()),
        }
    )
    return pd.DataFrame(rows)


def write_report(
    slice_summary: pd.DataFrame,
    true_bin_summary: pd.DataFrame,
    as2_counts: pd.DataFrame,
) -> None:
    sorted_slices = slice_summary.sort_values("delta_mae_vs_rest", ascending=False)
    top = sorted_slices.head(8)
    by_slice = slice_summary.set_index("slice")
    true_high = by_slice.loc["true_gte6"]
    true_low = by_slice.loc["true_lt3"]
    member_disagreement = by_slice.loc["tag_member_disagreement_top10"]
    potent_low_support = by_slice.loc["tag_potent_neighbor_low_support"]
    high_lf_saturated = by_slice.loc["tag_high_lf_saturated"]
    lines = [
        "# Phase 2 validation-slice OOF audit",
        "",
        "This report checks whether AS2 risk-map style slices are actually hard on",
        "labeled train OOF predictions. It uses current `ens_caruana_bag20` OOF as",
        "the train-side proxy, not the exact id55 post-hoc CSV gate.",
        "",
        "## Short read",
        "",
        f"- True `>=6` train compounds are the hardest OOF slice: MAE {true_high['mae']:.4f},",
        f"  bias {true_high['bias_pred_minus_true']:.4f}. This confirms high-tail",
        "  compression as a validation stress test.",
        f"- True `<3` train compounds are also hard: MAE {true_low['mae']:.4f},",
        f"  bias {true_low['bias_pred_minus_true']:.4f}. Low-tail overprediction",
        "  remains a real failure shape.",
        f"- Member-disagreement top10% is hard: MAE {member_disagreement['mae']:.4f}",
        f"  vs rest {member_disagreement['rest_mae']:.4f}. This is the strongest",
        "  general unlabeled risk flag in this pass.",
        f"- Potent-neighbor/low-support is very small on train (n={int(potent_low_support['n'])})",
        f"  but high-error: MAE {potent_low_support['mae']:.4f}. AS2 has many more",
        "  compounds with this tag, so it deserves manual review.",
        "- High-LF/high-prediction saturation is not hard on train OOF by itself",
        f"  (MAE {high_lf_saturated['mae']:.4f}); it should not be read as an automatic",
        "  upward-shift instruction.",
        "- AS2 map tags should be treated as slice definitions first, not as",
        "  prediction-shift instructions.",
        "- Compare any future calibrator against these slices before spending a",
        "  final submission.",
        "",
        "## Hardest train OOF slices",
        "",
        top.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## All slice summary",
        "",
        slice_summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## True-bin OOF summary",
        "",
        true_bin_summary.to_markdown(index=False, floatfmt=".4f"),
    ]
    if not as2_counts.empty:
        lines.extend(
            [
                "",
                "## AS2 map tag counts for comparison",
                "",
                as2_counts.to_markdown(index=False, floatfmt=".4f"),
            ]
        )
    lines.extend(
        [
            "",
            "## Generated files",
            "",
            "- `track1_activity/analysis/phase2_validation_slices/outputs/train_oof_validation_slices.csv`",
            "- `track1_activity/analysis/phase2_validation_slices/outputs/slice_summary.csv`",
            "- `track1_activity/analysis/phase2_validation_slices/outputs/true_bin_summary.csv`",
            "- `track1_activity/analysis/phase2_validation_slices/outputs/as2_tag_count_reference.csv`",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_train_proxy_frame()
    df = add_slice_tags(df)
    slice_summary = build_slice_summary(df)
    true_bin_summary = build_true_bin_summary(df)
    as2_counts = compare_as2_counts()

    df.to_csv(OUT_DIR / "train_oof_validation_slices.csv", index=False)
    slice_summary.to_csv(OUT_DIR / "slice_summary.csv", index=False)
    true_bin_summary.to_csv(OUT_DIR / "true_bin_summary.csv", index=False)
    as2_counts.to_csv(OUT_DIR / "as2_tag_count_reference.csv", index=False)
    write_report(slice_summary, true_bin_summary, as2_counts)

    print(f"Wrote train OOF slice audit to {OUT_DIR}")
    print(f"Wrote report to {REPORT_PATH}")


if __name__ == "__main__":
    main()

#!/usr/bin/env -S pixi run python
"""Build an unlabeled AS2 risk map for Track 1 Phase 2.

This analysis does not generate a new submission and does not fit to AS2
labels. It summarizes where the current id55 anchor may be extrapolating on
the still-blinded Analog Set 2 compounds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "scripts")))

from data import get_engine, load_train_smiles_target  # noqa: E402
from splits import _morgan_fp_matrix  # noqa: E402

import run_ensemble  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.joinpath("outputs")
ASSET_DIR = REPO_ROOT / "docs" / "track1_explain" / "assets" / "phase2_as2_risk_map"
REPORT_PATH = REPO_ROOT / "docs" / "track1_explain" / "phase2_as2_risk_map.md"
SUBMISSION_DIR = REPO_ROOT / "track1_activity" / "submissions"
ANCHOR_PATH = SUBMISSION_DIR / "ens_id51_top500_potent46_t40_soft_g35.csv"
LF_PATH = (
    REPO_ROOT
    / "data"
    / "chemprop_pretrain_log2fc_predictions_optuna_trial10_seed5ens.parquet"
)

TRAIN_BINS = [-np.inf, 3.0, 4.0, 5.0, 6.0, np.inf]
TRAIN_BIN_LABELS = ["lt3", "3to4", "4to5", "5to6", "gte6"]


def savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def pct_rank(values: pd.Series) -> pd.Series:
    """Return stable percentile ranks in [0, 1]."""
    return values.rank(method="average", pct=True).fillna(0.0)


def load_test_frame() -> pd.DataFrame:
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


def load_submission(path: Path, column: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.rename(columns={"Molecule Name": "molecule_name", "pEC50": column})[
        ["molecule_name", column]
    ]


def add_anchor_and_lf(df: pd.DataFrame) -> pd.DataFrame:
    out = df.merge(load_submission(ANCHOR_PATH, "pred_id55"), on="molecule_name")
    lf = pd.read_parquet(LF_PATH).loc[out["compound_id"].astype(int).tolist()].copy()
    lf["lf_mean"] = 0.5 * (lf["log2fc_8p25_pred"] + lf["log2fc_33_pred"])
    lf = lf.reset_index()
    return out.merge(lf, on="compound_id", how="left")


def add_member_context(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    member_names: list[str] = []
    missing: list[str] = []
    matrices: list[np.ndarray] = []
    out = df.copy()

    for name in run_ensemble.ENSEMBLE_MODELS:
        path = SUBMISSION_DIR / f"{name}.csv"
        if not path.exists():
            missing.append(name)
            continue
        col = f"member__{name}"
        pred = load_submission(path, col)
        out = out.merge(pred, on="molecule_name", how="left")
        member_names.append(name)
        matrices.append(out[col].to_numpy(dtype=np.float64))

    if not matrices:
        raise RuntimeError("No production member submission CSVs were found.")

    mat = np.column_stack(matrices)
    out["member_mean"] = mat.mean(axis=1)
    out["member_std"] = mat.std(axis=1)
    out["member_range"] = mat.max(axis=1) - mat.min(axis=1)
    out["member_min"] = mat.min(axis=1)
    out["member_max"] = mat.max(axis=1)

    family_mask = np.array(
        [
            ("chemprop" in name) or ("cheme_2d_full_boltz_log2fc_pred" in name)
            for name in member_names
        ],
        dtype=bool,
    )
    if family_mask.any() and (~family_mask).any():
        out["chemprop_family_mean"] = mat[:, family_mask].mean(axis=1)
        out["non_chemprop_mean"] = mat[:, ~family_mask].mean(axis=1)
        out["family_gap"] = out["chemprop_family_mean"] - out["non_chemprop_mean"]
        out["abs_family_gap"] = out["family_gap"].abs()
    else:
        out["family_gap"] = 0.0
        out["abs_family_gap"] = 0.0

    return out, member_names, missing


def tanimoto_matrix(query_fp: np.ndarray, ref_fp: np.ndarray) -> np.ndarray:
    query = query_fp.astype(bool)
    ref = ref_fp.astype(bool)
    inter = query.astype(np.uint16) @ ref.astype(np.uint16).T
    union = query.sum(axis=1, keepdims=True) + ref.sum(axis=1, keepdims=True).T - inter
    return np.divide(
        inter,
        union,
        out=np.zeros_like(inter, dtype=np.float32),
        where=union > 0,
    )


def add_train_support(df: pd.DataFrame) -> pd.DataFrame:
    train = load_train_smiles_target()
    train_fp = _morgan_fp_matrix(train["smiles"].tolist())
    test_fp = _morgan_fp_matrix(df["smiles"].tolist())
    sim = tanimoto_matrix(test_fp, train_fp)

    train_y = train["pec50"].to_numpy(dtype=np.float64)
    nn_idx = sim.argmax(axis=1)
    out = df.copy()
    out["nn_train_tanimoto"] = sim[np.arange(len(out)), nn_idx]
    out["nn_train_pec50"] = train_y[nn_idx]

    potent_mask = train_y >= 6.0
    weak_mask = train_y < 3.0
    out["nn_potent_tanimoto"] = sim[:, potent_mask].max(axis=1)
    out["nn_weak_tanimoto"] = sim[:, weak_mask].max(axis=1)

    for threshold in (0.40, 0.50, 0.60):
        near = sim >= threshold
        out[f"train_support_n_ge_{threshold:.2f}"] = near.sum(axis=1)
        out[f"train_support_weak_n_ge_{threshold:.2f}"] = (
            near[:, weak_mask].sum(axis=1)
        )
        out[f"train_support_potent_n_ge_{threshold:.2f}"] = (
            near[:, potent_mask].sum(axis=1)
        )

    train_bins = pd.cut(train_y, TRAIN_BINS, labels=TRAIN_BIN_LABELS)
    pred_bins = pd.cut(out["pred_id55"], TRAIN_BINS, labels=TRAIN_BIN_LABELS)
    for threshold in (0.40, 0.50):
        same_counts = []
        near = sim >= threshold
        for i, label in enumerate(pred_bins):
            same_counts.append(int((near[i] & (train_bins == label)).sum()))
        out[f"train_support_pred_bin_n_ge_{threshold:.2f}"] = same_counts

    return out


def add_as1_case_similarity(df: pd.DataFrame) -> pd.DataFrame:
    as1 = df[df["split"] == "AS1"].copy()
    if as1.empty:
        raise RuntimeError("AS1 rows are required to compute case similarity.")
    as1["as1_error"] = as1["pred_id55"] - as1["as1_pec50"]
    as1["as1_abs_error"] = as1["as1_error"].abs()

    low_cases = as1[(as1["as1_pec50"] < 3.0) & (as1["as1_error"] > 0.75)]
    high_cases = as1[(as1["as1_pec50"] >= 6.0) & (as1["as1_error"] < -0.40)]
    mid_cases = as1[
        (as1["as1_pec50"] >= 3.0)
        & (as1["as1_pec50"] < 4.0)
        & (as1["as1_abs_error"] > 0.50)
    ]

    test_fp = _morgan_fp_matrix(df["smiles"].tolist())
    out = df.copy()
    case_sets = {
        "as1_low_overpred": low_cases.index.to_numpy(),
        "as1_high_underpred": high_cases.index.to_numpy(),
        "as1_mid_3to4_large_error": mid_cases.index.to_numpy(),
    }
    for name, idx in case_sets.items():
        if len(idx) == 0:
            out[f"max_sim_to_{name}"] = 0.0
            continue
        sim = tanimoto_matrix(test_fp, test_fp[idx])
        out[f"max_sim_to_{name}"] = sim.max(axis=1)
    return out


def add_risk_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["split"] = np.where(out["as1_pec50"].notna(), "AS1", "AS2")

    pred_pct = pct_rank(out["pred_id55"])
    lf_pct = pct_rank(out["lf_mean"])
    low_lf_high_pred_gap = pred_pct - lf_pct
    low_support_shortage = 1.0 - pct_rank(out["train_support_pred_bin_n_ge_0.50"])

    out["low_tail_risk_score"] = (
        0.25 * pred_pct
        + 0.20 * pct_rank(out["nn_potent_tanimoto"])
        + 0.20 * pct_rank(low_lf_high_pred_gap)
        + 0.20 * pct_rank(out["max_sim_to_as1_low_overpred"])
        + 0.15 * low_support_shortage
    )
    out["high_tail_risk_score"] = (
        0.25 * pred_pct
        + 0.20 * lf_pct
        + 0.20 * pct_rank(out["member_max"])
        + 0.20 * pct_rank(out["max_sim_to_as1_high_underpred"])
        + 0.15 * pct_rank(out["nn_potent_tanimoto"])
    )
    out["mid_3to4_ambiguity_score"] = (
        0.30 * pct_rank(out["member_std"])
        + 0.20 * pct_rank(out["abs_family_gap"])
        + 0.20 * pct_rank(out["max_sim_to_as1_mid_3to4_large_error"])
        + 0.15 * pct_rank(out["nn_potent_tanimoto"])
        + 0.15 * low_support_shortage
    )
    out["overall_risk_score"] = out[
        ["low_tail_risk_score", "high_tail_risk_score", "mid_3to4_ambiguity_score"]
    ].max(axis=1)

    as1 = out[out["split"] == "AS1"]
    lf_low = float(as1["lf_mean"].quantile(0.25))
    lf_high = float(as1["lf_mean"].quantile(0.75))
    pred_mid = float(as1["pred_id55"].quantile(0.50))
    pred_high = float(as1["pred_id55"].quantile(0.85))
    std_high = float(as1["member_std"].quantile(0.90))

    tag_columns = {
        "tag_low_lf_high_pred": (out["lf_mean"] <= lf_low)
        & (out["pred_id55"] >= pred_mid),
        "tag_high_lf_saturated": (out["lf_mean"] >= lf_high)
        & (out["pred_id55"] >= pred_high),
        "tag_high_lf_but_not_high_pred": (out["lf_mean"] >= lf_high)
        & (out["pred_id55"] < pred_high),
        "tag_potent_neighbor_low_support": (out["nn_potent_tanimoto"] >= 0.50)
        & (out["train_support_pred_bin_n_ge_0.50"] == 0),
        "tag_member_disagreement": out["member_std"] >= std_high,
        "tag_as1_low_case_like": out["max_sim_to_as1_low_overpred"] >= 0.50,
        "tag_as1_high_case_like": out["max_sim_to_as1_high_underpred"] >= 0.50,
        "tag_as1_mid_case_like": out["max_sim_to_as1_mid_3to4_large_error"] >= 0.50,
    }
    for col, values in tag_columns.items():
        out[col] = values
    out["tag_count"] = out[list(tag_columns)].sum(axis=1)
    return out


def build_split_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = [
        "pred_id55",
        "lf_mean",
        "member_std",
        "nn_train_tanimoto",
        "nn_potent_tanimoto",
        "nn_weak_tanimoto",
        "train_support_pred_bin_n_ge_0.50",
        "low_tail_risk_score",
        "high_tail_risk_score",
        "mid_3to4_ambiguity_score",
        "overall_risk_score",
    ]
    for split, sub in df.groupby("split"):
        row = {"split": split, "n": len(sub)}
        for col in metrics:
            row[f"{col}_mean"] = float(sub[col].mean())
            row[f"{col}_p90"] = float(sub[col].quantile(0.90))
            row[f"{col}_max"] = float(sub[col].max())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("split")


def plot_maps(df: pd.DataFrame) -> None:
    colors = {"AS1": "#4c78a8", "AS2": "#f58518"}
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    for split, sub in df.groupby("split"):
        ax.scatter(
            sub["lf_mean"],
            sub["pred_id55"],
            s=26,
            alpha=0.78,
            label=f"{split} (n={len(sub)})",
            color=colors.get(split, "#777777"),
            linewidths=0,
        )
    ax.set_xlabel("Predicted log2fc mean")
    ax.set_ylabel("id55 pEC50 prediction")
    ax.set_title("AS1 and AS2 in LF-vs-anchor prediction space")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    savefig(fig, ASSET_DIR / "as1_as2_lf_vs_id55.png")

    as2 = df[df["split"] == "AS2"]
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    sc = ax.scatter(
        as2["lf_mean"],
        as2["pred_id55"],
        c=as2["overall_risk_score"],
        cmap="magma_r",
        s=34,
        alpha=0.9,
        linewidths=0,
    )
    ax.set_xlabel("Predicted log2fc mean")
    ax.set_ylabel("id55 pEC50 prediction")
    ax.set_title("AS2 unlabeled risk map")
    ax.grid(alpha=0.22)
    fig.colorbar(sc, ax=ax, label="overall risk score")
    savefig(fig, ASSET_DIR / "as2_risk_map_scatter.png")


def write_report(
    df: pd.DataFrame,
    split_summary: pd.DataFrame,
    member_names: list[str],
    missing_members: list[str],
) -> None:
    as2 = df[df["split"] == "AS2"].sort_values("overall_risk_score", ascending=False)
    top_cols = [
        "molecule_name",
        "pred_id55",
        "lf_mean",
        "member_std",
        "nn_train_tanimoto",
        "nn_train_pec50",
        "nn_potent_tanimoto",
        "train_support_pred_bin_n_ge_0.50",
        "max_sim_to_as1_low_overpred",
        "max_sim_to_as1_high_underpred",
        "low_tail_risk_score",
        "high_tail_risk_score",
        "mid_3to4_ambiguity_score",
        "overall_risk_score",
        "tag_count",
    ]
    top = as2[top_cols].head(20)
    tag_cols = [c for c in df.columns if c.startswith("tag_") and c != "tag_count"]
    tag_counts = (
        as2[tag_cols]
        .sum()
        .rename_axis("tag")
        .reset_index(name="as2_count")
        .sort_values("as2_count", ascending=False)
    )
    as1 = df[df["split"] == "AS1"]
    pred_delta = as2["pred_id55"].mean() - as1["pred_id55"].mean()
    lf_delta = as2["lf_mean"].mean() - as1["lf_mean"].mean()
    high_risk_top_n = int((as2["overall_risk_score"] >= 0.80).sum())
    as1_case_like_n = int(
        as2[
            [
                "tag_as1_low_case_like",
                "tag_as1_high_case_like",
                "tag_as1_mid_case_like",
            ]
        ]
        .any(axis=1)
        .sum()
    )

    lines = [
        "# Phase 2 AS2 unlabeled risk map",
        "",
        "Built from existing predictions and proxy features only. This does not fit",
        "AS2 labels, does not generate a new submission, and should be read as a",
        "triage map for final-evaluation risk rather than a leaderboard feedback",
        "loop.",
        "",
        "## Inputs",
        "",
        f"- Anchor: `{ANCHOR_PATH.relative_to(REPO_ROOT)}`.",
        f"- Production member CSVs found: {len(member_names)}.",
        f"- Missing production member CSVs: {len(missing_members)}.",
        "- LF proxy: chemprop optuna trial10 seed5ens predicted `log2_fc` parquet.",
        "- Chemistry support: Morgan nearest-neighbor context against train activity.",
        "- AS1 case anchors: released AS1 low-tail overpredictions, high-tail",
        "  underpredictions, and 3-4 large-error cases.",
        "",
        "## Split summary",
        "",
        split_summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Short read",
        "",
        f"- AS2 id55 predictions are {pred_delta:+.4f} pEC50 higher on average than AS1.",
        f"- AS2 LF mean is {lf_delta:+.4f} higher than AS1.",
        f"- AS2 compounds with overall risk score >= 0.80: {high_risk_top_n}.",
        f"- AS2 compounds directly similar to tagged AS1 miss sets at Tanimoto >= 0.50: {as1_case_like_n}.",
        "- The highest-ranked AS2 rows are mostly high-prediction/high-LF or",
        "  potent-neighbor/low-support cases. This points more toward high-tail",
        "  saturation risk than a clean replay of the AS1 low-tail cliff cases.",
        "",
        "## AS2 tag counts",
        "",
        tag_counts.to_markdown(index=False),
        "",
        "## Top AS2 compounds by overall triage score",
        "",
        top.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Figures",
        "",
        "![AS1 and AS2 LF-vs-anchor space](assets/phase2_as2_risk_map/as1_as2_lf_vs_id55.png)",
        "",
        "![AS2 unlabeled risk map](assets/phase2_as2_risk_map/as2_risk_map_scatter.png)",
        "",
        "## Interpretation guardrails",
        "",
        "- High risk is not an instruction to shift a prediction. It marks compounds",
        "  where the anchor may be extrapolating or where Phase 1 AS1 failure modes",
        "  have nearby analogs.",
        "- The risk scores are rank-based diagnostics, not calibrated probabilities.",
        "- AS2 is still blinded at compound level and is not available as a",
        "  live leaderboard feedback target during Phase 2.",
        "- The 2026-05-28 interim leaderboard snapshot includes AS1+AS2 full-test",
        "  scoring for each team's latest Phase 1 submission. It is useful as a",
        "  team/submission-level sanity check, but it does not reveal which AS2",
        "  compounds drove the score.",
        "",
        "## Generated files",
        "",
        "- `track1_activity/analysis/phase2_as2_risk_map/outputs/all_test_risk_map.csv`",
        "- `track1_activity/analysis/phase2_as2_risk_map/outputs/as2_risk_map.csv`",
        "- `track1_activity/analysis/phase2_as2_risk_map/outputs/split_summary.csv`",
        "- `track1_activity/analysis/phase2_as2_risk_map/outputs/as2_tag_counts.csv`",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    df = load_test_frame()
    df = add_anchor_and_lf(df)
    df, member_names, missing_members = add_member_context(df)
    df = add_train_support(df)
    df["split"] = np.where(df["as1_pec50"].notna(), "AS1", "AS2")
    df = add_as1_case_similarity(df)
    df = add_risk_scores(df)

    split_summary = build_split_summary(df)
    as2 = df[df["split"] == "AS2"].sort_values("overall_risk_score", ascending=False)
    tag_cols = [c for c in df.columns if c.startswith("tag_") and c != "tag_count"]
    tag_counts = (
        as2[tag_cols]
        .sum()
        .rename_axis("tag")
        .reset_index(name="as2_count")
        .sort_values("as2_count", ascending=False)
    )

    df.to_csv(OUT_DIR / "all_test_risk_map.csv", index=False)
    as2.to_csv(OUT_DIR / "as2_risk_map.csv", index=False)
    split_summary.to_csv(OUT_DIR / "split_summary.csv", index=False)
    tag_counts.to_csv(OUT_DIR / "as2_tag_counts.csv", index=False)
    plot_maps(df)
    write_report(df, split_summary, member_names, missing_members)

    print(f"Wrote {len(as2)} AS2 rows to {OUT_DIR / 'as2_risk_map.csv'}")
    print(f"Wrote report to {REPORT_PATH}")


if __name__ == "__main__":
    main()

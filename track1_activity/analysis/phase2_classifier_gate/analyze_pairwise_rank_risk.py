#!/usr/bin/env -S pixi run python
"""Audit pairwise assay-rank scores before using them as sparse gates."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_ROOT = Path(__file__).resolve().parent / "outputs"
PAIR_ROOT = OUT_ROOT / "pairwise_assay_rank"
RISK_MAP_PATH = (
    REPO_ROOT
    / "track1_activity"
    / "analysis"
    / "phase2_as2_risk_map"
    / "outputs"
    / "all_test_risk_map.csv"
)
MULTICLASS_TOP100 = (
    OUT_ROOT
    / "tabpfn_cheme_2d_full_boltz_log2fc_pred_seed10ens_pred_htchem_top100_v3_ne8_t0p9_balanced"
)
BINARY_TOP100 = (
    OUT_ROOT
    / "binary_gte6_tabpfn_cheme_2d_full_boltz_log2fc_pred_seed10ens_pred_htchem_top100_v3_ne8_t0p9_balanced"
)
AUDIT_DIR = PAIR_ROOT / "risk_audit"

CONFIGS = {
    "activation_aux": PAIR_ROOT / "activation_chembl_htchem_single_conc_mpa1500_top64",
    "allpxr_aux": PAIR_ROOT / "all_pxr_chembl_htchem_single_conc_mpa1500_top64",
}
SCORE_COLS = [
    "pairrank_all",
    "pairrank_chembl",
    "pairrank_htchem",
    "pairrank_single_conc",
]


def load_joined(config_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    pool = pd.read_csv(config_dir / "pool_pairrank_scores.csv")
    test = pd.read_csv(config_dir / "test_pairrank_scores.csv")
    risk_cols = [
        "test_id",
        "compound_id",
        "molecule_name",
        "pred_id55",
        "lf_mean",
        "log2fc_8p25_pred",
        "log2fc_33_pred",
        "member_std",
        "member_range",
        "nn_train_tanimoto",
        "nn_train_pec50",
        "nn_potent_tanimoto",
        "train_support_n_ge_0.50",
        "train_support_potent_n_ge_0.50",
        "pred_htchem",
        "nn_htchem_tanimoto",
        "nn_htchem_corrected_pec50",
        "low_tail_risk_score",
        "high_tail_risk_score",
        "tag_count",
    ]
    risk = pd.read_csv(RISK_MAP_PATH)[risk_cols]
    multi = pd.read_csv(MULTICLASS_TOP100 / "test_class_probabilities.csv")[
        ["test_id", "cv_p_gte6", "final_p_gte6", "final_pred_bin"]
    ]
    binary = pd.read_csv(BINARY_TOP100 / "test_binary_scores.csv")[
        ["test_id", "cv_score", "final_score"]
    ].rename(columns={"cv_score": "binary_cv_gte6", "final_score": "binary_final_gte6"})
    test = test.merge(risk, on=["test_id", "compound_id", "molecule_name"], how="left")
    test = test.merge(multi, on="test_id", how="left")
    test = test.merge(binary, on="test_id", how="left")
    test["id55_error"] = test["pred_id55"] - test["as1_pec50"]
    return pool, test


def summarize_thresholds(name: str, pool: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    as1 = test[test["split"].eq("AS1")].copy()
    as2 = test[test["split"].eq("AS2")].copy()
    for score in SCORE_COLS:
        thresholds = {
            "as1_q95": float(as1[score].quantile(0.95)),
            "test_q95": float(test[score].quantile(0.95)),
            "pool_q95": float(pool[score].quantile(0.95)),
            "train_q95": float(pool[pool["source"].eq("train")][score].quantile(0.95)),
        }
        y_high = as1["as1_pec50"].ge(6).astype(int)
        score_values = as1[score].to_numpy(dtype=float)
        auc = roc_auc_score(y_high, score_values) if y_high.nunique() == 2 else np.nan
        ap = average_precision_score(y_high, score_values) if y_high.nunique() == 2 else np.nan
        for label, threshold in thresholds.items():
            f1 = as1[score].ge(threshold)
            f2 = as2[score].ge(threshold)
            if f1.any():
                base_mae = float(np.mean(np.abs(as1["pred_id55"] - as1["as1_pec50"])))
                best_shift = None
                best_mae = 999.0
                for shift in [0.05, 0.10, 0.15, 0.20, 0.30]:
                    pred = as1["pred_id55"] + f1.astype(float) * shift
                    mae = float(np.mean(np.abs(pred - as1["as1_pec50"])))
                    if mae < best_mae:
                        best_mae = mae
                        best_shift = shift
            else:
                base_mae = float(np.mean(np.abs(as1["pred_id55"] - as1["as1_pec50"])))
                best_mae = base_mae
                best_shift = 0.0
            rows.append(
                {
                    "config": name,
                    "score": score,
                    "threshold_source": label,
                    "threshold": threshold,
                    "as1_flags": int(f1.sum()),
                    "as2_flags": int(f2.sum()),
                    "as1_true_high_flags": int((f1 & as1["as1_pec50"].ge(6)).sum()),
                    "as1_true_low_flags": int((f1 & as1["as1_pec50"].lt(3)).sum()),
                    "as1_flag_mean_pec50": float(as1.loc[f1, "as1_pec50"].mean()) if f1.any() else np.nan,
                    "as1_flag_mean_id55_error": float(as1.loc[f1, "id55_error"].mean()) if f1.any() else np.nan,
                    "as1_gte6_auc": auc,
                    "as1_gte6_ap": ap,
                    "id55_as1_mae": base_mae,
                    "best_high_lift_shift": best_shift,
                    "best_shift_as1_mae": best_mae,
                    "as2_flag_mean_pred_id55": float(as2.loc[f2, "pred_id55"].mean()) if f2.any() else np.nan,
                    "as2_flag_mean_final_p_gte6": float(as2.loc[f2, "final_p_gte6"].mean()) if f2.any() else np.nan,
                    "as2_flag_mean_member_std": float(as2.loc[f2, "member_std"].mean()) if f2.any() else np.nan,
                }
            )
    return pd.DataFrame(rows)


def decile_calibration(name: str, test: pd.DataFrame) -> pd.DataFrame:
    as1 = test[test["split"].eq("AS1")].copy()
    rows = []
    for score in SCORE_COLS:
        as1[f"{score}_decile"] = pd.qcut(as1[score].rank(method="first"), 10, labels=False)
        for decile, sub in as1.groupby(f"{score}_decile", sort=True):
            rows.append(
                {
                    "config": name,
                    "score": score,
                    "decile": int(decile),
                    "n": len(sub),
                    "score_min": float(sub[score].min()),
                    "score_max": float(sub[score].max()),
                    "mean_pec50": float(sub["as1_pec50"].mean()),
                    "gte6_rate": float(sub["as1_pec50"].ge(6).mean()),
                    "lt3_rate": float(sub["as1_pec50"].lt(3).mean()),
                    "mean_id55_error": float(sub["id55_error"].mean()),
                    "mae_id55": float(np.mean(np.abs(sub["id55_error"]))),
                }
            )
    return pd.DataFrame(rows)


def flag_tables(name: str, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    as1 = test[test["split"].eq("AS1")]
    for score in ["pairrank_chembl", "pairrank_htchem"]:
        threshold = float(as1[score].quantile(0.95))
        sub = test[test[score].ge(threshold)].copy()
        sub["config"] = name
        sub["flag_score"] = score
        sub["flag_threshold"] = threshold
        rows.append(sub)
    flagged = pd.concat(rows, ignore_index=True)
    keep = [
        "config",
        "flag_score",
        "test_id",
        "compound_id",
        "molecule_name",
        "split",
        "as1_pec50",
        "pred_id55",
        "id55_error",
        "pairrank_chembl",
        "pairrank_htchem",
        "pairrank_single_conc",
        "final_p_gte6",
        "binary_final_gte6",
        "pred_htchem",
        "lf_mean",
        "member_std",
        "nn_train_tanimoto",
        "nn_train_pec50",
        "nn_potent_tanimoto",
        "high_tail_risk_score",
        "tag_count",
    ]
    flagged = flagged[keep].sort_values(["config", "flag_score", "split", "as1_pec50"])

    overlap_rows = []
    for split, sub in test.groupby("split"):
        flags = {
            "chembl_q95": sub["pairrank_chembl"].ge(as1["pairrank_chembl"].quantile(0.95)),
            "htchem_q95": sub["pairrank_htchem"].ge(as1["pairrank_htchem"].quantile(0.95)),
            "multi_gte6_q95": sub["final_p_gte6"].ge(as1["final_p_gte6"].quantile(0.95)),
            "binary_gte6_q95": sub["binary_final_gte6"].ge(as1["binary_final_gte6"].quantile(0.95)),
        }
        keys = list(flags)
        for i, a in enumerate(keys):
            for b in keys[i:]:
                both = flags[a] & flags[b]
                overlap_rows.append(
                    {
                        "config": name,
                        "split": split,
                        "flag_a": a,
                        "flag_b": b,
                        "n_a": int(flags[a].sum()),
                        "n_b": int(flags[b].sum()),
                        "n_both": int(both.sum()),
                    }
                )
    return flagged, pd.DataFrame(overlap_rows)


def write_report(
    threshold_summary: pd.DataFrame,
    deciles: pd.DataFrame,
    flagged: pd.DataFrame,
    overlaps: pd.DataFrame,
) -> None:
    best = threshold_summary.sort_values("best_shift_as1_mae").head(16)
    top_decile = deciles[deciles["decile"].eq(9)].sort_values(
        ["config", "gte6_rate", "mean_pec50"], ascending=[True, False, False]
    )
    overlap_view = overlaps[
        overlaps["flag_a"].isin(["chembl_q95", "htchem_q95"])
        & overlaps["flag_b"].isin(["multi_gte6_q95", "binary_gte6_q95", "chembl_q95", "htchem_q95"])
    ]
    lines = [
        "# Pairwise Assay-Rank Risk Audit",
        "",
        "This audit joins pairwise assay-rank scores with the id55 risk map and",
        "existing high-classifier scores. It is intended to catch sparse-gate overfit.",
        "",
        "## Best Threshold Rows",
        "",
        best.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Top Decile Calibration",
        "",
        top_decile.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Q95 Overlap With Existing Classifiers",
        "",
        overlap_view.to_markdown(index=False),
        "",
        "## AS1 Q95 Flagged Compounds",
        "",
        flagged[flagged["split"].eq("AS1")]
        .sort_values(["config", "flag_score", "as1_pec50"])
        .to_markdown(index=False, floatfmt=".4f"),
        "",
        "## AS2 Q95 Flagged Compounds",
        "",
        flagged[flagged["split"].eq("AS2")]
        .sort_values(["config", "flag_score", "pred_id55"], ascending=[True, True, False])
        .to_markdown(index=False, floatfmt=".4f"),
    ]
    (AUDIT_DIR / "pairwise_rank_risk_audit.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    threshold_frames = []
    decile_frames = []
    flagged_frames = []
    overlap_frames = []
    for name, config_dir in CONFIGS.items():
        pool, test = load_joined(config_dir)
        threshold_frames.append(summarize_thresholds(name, pool, test))
        decile_frames.append(decile_calibration(name, test))
        flagged, overlaps = flag_tables(name, test)
        flagged_frames.append(flagged)
        overlap_frames.append(overlaps)

    threshold_summary = pd.concat(threshold_frames, ignore_index=True)
    deciles = pd.concat(decile_frames, ignore_index=True)
    flagged = pd.concat(flagged_frames, ignore_index=True)
    overlaps = pd.concat(overlap_frames, ignore_index=True)

    threshold_summary.to_csv(AUDIT_DIR / "threshold_summary.csv", index=False)
    deciles.to_csv(AUDIT_DIR / "decile_calibration.csv", index=False)
    flagged.to_csv(AUDIT_DIR / "q95_flagged_compounds.csv", index=False)
    overlaps.to_csv(AUDIT_DIR / "q95_overlap.csv", index=False)
    write_report(threshold_summary, deciles, flagged, overlaps)

    print("Best threshold rows")
    print(threshold_summary.sort_values("best_shift_as1_mae").head(16).to_string(index=False))
    print("\nTop decile calibration")
    print(
        deciles[deciles["decile"].eq(9)]
        .sort_values(["config", "gte6_rate", "mean_pec50"], ascending=[True, False, False])
        .to_string(index=False)
    )
    print(f"\nWrote {AUDIT_DIR}")


if __name__ == "__main__":
    main()

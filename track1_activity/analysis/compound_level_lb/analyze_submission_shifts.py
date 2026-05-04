#!/usr/bin/env python
"""Compound-level comparison of Track 1 leaderboard submissions.

This script does not train models. It joins existing submission CSVs with
test-compound metadata and reports how LB-known variants move predictions
relative to the current stable baseline / hybrid submissions.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))

from data import get_engine, load_test_smiles, load_train_smiles_target  # noqa: E402
from splits import _morgan_fp_matrix  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "outputs"
SUB_DIR = REPO_ROOT.joinpath("track1_activity", "submissions")


@dataclass(frozen=True)
class SubmissionCase:
    label: str
    path: Path
    lb_mae: float
    lb_sp: float
    notes: str


CASES = [
    SubmissionCase(
        "id32_baseline9",
        SUB_DIR / "ens_caruana_bag20_calibrated_importance_baseline_9pool.csv",
        0.4078471063458229,
        0.8454463307068544,
        "10-seed baseline before family-meta blend",
    ),
    SubmissionCase(
        "id39_swap_t10",
        SUB_DIR
        / "archive"
        / "swap_default_t10_ens_caruana_bag20_calibrated_importance_2026-04-27T12-35.csv",
        0.4080327859318756,
        0.8465176451241602,
        "single optuna trial10 SWAP, near-tie",
    ),
    SubmissionCase(
        "id42_family_meta",
        SUB_DIR / "ens_caruana_bag20_calibrated_importance_meta_id42.csv",
        0.4090744542353690,
        0.8475717867383492,
        "chemprop-family collapsed meta member",
    ),
    SubmissionCase(
        "id43_hybrid",
        SUB_DIR / "ens_hybrid_meta_baseline_5050.csv",
        0.4074838675224626,
        0.8470495157914333,
        "50/50 baseline9 + family_meta; current best",
    ),
    SubmissionCase(
        "id44_anchor",
        SUB_DIR / "ens_caruana_bag20_anchor_residual.csv",
        0.4090205188659963,
        0.8447606118985267,
        "anchor residual correction",
    ),
    SubmissionCase(
        "id45_admet_no_lf",
        SUB_DIR / "ens_caruana_bag20_admet_ai_no_log2fc_calibrated_importance.csv",
        0.4093625824408419,
        0.8428619387501411,
        "ADMET-AI no-log2fc ADD",
    ),
    SubmissionCase(
        "id46_region_a8",
        SUB_DIR / "ens_region_v2_blend_a8.csv",
        0.4091592405210825,
        0.8435311179844552,
        "region-conditioned routing hedged 80/20",
    ),
]


def load_submission(case: SubmissionCase) -> pd.DataFrame:
    if not case.path.exists():
        raise FileNotFoundError(case.path)
    df = pd.read_csv(case.path)
    return df.rename(columns={"Molecule Name": "molecule_name", "pEC50": case.label})[
        ["molecule_name", case.label]
    ]


def tanimoto_matrix(query: np.ndarray, ref: np.ndarray) -> np.ndarray:
    query_bool = query.astype(bool)
    ref_bool = ref.astype(bool)
    inter = query_bool.astype(np.uint16) @ ref_bool.astype(np.uint16).T
    q_sum = query_bool.sum(axis=1, keepdims=True)
    r_sum = ref_bool.sum(axis=1, keepdims=True).T
    union = q_sum + r_sum - inter
    return np.divide(
        inter, union, out=np.zeros_like(inter, dtype=np.float32), where=union > 0
    )


def load_test_metadata() -> pd.DataFrame:
    engine = get_engine()
    return pd.read_sql(
        """
        SELECT
            t.id AS test_idx,
            c.id AS compound_id,
            c.molecule_name,
            c.std_smiles AS smiles,
            d.murcko_scaffold,
            d.logp,
            d.tpsa,
            d.exactmw,
            d.num_heavy_atoms,
            d.num_rings,
            d.num_aromatic_rings,
            d.num_heteroatoms,
            d.num_rotatable_bonds
        FROM test_activity t
        JOIN compounds c ON c.id = t.compound_id
        LEFT JOIN compound_descriptors d ON d.compound_id = c.id
        ORDER BY t.id
        """,
        engine,
    )


def add_similarity_metadata(test_meta: pd.DataFrame) -> pd.DataFrame:
    train = load_train_smiles_target()
    test = load_test_smiles()
    train_fp = _morgan_fp_matrix(train["smiles"].tolist())
    test_fp = _morgan_fp_matrix(test["smiles"].tolist())

    sim_all = tanimoto_matrix(test_fp, train_fp)
    nn_all_idx = sim_all.argmax(axis=1)
    out = test_meta.copy()
    out["nn_train_tanimoto"] = sim_all[np.arange(len(out)), nn_all_idx]
    out["nn_train_pec50"] = train["pec50"].to_numpy()[nn_all_idx]
    out["nn_train_name"] = train["molecule_name"].to_numpy()[nn_all_idx]

    potent_mask = train["pec50"].to_numpy() >= 6.0
    potent_fp = train_fp[potent_mask]
    potent_y = train.loc[potent_mask, "pec50"].to_numpy()
    sim_potent = tanimoto_matrix(test_fp, potent_fp)
    nn_pot_idx = sim_potent.argmax(axis=1)
    out["nn_potent_tanimoto"] = sim_potent[np.arange(len(out)), nn_pot_idx]
    out["nn_potent_pec50"] = potent_y[nn_pot_idx]

    scaffold_counts = (
        out["murcko_scaffold"].fillna("").value_counts().rename("test_scaffold_count")
    )
    out = out.join(scaffold_counts, on="murcko_scaffold")
    return out


def load_prediction_table() -> pd.DataFrame:
    table: pd.DataFrame | None = None
    for case in CASES:
        sub = load_submission(case)
        table = (
            sub if table is None else table.merge(sub, on="molecule_name", how="inner")
        )
    assert table is not None
    if len(table) != 513:
        raise RuntimeError(f"Expected 513 aligned test predictions, got {len(table)}")
    return table


def write_shift_tables(df: pd.DataFrame) -> None:
    pred_cols = [case.label for case in CASES]
    corr = df[pred_cols].corr(method="pearson")
    corr.to_csv(OUT_DIR / "prediction_correlation.csv")

    lb_rows = []
    for case in CASES:
        delta = df[case.label] - df["id32_baseline9"]
        delta_hybrid = df[case.label] - df["id43_hybrid"]
        lb_rows.append(
            {
                "case": case.label,
                "lb_mae": case.lb_mae,
                "lb_delta_vs_id32": case.lb_mae - CASES[0].lb_mae,
                "lb_sp": case.lb_sp,
                "mean_pred": df[case.label].mean(),
                "std_pred": df[case.label].std(ddof=0),
                "mean_delta_vs_id32": delta.mean(),
                "mean_abs_delta_vs_id32": delta.abs().mean(),
                "p90_abs_delta_vs_id32": delta.abs().quantile(0.90),
                "max_abs_delta_vs_id32": delta.abs().max(),
                "mean_abs_delta_vs_hybrid": delta_hybrid.abs().mean(),
                "pearson_vs_id32": df[["id32_baseline9", case.label]].corr().iloc[0, 1],
                "spearman_vs_id32": stats.spearmanr(
                    df["id32_baseline9"], df[case.label]
                ).statistic,
                "notes": case.notes,
            }
        )
    pd.DataFrame(lb_rows).to_csv(OUT_DIR / "case_shift_summary.csv", index=False)

    detail = df.copy()
    for case in CASES[1:]:
        detail[f"{case.label}_minus_id32"] = (
            detail[case.label] - detail["id32_baseline9"]
        )
        detail[f"{case.label}_minus_hybrid"] = (
            detail[case.label] - detail["id43_hybrid"]
        )
    detail.to_csv(OUT_DIR / "compound_prediction_shifts.csv", index=False)


def summarize_bins(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    working["baseline_pred_bin"] = pd.qcut(
        working["id32_baseline9"], q=5, labels=False, duplicates="drop"
    )
    working["nn_train_bin"] = pd.qcut(
        working["nn_train_tanimoto"], q=5, labels=False, duplicates="drop"
    )
    working["logp_bin"] = pd.qcut(working["logp"], q=5, labels=False, duplicates="drop")
    working["scaffold_size_bin"] = pd.cut(
        working["test_scaffold_count"],
        bins=[0, 1, 2, 5, 999],
        labels=["singleton", "pair", "3-5", "6+"],
        include_lowest=True,
    )

    rows = []
    for group_col in [
        "baseline_pred_bin",
        "nn_train_bin",
        "logp_bin",
        "scaffold_size_bin",
    ]:
        for group_value, g in working.groupby(group_col, observed=True):
            row = {
                "group": group_col,
                "value": str(group_value),
                "n": len(g),
                "mean_id32": g["id32_baseline9"].mean(),
                "mean_hybrid": g["id43_hybrid"].mean(),
                "hybrid_minus_id32": (g["id43_hybrid"] - g["id32_baseline9"]).mean(),
                "meta_minus_id32": (g["id42_family_meta"] - g["id32_baseline9"]).mean(),
                "region_minus_id32": (g["id46_region_a8"] - g["id32_baseline9"]).mean(),
                "anchor_minus_id32": (g["id44_anchor"] - g["id32_baseline9"]).mean(),
                "admet_minus_id32": (
                    g["id45_admet_no_lf"] - g["id32_baseline9"]
                ).mean(),
            }
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "shift_by_bins.csv", index=False)
    return out


def write_markdown_report(df: pd.DataFrame, bins: pd.DataFrame) -> None:
    case_summary = pd.read_csv(OUT_DIR / "case_shift_summary.csv")
    corr = pd.read_csv(OUT_DIR / "prediction_correlation.csv", index_col=0)

    top_hybrid = (
        df.assign(abs_shift=(df["id43_hybrid"] - df["id32_baseline9"]).abs())
        .sort_values("abs_shift", ascending=False)
        .head(20)
    )
    harmful_like = case_summary.sort_values("lb_delta_vs_id32", ascending=False).head(3)

    report = [
        "# Compound-Level LB Shift Analysis",
        "",
        "## Scope",
        "",
        "This is a test-side analysis of LB-known submission CSVs. It cannot identify",
        "true per-compound errors because blinded labels are unavailable. It instead",
        "summarizes which chemical/prediction regions each LB-known variant moves.",
        "",
        "## Case Summary",
        "",
        case_summary[
            [
                "case",
                "lb_mae",
                "lb_delta_vs_id32",
                "lb_sp",
                "mean_abs_delta_vs_id32",
                "p90_abs_delta_vs_id32",
                "pearson_vs_id32",
            ]
        ].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Correlation To Baseline",
        "",
        corr[["id32_baseline9"]]
        .sort_values("id32_baseline9")
        .to_markdown(floatfmt=".5f"),
        "",
        "## Largest Hybrid Moves",
        "",
        top_hybrid[
            [
                "molecule_name",
                "id32_baseline9",
                "id43_hybrid",
                "abs_shift",
                "nn_train_tanimoto",
                "nn_potent_tanimoto",
                "logp",
                "exactmw",
                "test_scaffold_count",
            ]
        ].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Bin Summary",
        "",
        bins.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Most Regressive LB Cases Included",
        "",
        harmful_like[
            ["case", "lb_delta_vs_id32", "mean_abs_delta_vs_id32", "notes"]
        ].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Files",
        "",
        "- `compound_prediction_shifts.csv`: per-compound metadata + prediction shifts",
        "- `case_shift_summary.csv`: one row per LB-known submission",
        "- `shift_by_bins.csv`: mean shifts by prediction/similarity/property bins",
        "- `prediction_correlation.csv`: pairwise prediction correlations",
    ]
    (OUT_DIR / "report.md").write_text("\n".join(report) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metadata = add_similarity_metadata(load_test_metadata())
    preds = load_prediction_table()
    df = metadata.merge(preds, on="molecule_name", how="inner")
    if len(df) != 513:
        raise RuntimeError(f"metadata/prediction alignment produced {len(df)} rows")

    write_shift_tables(df)
    bins = summarize_bins(df)
    write_markdown_report(df, bins)
    print(f"Wrote analysis outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()

#!/usr/bin/env -S pixi run python
"""Build compound-level figures for the Track 1 Phase 2 case-study report."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))

from data import get_engine, load_test_smiles, load_train_smiles_target  # noqa: E402
from splits import _morgan_fp_matrix  # noqa: E402

ASSET_DIR = REPO_ROOT / "docs" / "track1_explain" / "assets" / "phase2_compound_cases"
SUBMISSION_DIR = REPO_ROOT / "track1_activity" / "submissions"
LF_PATH = (
    REPO_ROOT
    / "data"
    / "chemprop_pretrain_log2fc_predictions_optuna_trial10_seed5ens.parquet"
)


def savefig(fig: plt.Figure, name: str) -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(ASSET_DIR / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def load_submission(path: Path, col: str = "pred_id55") -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.rename(columns={"Molecule Name": "molecule_name", "pEC50": col})[
        ["molecule_name", col]
    ]


def load_context() -> pd.DataFrame:
    engine = get_engine()
    as1 = pd.read_sql(
        """
        SELECT
            t.id AS test_id,
            c.id AS compound_id,
            c.molecule_name,
            c.std_smiles AS smiles,
            l.pec50 AS true_pec50,
            l.emax_estimate,
            l.emax_vs_pos_ctrl
        FROM test_activity_phase1_labels l
        JOIN test_activity t ON t.compound_id = l.compound_id
        JOIN compounds c ON c.id = l.compound_id
        ORDER BY t.id
        """,
        engine,
    )
    df = as1.merge(
        load_submission(SUBMISSION_DIR / "ens_id51_top500_potent46_t40_soft_g35.csv"),
        on="molecule_name",
        how="left",
    )
    df["error"] = df["pred_id55"] - df["true_pec50"]
    df["abs_error"] = df["error"].abs()

    lf = pd.read_parquet(LF_PATH).loc[df["compound_id"].astype(int).tolist()].copy()
    lf["lf_mean"] = 0.5 * (lf["log2fc_8p25_pred"] + lf["log2fc_33_pred"])
    df = df.merge(lf.reset_index(), on="compound_id", how="left")

    train = load_train_smiles_target()
    train_fp = _morgan_fp_matrix(train["smiles"].tolist()).astype(bool)
    test_fp = _morgan_fp_matrix(load_test_smiles()["smiles"].tolist()).astype(bool)
    query_fp = test_fp[df["test_id"].to_numpy(dtype=int) - 1]

    inter = query_fp.astype(np.uint16) @ train_fp.astype(np.uint16).T
    union = (
        query_fp.sum(axis=1, keepdims=True)
        + train_fp.sum(axis=1, keepdims=True).T
        - inter
    )
    sim = np.divide(
        inter, union, out=np.zeros_like(inter, dtype=np.float32), where=union > 0
    )
    nn_idx = sim.argmax(axis=1)
    df["nn_train_tanimoto"] = sim[np.arange(len(df)), nn_idx]
    df["nn_train_pec50"] = train["pec50"].to_numpy()[nn_idx]
    df["nn_train_name"] = train["molecule_name"].to_numpy()[nn_idx]
    df["nn_train_smiles"] = train["smiles"].to_numpy()[nn_idx]

    potent_mask = train["pec50"].to_numpy() >= 6.0
    potent_fp = train_fp[potent_mask]
    potent_y = train.loc[potent_mask, "pec50"].to_numpy()
    inter_p = query_fp.astype(np.uint16) @ potent_fp.astype(np.uint16).T
    union_p = (
        query_fp.sum(axis=1, keepdims=True)
        + potent_fp.sum(axis=1, keepdims=True).T
        - inter_p
    )
    sim_p = np.divide(
        inter_p,
        union_p,
        out=np.zeros_like(inter_p, dtype=np.float32),
        where=union_p > 0,
    )
    nn_p = sim_p.argmax(axis=1)
    df["nn_potent_tanimoto"] = sim_p[np.arange(len(df)), nn_p]
    df["nn_potent_pec50"] = potent_y[nn_p]
    return df


def mol_from_smiles(smiles: str) -> Chem.Mol | None:
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is not None:
        Chem.rdDepictor.Compute2DCoords(mol)
    return mol


def legend(row: pd.Series, prefix: str = "") -> str:
    return (
        f"{prefix}{row.molecule_name}\n"
        f"true {row.true_pec50:.2f}  pred {row.pred_id55:.2f}  err {row.error:+.2f}\n"
        f"LFmean {row.lf_mean:.2f}  NN {row.nn_train_tanimoto:.2f}/{row.nn_train_pec50:.2f}"
    )


def draw_compound_grid(df: pd.DataFrame, name: str, mols_per_row: int = 4) -> None:
    rows = df.copy()
    mols = [mol_from_smiles(smi) for smi in rows["smiles"]]
    legends = [legend(row) for row in rows.itertuples(index=False)]
    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=mols_per_row,
        subImgSize=(360, 270),
        legends=legends,
        useSVG=False,
    )
    out = ASSET_DIR / name
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    img.save(out)


def draw_nn_pairs(df: pd.DataFrame, name: str) -> None:
    mols: list[Chem.Mol | None] = []
    legends: list[str] = []
    for row in df.itertuples(index=False):
        mols.append(mol_from_smiles(row.smiles))
        legends.append(
            f"AS1 {row.molecule_name}\n"
            f"true {row.true_pec50:.2f} pred {row.pred_id55:.2f} err {row.error:+.2f}"
        )
        mols.append(mol_from_smiles(row.nn_train_smiles))
        legends.append(
            f"train NN {row.nn_train_name}\n"
            f"pEC50 {row.nn_train_pec50:.2f} sim {row.nn_train_tanimoto:.2f}"
        )
    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=2,
        subImgSize=(380, 270),
        legends=legends,
        useSVG=False,
    )
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    img.save(ASSET_DIR / name)


def plot_bin_direction_summary(df: pd.DataFrame) -> pd.DataFrame:
    labels = ["<3", "3-4", "4-5", "5-6", ">=6"]
    frame = df.copy()
    frame["true_bin"] = pd.cut(
        frame["true_pec50"], [-np.inf, 3, 4, 5, 6, np.inf], labels=labels
    )
    frame["direction"] = np.where(frame["error"] >= 0, "overpred", "underpred")
    summary = (
        frame.groupby(["true_bin", "direction"], observed=True)
        .agg(n=("error", "size"), mae=("abs_error", "mean"), bias=("error", "mean"))
        .reset_index()
    )
    summary.to_csv(ASSET_DIR / "bin_error_direction_summary.csv", index=False)

    counts = summary.pivot(index="true_bin", columns="direction", values="n").fillna(0)
    for col in ["overpred", "underpred"]:
        if col not in counts:
            counts[col] = 0
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    x = np.arange(len(counts))
    ax.bar(x, counts["overpred"], color="#e45756", label="overpred")
    ax.bar(x, -counts["underpred"], color="#4c78a8", label="underpred")
    ax.axhline(0, color="#333333", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(counts.index)
    ax.set_ylabel("compound count (underpred shown negative)")
    ax.set_title("id55 error direction by true pEC50 bin")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=True)
    savefig(fig, "bin_error_direction_counts.png")
    return summary


def case_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    low_over = (
        df[(df["true_pec50"] < 3) & (df["error"] > 0)]
        .sort_values("abs_error", ascending=False)
        .head(8)
    )
    mid_over = (
        df[(df["true_pec50"].between(3, 4, inclusive="left")) & (df["error"] > 0)]
        .sort_values("abs_error", ascending=False)
        .head(4)
    )
    mid_under = (
        df[(df["true_pec50"].between(3, 4, inclusive="left")) & (df["error"] < 0)]
        .sort_values("abs_error", ascending=False)
        .head(4)
    )
    high_under = (
        df[(df["true_pec50"] >= 6) & (df["error"] < 0)]
        .sort_values("abs_error", ascending=False)
        .head(8)
    )
    bins = [
        ("<3", df["true_pec50"] < 3),
        ("3-4", df["true_pec50"].between(3, 4, inclusive="left")),
        ("4-5", df["true_pec50"].between(4, 5, inclusive="left")),
        ("5-6", df["true_pec50"].between(5, 6, inclusive="left")),
        (">=6", df["true_pec50"] >= 6),
    ]
    well_parts = []
    for label, mask in bins:
        part = df[mask].sort_values("abs_error", ascending=True).head(2).copy()
        part["true_bin"] = label
        well_parts.append(part)
    return {
        "low_tail_overpred_cases": low_over,
        "mid_3_4_bidirectional_cases": pd.concat([mid_over, mid_under]).sort_values(
            "abs_error", ascending=False
        ),
        "high_tail_underpred_cases": high_under,
        "well_predicted_cases": pd.concat(well_parts).sort_values(
            ["true_bin", "abs_error"]
        ),
    }


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    df = load_context()
    plot_bin_direction_summary(df)

    tables = case_tables(df)
    keep_cols = [
        "molecule_name",
        "compound_id",
        "true_pec50",
        "pred_id55",
        "error",
        "abs_error",
        "emax_estimate",
        "log2fc_8p25_pred",
        "log2fc_33_pred",
        "lf_mean",
        "nn_train_tanimoto",
        "nn_train_pec50",
        "nn_train_name",
        "nn_potent_tanimoto",
        "nn_potent_pec50",
        "true_bin",
        "smiles",
    ]
    for name, table in tables.items():
        for col in keep_cols:
            if col not in table.columns:
                table[col] = pd.NA
        table[keep_cols].to_csv(ASSET_DIR / f"{name}.csv", index=False)

    draw_compound_grid(
        tables["low_tail_overpred_cases"],
        "low_tail_overpred_structures.png",
    )
    draw_compound_grid(
        tables["mid_3_4_bidirectional_cases"],
        "mid_3_4_bidirectional_structures.png",
    )
    draw_compound_grid(
        tables["high_tail_underpred_cases"],
        "high_tail_underpred_structures.png",
    )
    draw_compound_grid(
        tables["well_predicted_cases"],
        "well_predicted_structures.png",
        mols_per_row=5,
    )
    draw_nn_pairs(tables["low_tail_overpred_cases"].head(4), "low_tail_nn_pairs.png")

    print(f"Wrote compound case assets to {ASSET_DIR}")


if __name__ == "__main__":
    main()

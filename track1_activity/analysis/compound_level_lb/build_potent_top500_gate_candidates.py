#!/usr/bin/env python
"""Build CSV-only candidates that borrow top500 signal near potent analogs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
SUB_DIR = REPO_ROOT.joinpath("track1_activity", "submissions")
OUT_DIR = Path(__file__).resolve().parent / "outputs" / "potent_top500_gate"

ANCHOR_PATH = SUB_DIR / "ens_meta_axis_reverse_id50_g10.csv"
ENSEMBLE_PATH = SUB_DIR / "ens_caruana_bag20.csv"
TOP500_PATH = (
    SUB_DIR / "tabpfn_cheme_2d_full_boltz_log2fc_pred_seed10ens_top500_umap.csv"
)


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


def load_potent46_test_gate() -> pd.DataFrame:
    import sys

    sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
    from data import get_engine, load_test_smiles, load_train_smiles_target
    from splits import _morgan_fp_matrix

    engine = get_engine()
    aux = pd.read_sql(
        """
        SELECT
            t.compound_id,
            ca.pec50 AS counter_pec50
        FROM train_activity t
        LEFT JOIN counter_assay ca ON ca.compound_id = t.compound_id
        ORDER BY t.id
        """,
        engine,
    )
    train = load_train_smiles_target().copy()
    test = load_test_smiles().copy()
    train["counter_pec50"] = aux["counter_pec50"].to_numpy()
    selectivity = train["pec50"] - train["counter_pec50"]
    potent46 = (train["pec50"] >= 6.0) & (selectivity >= 1.5)

    train_fp = _morgan_fp_matrix(train["smiles"].tolist())
    test_fp = _morgan_fp_matrix(test["smiles"].tolist())
    potent_fp = train_fp[potent46.to_numpy()]
    sim = tanimoto_matrix(test_fp, potent_fp)
    nn = sim.max(axis=1)
    return pd.DataFrame(
        {
            "molecule_name": test["molecule_name"].to_numpy(),
            "nn_potent46_tanimoto": nn,
        }
    )


def align_submission(path: Path, reference: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(path)
    aligned = reference[["Molecule Name", "SMILES"]].merge(
        df[["Molecule Name", "SMILES", "pEC50"]],
        on="Molecule Name",
        how="left",
        suffixes=("_reference", ""),
        validate="one_to_one",
    )
    if aligned["pEC50"].isna().any():
        raise RuntimeError(f"missing aligned predictions from {path}")
    if not (
        aligned["SMILES_reference"].to_numpy() == aligned["SMILES"].to_numpy()
    ).all():
        raise RuntimeError(f"SMILES mismatch after molecule-name alignment: {path}")
    return pd.DataFrame(
        {
            "SMILES": aligned["SMILES_reference"],
            "Molecule Name": aligned["Molecule Name"],
            "pEC50": aligned["pEC50"],
        }
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    anchor = pd.read_csv(ANCHOR_PATH)
    ensemble = align_submission(ENSEMBLE_PATH, anchor)
    top500 = align_submission(TOP500_PATH, anchor)
    gate_meta = load_potent46_test_gate()
    if not (
        anchor["Molecule Name"].to_numpy() == gate_meta["molecule_name"].to_numpy()
    ).all():
        raise RuntimeError("gate metadata order mismatch")

    anchor_pred = anchor["pEC50"].to_numpy(dtype=np.float64)
    delta = top500["pEC50"].to_numpy(dtype=np.float64) - ensemble["pEC50"].to_numpy(
        dtype=np.float64
    )
    nn = gate_meta["nn_potent46_tanimoto"].to_numpy(dtype=np.float64)

    rows = []
    for threshold in [0.30, 0.35, 0.40]:
        hard_gate = (nn >= threshold).astype(np.float64)
        # Soft gate avoids a sharp boundary at threshold while staying local.
        soft_gate = np.clip((nn - threshold) / 0.15, 0.0, 1.0)
        for gate_name, gate in [("hard", hard_gate), ("soft", soft_gate)]:
            for gamma in [0.15, 0.25, 0.35, 0.50]:
                shift = gamma * gate * delta
                out = anchor.copy()
                out["pEC50"] = anchor_pred + shift
                suffix = (
                    f"t{int(threshold * 100):02d}_{gate_name}_g{int(gamma * 100):02d}"
                )
                path = SUB_DIR / f"ens_id51_top500_potent46_{suffix}.csv"
                out.to_csv(path, index=False)
                rows.append(
                    {
                        "candidate": path.name,
                        "threshold": threshold,
                        "gate": gate_name,
                        "gamma": gamma,
                        "n_gate_nonzero": int(np.count_nonzero(gate > 0)),
                        "n_gate_full": int(np.count_nonzero(gate >= 1.0)),
                        "mean_gate": float(gate.mean()),
                        "mean_shift": float(shift.mean()),
                        "mean_abs_shift": float(np.abs(shift).mean()),
                        "p90_abs_shift": float(np.quantile(np.abs(shift), 0.90)),
                        "max_abs_shift": float(np.abs(shift).max()),
                        "corr_vs_anchor": float(
                            np.corrcoef(
                                out["pEC50"].to_numpy(dtype=np.float64), anchor_pred
                            )[0, 1]
                        ),
                        "path": str(path.relative_to(REPO_ROOT)),
                    }
                )

    summary = pd.DataFrame(rows).sort_values(["threshold", "gate", "gamma"])
    summary.to_csv(OUT_DIR / "candidate_summary.csv", index=False)
    gate_meta.to_csv(OUT_DIR / "test_potent46_gate.csv", index=False)

    report = [
        "# Potent46-Gated Top500 CSV Candidates",
        "",
        "Candidate formula:",
        "",
        "```text",
        "candidate = id51 + gamma * gate(nn_potent46_tanimoto) * (top500 - ens_caruana_bag20)",
        "```",
        "",
        "This is a CSV-only probe. It does not retrain or change the ensemble pool.",
        "",
        "## Candidate Summary",
        "",
        summary.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Recommended first probe",
        "",
        "Submitted `ens_id51_top500_potent46_t40_soft_g35.csv` as id=55.",
        "It improved id51 by 0.000246 MAE on LB, with a small Spearman drop.",
    ]
    (OUT_DIR / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote potent top500 gate candidates to {OUT_DIR}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

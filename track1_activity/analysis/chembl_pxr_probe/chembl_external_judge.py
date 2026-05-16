#!/usr/bin/env python
"""Use ChEMBL PXR activation data as an external judge, not as training data.

The key safety rule is to exclude any ChEMBL molecule whose InChIKey exactly
matches a challenge train or test compound before building nearest-neighbor
signals. The remaining external data are used only for diagnostics on submitted
or candidate CSVs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "track1_activity" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from chembl_pxr_activation_probe import (  # noqa: E402
    build_nn_features_from_similarity,
    load_chembl_pxr_activities,
    tanimoto_matrix,
)
from data import get_engine  # noqa: E402
from splits import _morgan_fp_matrix  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "outputs" / "external_judge"
SUB_DIR = REPO_ROOT / "track1_activity" / "submissions"

CANDIDATES = {
    "id55_soft_g35": SUB_DIR / "ens_id51_top500_potent46_t40_soft_g35.csv",
    "id57_soft_g50": SUB_DIR / "ens_id51_top500_potent46_t40_soft_g50.csv",
    "id58_combo_rank1": SUB_DIR / "ens_id55_combo_gate_rank1.csv",
    "log2fc_gate_optuna_q60_g50": SUB_DIR
    / "ens_id55_log2fc_gate_optuna_log2fc33_soft_q60_g50.csv",
    "shap_numrings_pos_g50": SUB_DIR / "ens_id55_shap_numrings_pos_highsoftq20_g50.csv",
    "combo_rank2_numrings": SUB_DIR / "ens_id55_combo_gate_rank2.csv",
    "combo_rank3_familygap": SUB_DIR / "ens_id55_combo_gate_rank3.csv",
}


def load_challenge() -> tuple[pd.DataFrame, pd.DataFrame]:
    engine = get_engine()
    train = pd.read_sql(
        """
        SELECT c.molecule_name, c.std_smiles AS smiles, t.pec50, d.inchikey
        FROM train_activity t
        JOIN compounds c ON c.id = t.compound_id
        LEFT JOIN compound_descriptors d ON d.compound_id = c.id
        ORDER BY t.id
        """,
        engine,
    )
    test = pd.read_sql(
        """
        SELECT c.molecule_name, c.std_smiles AS smiles, d.inchikey
        FROM test_activity t
        JOIN compounds c ON c.id = t.compound_id
        LEFT JOIN compound_descriptors d ON d.compound_id = c.id
        ORDER BY t.id
        """,
        engine,
    )
    return train, test


def prepare_external(
    chembl: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    challenge_keys = set(train["inchikey"].dropna()).union(
        set(test["inchikey"].dropna())
    )
    raw = chembl.copy()
    raw["overlaps_challenge_exact"] = raw["inchikey"].isin(challenge_keys)
    external = raw[~raw["overlaps_challenge_exact"]].reset_index(drop=True)
    return raw, external


def build_features(
    compounds: pd.DataFrame, external: pd.DataFrame, *, include_target: bool
) -> pd.DataFrame:
    ext_fp = _morgan_fp_matrix(external["smiles"].tolist())
    values = external["chembl_pxr_pchembl"].to_numpy(dtype=np.float32)
    ext_keys = external["inchikey"].to_numpy()
    sim = tanimoto_matrix(_morgan_fp_matrix(compounds["smiles"].tolist()), ext_fp)
    exact = compounds["inchikey"].to_numpy()[:, None] == ext_keys[None, :]
    features = build_nn_features_from_similarity(sim, values, exact)
    features = features.rename(
        columns={
            "chembl_pxr_nn_tanimoto": "external_nn_tanimoto",
            "chembl_pxr_nn_pchembl": "external_nn_pchembl",
            "chembl_pxr_top5_pchembl": "external_top5_pchembl",
            "chembl_pxr_has_exact_match": "external_has_exact_match",
            "chembl_pxr_covered_t03": "external_covered_t03",
            "chembl_pxr_covered_t04": "external_covered_t04",
        }
    )
    base_cols = ["molecule_name", "smiles", "inchikey"]
    if include_target:
        base_cols.append("pec50")
    return pd.concat(
        [compounds[base_cols].reset_index(drop=True), features.reset_index(drop=True)],
        axis=1,
    )


def safe_corr(x: np.ndarray, y: np.ndarray, *, method: str) -> float:
    if len(x) < 3 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    if method == "pearson":
        return float(np.corrcoef(x, y)[0, 1])
    if method == "spearman":
        return float(stats.spearmanr(x, y).statistic)
    raise ValueError(method)


def train_signal_summary(train_features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    y = train_features["pec50"].to_numpy(dtype=np.float64)
    ext = train_features["external_nn_pchembl"].to_numpy(dtype=np.float64)
    sim = train_features["external_nn_tanimoto"].to_numpy(dtype=np.float64)
    for threshold in (0.0, 0.20, 0.25, 0.30, 0.35, 0.40):
        mask = sim >= threshold
        if mask.sum() == 0:
            continue
        rows.append(
            {
                "threshold": threshold,
                "n": int(mask.sum()),
                "frac": float(mask.mean()),
                "nn_sim_median": float(np.median(sim[mask])),
                "pec50_mean": float(np.mean(y[mask])),
                "external_pchembl_mean": float(np.mean(ext[mask])),
                "pearson_pec50_vs_external": safe_corr(
                    y[mask], ext[mask], method="pearson"
                ),
                "spearman_pec50_vs_external": safe_corr(
                    y[mask], ext[mask], method="spearman"
                ),
                "mae_direct_external": float(np.mean(np.abs(y[mask] - ext[mask]))),
                "centered_mae_external": float(
                    np.mean(
                        np.abs(
                            (y[mask] - y[mask].mean()) - (ext[mask] - ext[mask].mean())
                        )
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def load_candidate(path: Path, test_features: pd.DataFrame) -> np.ndarray | None:
    if not path.exists():
        return None
    sub = pd.read_csv(path)
    if len(sub) != len(test_features):
        raise RuntimeError(f"{path} has {len(sub)} rows, expected {len(test_features)}")
    if not (
        sub["Molecule Name"].to_numpy() == test_features["molecule_name"].to_numpy()
    ).all():
        raise RuntimeError(f"{path} molecule order mismatch")
    return sub["pEC50"].to_numpy(dtype=np.float64)


def candidate_judge(test_features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    preds: dict[str, np.ndarray] = {}
    for name, path in CANDIDATES.items():
        pred = load_candidate(path, test_features)
        if pred is not None:
            preds[name] = pred
    if "id57_soft_g50" in preds:
        anchor_name = "id57_soft_g50"
    elif "id55_soft_g35" in preds:
        anchor_name = "id55_soft_g35"
    else:
        anchor_name = next(iter(preds))
    anchor = preds[anchor_name]

    sim = test_features["external_nn_tanimoto"].to_numpy(dtype=np.float64)
    ext = test_features["external_nn_pchembl"].to_numpy(dtype=np.float64)
    rows = []
    detail_rows = []
    for name, pred in preds.items():
        delta = pred - anchor
        for threshold in (0.0, 0.20, 0.25, 0.30, 0.35, 0.40):
            mask = sim >= threshold
            if mask.sum() == 0:
                continue
            ext_centered = ext[mask] - ext[mask].mean()
            pred_centered = pred[mask] - pred[mask].mean()
            anchor_centered = anchor[mask] - anchor[mask].mean()
            rows.append(
                {
                    "candidate": name,
                    "anchor": anchor_name,
                    "threshold": threshold,
                    "n": int(mask.sum()),
                    "pred_mean": float(pred[mask].mean()),
                    "external_mean": float(ext[mask].mean()),
                    "pred_external_spearman": safe_corr(
                        pred[mask], ext[mask], method="spearman"
                    ),
                    "pred_external_pearson": safe_corr(
                        pred[mask], ext[mask], method="pearson"
                    ),
                    "direct_mae_to_external": float(
                        np.mean(np.abs(pred[mask] - ext[mask]))
                    ),
                    "centered_mae_to_external": float(
                        np.mean(np.abs(pred_centered - ext_centered))
                    ),
                    "direct_mae_delta_vs_anchor": float(
                        np.mean(np.abs(pred[mask] - ext[mask]))
                        - np.mean(np.abs(anchor[mask] - ext[mask]))
                    ),
                    "centered_mae_delta_vs_anchor": float(
                        np.mean(np.abs(pred_centered - ext_centered))
                        - np.mean(np.abs(anchor_centered - ext_centered))
                    ),
                    "mean_shift_vs_anchor": float(delta[mask].mean()),
                    "mean_abs_shift_vs_anchor": float(np.abs(delta[mask]).mean()),
                    "alignment_dot_vs_anchor": float(
                        np.mean(delta[mask] * (ext[mask] - anchor[mask]))
                    ),
                }
            )
        if name != anchor_name:
            top = np.argsort(-np.abs(delta))[:30]
            for idx in top:
                detail_rows.append(
                    {
                        "candidate": name,
                        "molecule_name": test_features.iloc[idx]["molecule_name"],
                        "smiles": test_features.iloc[idx]["smiles"],
                        "external_nn_tanimoto": sim[idx],
                        "external_nn_pchembl": ext[idx],
                        "anchor_pred": anchor[idx],
                        "candidate_pred": pred[idx],
                        "shift_vs_anchor": delta[idx],
                        "move_toward_external": delta[idx] * (ext[idx] - anchor[idx])
                        > 0,
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(detail_rows)


def write_report(
    raw_external: pd.DataFrame,
    external: pd.DataFrame,
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    train_signal: pd.DataFrame,
    judge: pd.DataFrame,
) -> None:
    coverage = pd.DataFrame(
        [
            {
                "split": "train",
                "n": len(train_features),
                "exact_after_exclusion": int(
                    train_features["external_has_exact_match"].sum()
                ),
                "nn_ge_0.25": int(
                    (train_features["external_nn_tanimoto"] >= 0.25).sum()
                ),
                "nn_ge_0.30": int(
                    (train_features["external_nn_tanimoto"] >= 0.30).sum()
                ),
                "nn_ge_0.35": int(
                    (train_features["external_nn_tanimoto"] >= 0.35).sum()
                ),
                "nn_ge_0.40": int(
                    (train_features["external_nn_tanimoto"] >= 0.40).sum()
                ),
                "nn_max": float(train_features["external_nn_tanimoto"].max()),
                "nn_median": float(train_features["external_nn_tanimoto"].median()),
            },
            {
                "split": "test",
                "n": len(test_features),
                "exact_after_exclusion": int(
                    test_features["external_has_exact_match"].sum()
                ),
                "nn_ge_0.25": int(
                    (test_features["external_nn_tanimoto"] >= 0.25).sum()
                ),
                "nn_ge_0.30": int(
                    (test_features["external_nn_tanimoto"] >= 0.30).sum()
                ),
                "nn_ge_0.35": int(
                    (test_features["external_nn_tanimoto"] >= 0.35).sum()
                ),
                "nn_ge_0.40": int(
                    (test_features["external_nn_tanimoto"] >= 0.40).sum()
                ),
                "nn_max": float(test_features["external_nn_tanimoto"].max()),
                "nn_median": float(test_features["external_nn_tanimoto"].median()),
            },
        ]
    )
    coverage.to_csv(OUT_DIR / "external_judge_coverage.csv", index=False)

    judge_t03 = judge[judge["threshold"].eq(0.30)].sort_values(
        ["centered_mae_delta_vs_anchor", "direct_mae_delta_vs_anchor"]
    )
    report = [
        "# ChEMBL External Judge",
        "",
        "Exact challenge train/test InChIKeys were excluded from the external ChEMBL",
        "PXR activation reference before nearest-neighbor scoring.",
        "",
        "## External Set",
        "",
        f"- raw filtered ChEMBL PXR activation molecules: {len(raw_external)}",
        f"- exact challenge overlaps removed: {int(raw_external['overlaps_challenge_exact'].sum())}",
        f"- external molecules after exclusion: {len(external)}",
        "",
        "## Coverage",
        "",
        coverage.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Train Signal",
        "",
        train_signal.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Candidate Judge at Tanimoto >= 0.30",
        "",
        judge_t03.to_markdown(index=False, floatfmt=".5f"),
        "",
        "## Interpretation",
        "",
        "Coverage is intentionally reported before using this as a decision aid.",
        "If test coverage is sparse or train correlation is weak, ChEMBL should be",
        "treated as a qualitative warning light rather than a submission gate.",
    ]
    (OUT_DIR / "external_judge_report.md").write_text("\n".join(report) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train, test = load_challenge()
    chembl = load_chembl_pxr_activities()
    raw_external, external = prepare_external(chembl, train, test)
    if external.empty:
        raise RuntimeError("all external ChEMBL rows were exact challenge overlaps")

    train_features = build_features(train, external, include_target=True)
    test_features = build_features(test, external, include_target=False)
    train_signal = train_signal_summary(train_features)
    judge, detail = candidate_judge(test_features)

    raw_external.to_csv(
        OUT_DIR / "external_pxr_activation_raw_with_overlap_flag.csv", index=False
    )
    external.to_csv(
        OUT_DIR / "external_pxr_activation_no_challenge_exact.csv", index=False
    )
    train_features.to_csv(OUT_DIR / "train_external_judge_features.csv", index=False)
    test_features.to_csv(OUT_DIR / "test_external_judge_features.csv", index=False)
    train_signal.to_csv(OUT_DIR / "train_external_signal_summary.csv", index=False)
    judge.to_csv(OUT_DIR / "candidate_external_judge_summary.csv", index=False)
    detail.to_csv(OUT_DIR / "candidate_external_judge_largest_shifts.csv", index=False)
    write_report(
        raw_external, external, train_features, test_features, train_signal, judge
    )

    print(f"Raw ChEMBL activation molecules: {len(raw_external)}")
    print(
        f"Removed exact challenge overlaps: {int(raw_external['overlaps_challenge_exact'].sum())}"
    )
    print(f"External after exclusion: {len(external)}")
    print("\n=== Train signal ===")
    print(train_signal.to_markdown(index=False, floatfmt=".4f"))
    print("\n=== Candidate judge Tanimoto >= 0.30 ===")
    print(
        judge[judge["threshold"].eq(0.30)]
        .sort_values(["centered_mae_delta_vs_anchor", "direct_mae_delta_vs_anchor"])
        .to_markdown(index=False, floatfmt=".5f")
    )
    print(f"\nWrote {OUT_DIR}")


if __name__ == "__main__":
    main()

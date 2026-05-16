#!/usr/bin/env python
"""Summarize pseudo-public CV split designs for future candidate validation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "track1_activity" / "src"
sys.path.insert(0, str(SRC_DIR))

from data import get_engine, load_test_smiles, load_train_smiles_with_counter  # noqa: E402
from splits import (  # noqa: E402
    adversarial_split_indices,
    mixed_analog_diversity_split_indices,
    test_nn_split_indices,
    umap_split_indices,
)

OUT_DIR = Path(__file__).resolve().parent / "outputs" / "pseudo_public_splits"
LF_PATH = (
    REPO_ROOT
    / "data"
    / "chemprop_pretrain_log2fc_predictions_optuna_trial10_seed5ens.parquet"
)
CHEMBL_FEATURES = (
    REPO_ROOT
    / "track1_activity"
    / "analysis"
    / "chembl_pxr_probe"
    / "outputs"
    / "external_judge"
    / "train_external_judge_features.csv"
)


def max_tanimoto(query_smiles: list[str], ref_smiles: list[str]) -> np.ndarray:
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    query_mols = [Chem.MolFromSmiles(smi) for smi in query_smiles]
    ref_mols = [Chem.MolFromSmiles(smi) for smi in ref_smiles]
    query_fps = [gen.GetFingerprint(mol) for mol in query_mols if mol is not None]
    ref_fps = [gen.GetFingerprint(mol) for mol in ref_mols if mol is not None]
    out = np.zeros(len(query_fps), dtype=np.float64)
    for i, fp in enumerate(query_fps):
        sims = DataStructs.BulkTanimotoSimilarity(fp, ref_fps)
        out[i] = max(sims) if sims else 0.0
    return out


def build_frame() -> pd.DataFrame:
    train = load_train_smiles_with_counter()
    test = load_test_smiles()
    y = train["pec50"].to_numpy(dtype=np.float64)
    selectivity = y - train["counter_pec50"].to_numpy(dtype=np.float64)
    smiles = train["smiles"].tolist()

    train_ids = pd.read_sql(
        "SELECT compound_id FROM train_activity ORDER BY id", get_engine()
    )["compound_id"].astype(int)
    lf = pd.read_parquet(LF_PATH).loc[train_ids].reset_index(drop=True)

    potent_mask = (y >= 6.0) & (np.nan_to_num(selectivity, nan=-np.inf) >= 1.5)
    potent_smiles = [smi for smi, keep in zip(smiles, potent_mask) if keep]
    chembl = pd.read_csv(CHEMBL_FEATURES)

    _, p_adv = adversarial_split_indices(
        smiles,
        test["smiles"].tolist(),
        n_splits=5,
        n_top=849,
        seed=42,
    )
    return pd.DataFrame(
        {
            "smiles": smiles,
            "pec50": y,
            "selectivity": selectivity,
            "is_potent46": potent_mask,
            "test_nn": max_tanimoto(smiles, test["smiles"].tolist()),
            "potent_nn": max_tanimoto(smiles, potent_smiles),
            "adv_p_test": p_adv,
            "log2fc_33_pred": lf["log2fc_33_pred"].to_numpy(dtype=np.float64),
            "lf_mean": 0.5
            * (
                lf["log2fc_8p25_pred"].to_numpy(dtype=np.float64)
                + lf["log2fc_33_pred"].to_numpy(dtype=np.float64)
            ),
            "chembl_ext_nn": chembl["external_nn_tanimoto"].to_numpy(dtype=np.float64),
        }
    )


def split_registry(
    frame: pd.DataFrame,
) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    smiles = frame["smiles"].tolist()
    y = frame["pec50"].to_numpy(dtype=np.float64)
    selectivity = frame["selectivity"].to_numpy(dtype=np.float64)
    test_smiles = load_test_smiles()["smiles"].tolist()
    registry = {
        "umap_canonical": umap_split_indices(smiles, n_splits=5, seed=42),
    }
    for t in (0.20, 0.25, 0.30):
        registry[f"mixed_analog_t{int(t * 100)}"] = (
            mixed_analog_diversity_split_indices(
                smiles, y, selectivity, n_splits=5, analog_tanimoto_threshold=t, seed=42
            )
        )
    for t in (0.20, 0.25, 0.30):
        registry[f"test_nn_t{int(t * 100)}"] = test_nn_split_indices(
            smiles, test_smiles, n_splits=5, test_nn_threshold=t, seed=42
        )
    for n_top in (513, 849, 1200):
        splits, _ = adversarial_split_indices(
            smiles, test_smiles, n_splits=5, n_top=n_top, seed=42
        )
        registry[f"adversarial_top{n_top}"] = splits
    n = len(frame)
    all_idx = np.arange(n, dtype=np.int64)

    def holdout_by_score(name: str, score: np.ndarray, size: int) -> None:
        val_idx = np.argsort(-score)[:size].astype(np.int64)
        train_idx = np.setdiff1d(all_idx, val_idx, assume_unique=False).astype(np.int64)
        registry[name] = [(train_idx, val_idx)]

    def rank01(values: pd.Series) -> np.ndarray:
        return values.rank(method="average", pct=True).to_numpy(dtype=np.float64)

    hybrid_nolabel = (
        rank01(frame["adv_p_test"])
        + rank01(frame["test_nn"])
        + 0.5 * rank01(frame["log2fc_33_pred"])
        + 0.25 * rank01(frame["chembl_ext_nn"])
    )
    hybrid_with_y = hybrid_nolabel + 0.75 * rank01(frame["pec50"])
    for size in (513, 849, 1200):
        holdout_by_score(f"public_adv_top{size}", frame["adv_p_test"].to_numpy(), size)
        holdout_by_score(f"public_testnn_top{size}", frame["test_nn"].to_numpy(), size)
        holdout_by_score(
            f"public_log2fc_top{size}", frame["log2fc_33_pred"].to_numpy(), size
        )
        holdout_by_score(f"public_hybrid_nolabel_top{size}", hybrid_nolabel, size)
        holdout_by_score(f"public_hybrid_with_y_top{size}", hybrid_with_y, size)
    chembl_val = np.where(frame["chembl_ext_nn"].to_numpy(dtype=np.float64) >= 0.25)[0]
    if len(chembl_val) > 0:
        registry["public_chembl_ext_nn_ge025"] = [
            (
                np.setdiff1d(all_idx, chembl_val, assume_unique=False).astype(np.int64),
                chembl_val.astype(np.int64),
            )
        ]
    return registry


def summarize_split(
    frame: pd.DataFrame, name: str, splits: list[tuple[np.ndarray, np.ndarray]]
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for fold, (_tr, va) in enumerate(splits):
        val = frame.iloc[va]
        rows.append(
            {
                "split": name,
                "fold": fold,
                "n_val": int(len(val)),
                "pec50_mean": float(val["pec50"].mean()),
                "pec50_std": float(val["pec50"].std(ddof=1)),
                "pec50_top20_frac": float(
                    (val["pec50"] >= frame["pec50"].quantile(0.80)).mean()
                ),
                "potent46_count": int(val["is_potent46"].sum()),
                "test_nn_mean": float(val["test_nn"].mean()),
                "test_nn_ge_025_frac": float((val["test_nn"] >= 0.25).mean()),
                "potent_nn_ge_025_frac": float((val["potent_nn"] >= 0.25).mean()),
                "adv_p_test_mean": float(val["adv_p_test"].mean()),
                "adv_top20_frac": float(
                    (val["adv_p_test"] >= frame["adv_p_test"].quantile(0.80)).mean()
                ),
                "log2fc33_mean": float(val["log2fc_33_pred"].mean()),
                "log2fc33_top20_frac": float(
                    (
                        val["log2fc_33_pred"] >= frame["log2fc_33_pred"].quantile(0.80)
                    ).mean()
                ),
                "chembl_nn_ge_030_count": int((val["chembl_ext_nn"] >= 0.30).sum()),
            }
        )
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = build_frame()
    registry = split_registry(frame)
    rows: list[dict[str, float | str]] = []
    for name, splits in registry.items():
        rows.extend(summarize_split(frame, name, splits))
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "pseudo_public_split_fold_summary.csv", index=False)

    agg = (
        summary.groupby("split", as_index=False)
        .agg(
            n_val_mean=("n_val", "mean"),
            n_val_min=("n_val", "min"),
            n_val_max=("n_val", "max"),
            pec50_mean=("pec50_mean", "mean"),
            pec50_std_mean=("pec50_std", "mean"),
            pec50_top20_frac=("pec50_top20_frac", "mean"),
            pec50_top20_frac_std=("pec50_top20_frac", "std"),
            potent46_count_mean=("potent46_count", "mean"),
            test_nn_ge_025_frac=("test_nn_ge_025_frac", "mean"),
            test_nn_ge_025_frac_std=("test_nn_ge_025_frac", "std"),
            adv_p_test_mean=("adv_p_test_mean", "mean"),
            adv_top20_frac=("adv_top20_frac", "mean"),
            adv_top20_frac_std=("adv_top20_frac", "std"),
            log2fc33_top20_frac=("log2fc33_top20_frac", "mean"),
            log2fc33_top20_frac_std=("log2fc33_top20_frac", "std"),
            chembl_nn_ge_030_count_mean=("chembl_nn_ge_030_count", "mean"),
        )
        .sort_values(["adv_top20_frac", "test_nn_ge_025_frac"], ascending=False)
    )
    agg = agg.fillna(0.0)
    agg.to_csv(OUT_DIR / "pseudo_public_split_summary.csv", index=False)
    report = [
        "# Pseudo-Public Split Audit",
        "",
        "These splits do not use LB outcomes. They summarize candidate validation",
        "folds based on train/test geometry, potent-neighborhood structure,",
        "predicted log2fc, and weak ChEMBL coverage.",
        "",
        agg.to_markdown(index=False, floatfmt=".4f"),
        "",
    ]
    (OUT_DIR / "pseudo_public_split_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    print(agg.to_markdown(index=False, floatfmt=".4f"))
    print(f"\nWrote {OUT_DIR}")


if __name__ == "__main__":
    main()

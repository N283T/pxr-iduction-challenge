"""Pre-model CV diagnostic: evaluate any split scheme against the
three failure modes that killed PR #69 analog CV.

For every named split (list of (train_idx, val_idx) tuples), compute
per-fold and aggregate:

  1. val_size -- warn if very small (< 300) or very large (> 1500).
  2. val_y_dispersion = E|y_val - median(y_val)|.
     LB reference is 0.797. Below 0.70 flags "narrow val" (the
     analog-CV failure mode).
  3. val_to_train_NN_mean -- mean max Morgan r=2/2048 Tanimoto
     from each val compound to its closest train compound. Above
     0.40 flags "easy val" (val is basically interpolation from
     train).
  4. coverage = |union(val)| / n_train. Below 1.0 means some train
     rows never appear in any val; OOF is partial and the fold
     cannot be used as ensemble input.

Usage:
  pixi run python track1_activity/scripts/eda_cv_prep/05_split_diagnostic.py

Output:
  - prints a markdown table to stdout (aggregate row per split)
  - data/eda_cv_prep/split_diagnostic_fold.csv (per-fold detail)
  - data/eda_cv_prep/split_diagnostic_summary.csv (per-split aggregate)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import psycopg2
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
from data import DB_PARAMS  # noqa: E402
from splits import (  # noqa: E402
    analog_aware_split_indices,
    mixed_analog_diversity_split_indices,
    scaffold_split_indices,
    umap_split_indices,
)

OUT_DIR = REPO_ROOT.joinpath("data", "eda_cv_prep")
OUT_DIR.mkdir(parents=True, exist_ok=True)


LB_Y_DISPERSION = 0.797
NARROW_VAL_THRESHOLD = 0.70
EASY_VAL_THRESHOLD = 0.40
SMALL_VAL_THRESHOLD = 300
LARGE_VAL_THRESHOLD = 1500


def load_train() -> pd.DataFrame:
    conn = psycopg2.connect(**DB_PARAMS)
    try:
        df = pd.read_sql(
            """
            SELECT t.compound_id, c.std_smiles AS smiles, t.pec50,
                   ca.pec50 AS counter_pec50
            FROM train_activity t
            JOIN compounds c ON t.compound_id = c.id
            LEFT JOIN counter_assay ca ON ca.compound_id = t.compound_id
            ORDER BY t.id
            """,
            conn,
        )
    finally:
        conn.close()
    df["selectivity"] = df["pec50"] - df["counter_pec50"]
    return df


def build_morgan_fps(smiles_list: list[str]):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles_list]


def val_y_dispersion(y_val: np.ndarray) -> float:
    return float(np.mean(np.abs(y_val - np.median(y_val))))


def val_to_train_nn_mean(
    val_idx: np.ndarray,
    train_idx: np.ndarray,
    all_fps,
) -> float:
    train_fps = [all_fps[i] for i in train_idx]
    sims_per_val = []
    for i in val_idx:
        sims = DataStructs.BulkTanimotoSimilarity(all_fps[i], train_fps)
        sims_per_val.append(max(sims) if sims else 0.0)
    return float(np.mean(sims_per_val))


def diagnose_split(
    splits: list[tuple[np.ndarray, np.ndarray]],
    pec50: np.ndarray,
    all_fps,
    n_total: int,
) -> tuple[pd.DataFrame, dict]:
    fold_rows = []
    all_val_idx: list[int] = []
    for k, (train_idx, val_idx) in enumerate(splits):
        y_val = pec50[val_idx]
        disp = val_y_dispersion(y_val)
        nn = val_to_train_nn_mean(val_idx, train_idx, all_fps)
        fold_rows.append(
            {
                "fold": k,
                "val_size": len(val_idx),
                "train_size": len(train_idx),
                "val_y_dispersion": disp,
                "val_to_train_nn_mean": nn,
            }
        )
        all_val_idx.extend(val_idx.tolist())

    fold_df = pd.DataFrame(fold_rows)
    coverage = len(set(all_val_idx)) / n_total
    summary = {
        "val_size_mean": fold_df["val_size"].mean(),
        "val_size_min": fold_df["val_size"].min(),
        "val_size_max": fold_df["val_size"].max(),
        "val_y_dispersion_mean": fold_df["val_y_dispersion"].mean(),
        "val_y_dispersion_std": fold_df["val_y_dispersion"].std(),
        "val_to_train_nn_mean": fold_df["val_to_train_nn_mean"].mean(),
        "coverage": coverage,
    }
    return fold_df, summary


def flag_warnings(s: dict) -> list[str]:
    warnings = []
    if s["val_size_min"] < SMALL_VAL_THRESHOLD:
        warnings.append(f"SMALL_VAL (min={s['val_size_min']})")
    if s["val_size_max"] > LARGE_VAL_THRESHOLD:
        warnings.append(f"LARGE_VAL (max={s['val_size_max']})")
    if s["val_y_dispersion_mean"] < NARROW_VAL_THRESHOLD:
        warnings.append(
            f"NARROW_VAL (disp={s['val_y_dispersion_mean']:.3f} "
            f"< {NARROW_VAL_THRESHOLD}, LB={LB_Y_DISPERSION})"
        )
    if s["val_to_train_nn_mean"] > EASY_VAL_THRESHOLD:
        warnings.append(
            f"EASY_VAL (NN={s['val_to_train_nn_mean']:.3f} > {EASY_VAL_THRESHOLD})"
        )
    if s["coverage"] < 1.0 - 1e-6:
        warnings.append(f"PARTIAL_COVERAGE ({s['coverage']:.3f})")
    return warnings


def main() -> None:
    print("Loading train data...")
    df = load_train()
    smiles = df["smiles"].tolist()
    pec50 = df["pec50"].to_numpy(dtype=np.float64)
    selectivity = df["selectivity"].to_numpy(dtype=np.float64)
    n = len(df)
    print(f"  n_train = {n}")

    print("Building Morgan FPs for diagnostic NN distances...")
    all_fps = build_morgan_fps(smiles)

    schemes: dict[str, Callable] = {
        "umap_canonical": lambda: umap_split_indices(
            smiles, n_splits=5, n_clusters=50, seed=42
        ),
        "scaffold": lambda: scaffold_split_indices(smiles, n_splits=5, seed=42),
        "analog_t025": lambda: analog_aware_split_indices(
            smiles,
            pec50,
            selectivity,
            n_splits=5,
            analog_tanimoto_threshold=0.25,
            seed=42,
        ),
        "mixed_t025": lambda: mixed_analog_diversity_split_indices(
            smiles,
            pec50,
            selectivity,
            n_splits=5,
            analog_tanimoto_threshold=0.25,
            seed=42,
        ),
    }

    per_fold_frames = []
    summaries = []
    for name, build in schemes.items():
        print(f"\n=== {name} ===")
        splits = build()
        fold_df, s = diagnose_split(splits, pec50, all_fps, n)
        fold_df["scheme"] = name
        per_fold_frames.append(fold_df)
        warnings = flag_warnings(s)
        summary_row = {"scheme": name, **s, "warnings": "; ".join(warnings) or "-"}
        summaries.append(summary_row)
        print(fold_df.to_string(index=False))
        print(f"  warnings: {warnings or 'none'}")

    summary_df = pd.DataFrame(summaries)
    all_folds_df = pd.concat(per_fold_frames, ignore_index=True)

    summary_path = OUT_DIR.joinpath("split_diagnostic_summary.csv")
    folds_path = OUT_DIR.joinpath("split_diagnostic_fold.csv")
    summary_df.to_csv(summary_path, index=False)
    all_folds_df.to_csv(folds_path, index=False)

    print("\n=== SUMMARY ===")
    print(
        summary_df[
            [
                "scheme",
                "val_size_mean",
                "val_y_dispersion_mean",
                "val_to_train_nn_mean",
                "coverage",
                "warnings",
            ]
        ].to_string(index=False)
    )
    print(f"\nWrote {summary_path}")
    print(f"Wrote {folds_path}")


if __name__ == "__main__":
    main()

"""Analog distance distribution: test compounds vs train subsets.

Hypothesis (from CLAUDE.md + docs/aux_revisit_observation.md):
  Test compounds = 46 potent drug-like inducers (pEC50>=6 AND sel>=1.5)
  plus their Enamine/FDA analogs.

  If true, test's nearest-neighbor distance distribution to the
  "potent-46" subset should be significantly tighter than its NN
  distance to the rest of train. This informs whether an
  analog-aware CV split (holding out val compounds that are close
  to train "potents") would correlate better with LB than the
  current Morgan+UMAP+KMeans split.

Usage:
  pixi run python track1_activity/scripts/eda_cv_prep/01_test_analog_distance.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psycopg2
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
from data import DB_PARAMS  # noqa: E402

OUT_DIR = REPO_ROOT.joinpath("data", "eda_cv_prep")
FIG_DIR = REPO_ROOT.joinpath("docs", "figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

POTENT_PEC50 = 6.0
POTENT_SEL = 1.5


def load_sets() -> dict[str, pd.DataFrame]:
    conn = psycopg2.connect(**DB_PARAMS)
    try:
        train = pd.read_sql(
            """
            SELECT t.compound_id, c.std_smiles, t.pec50,
                   (t.pec50 - ca.pec50) AS selectivity
            FROM train_activity t
            JOIN compounds c ON t.compound_id = c.id
            LEFT JOIN counter_assay ca ON t.compound_id = ca.compound_id
            ORDER BY t.compound_id
            """,
            conn,
        )
        test = pd.read_sql(
            """
            SELECT te.compound_id, c.std_smiles
            FROM test_activity te
            JOIN compounds c ON te.compound_id = c.id
            ORDER BY te.compound_id
            """,
            conn,
        )
    finally:
        conn.close()
    return {"train": train, "test": test}


def morgan_fps(smiles_iter) -> list:
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = []
    for s in smiles_iter:
        mol = Chem.MolFromSmiles(s)
        fps.append(gen.GetFingerprint(mol) if mol is not None else None)
    return fps


def nn_similarity(query_fps: list, ref_fps: list) -> np.ndarray:
    """For each query fp, return max Tanimoto vs any ref fp."""
    ref_valid = [f for f in ref_fps if f is not None]
    out = np.zeros(len(query_fps), dtype=np.float64)
    for i, qf in enumerate(query_fps):
        if qf is None:
            out[i] = np.nan
            continue
        sims = DataStructs.BulkTanimotoSimilarity(qf, ref_valid)
        out[i] = max(sims) if sims else np.nan
    return out


def summarize(arr: np.ndarray, name: str) -> dict:
    arr = arr[~np.isnan(arr)]
    return {
        "set": name,
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p05": float(np.quantile(arr, 0.05)),
        "p25": float(np.quantile(arr, 0.25)),
        "median": float(np.quantile(arr, 0.50)),
        "p75": float(np.quantile(arr, 0.75)),
        "p95": float(np.quantile(arr, 0.95)),
        "max": float(arr.max()),
    }


def main() -> None:
    data = load_sets()
    train = data["train"]
    test = data["test"]

    potent_mask = (train["pec50"] >= POTENT_PEC50) & (
        train["selectivity"].fillna(-np.inf) >= POTENT_SEL
    )
    potent_train = train.loc[potent_mask].reset_index(drop=True)
    nonpotent_train = train.loc[~potent_mask].reset_index(drop=True)

    print(
        f"train total={len(train)}, potent (pEC50>={POTENT_PEC50} & sel>={POTENT_SEL})={len(potent_train)}, "
        f"nonpotent={len(nonpotent_train)}, test={len(test)}"
    )

    print("Computing Morgan r=2, 2048-bit fingerprints...")
    test_fps = morgan_fps(test["std_smiles"])
    potent_fps = morgan_fps(potent_train["std_smiles"])
    nonpotent_fps = morgan_fps(nonpotent_train["std_smiles"])
    train_fps = morgan_fps(train["std_smiles"])

    print("Computing NN similarities...")
    nn_all = nn_similarity(test_fps, train_fps)
    nn_potent = nn_similarity(test_fps, potent_fps)
    nn_nonpotent = nn_similarity(test_fps, nonpotent_fps)

    per_test = pd.DataFrame(
        {
            "compound_id": test["compound_id"],
            "std_smiles": test["std_smiles"],
            "nn_tanimoto_all_train": nn_all,
            "nn_tanimoto_potent": nn_potent,
            "nn_tanimoto_nonpotent": nn_nonpotent,
        }
    )
    per_test["closer_to_potent"] = (
        per_test["nn_tanimoto_potent"] > per_test["nn_tanimoto_nonpotent"]
    )
    per_test.to_parquet(OUT_DIR.joinpath("test_analog_distance.parquet"), index=False)

    summary = pd.DataFrame(
        [
            summarize(nn_all, "test_vs_all_train"),
            summarize(nn_potent, "test_vs_potent46"),
            summarize(nn_nonpotent, "test_vs_nonpotent"),
        ]
    )
    print("\n=== NN Tanimoto summary ===")
    print(summary.to_string(index=False))
    summary.to_csv(OUT_DIR.joinpath("test_analog_distance_summary.csv"), index=False)

    closer_potent = int(per_test["closer_to_potent"].sum())
    print(
        f"\nTest compounds whose NN is the potent-46 subset: "
        f"{closer_potent} / {len(per_test)} ({100 * closer_potent / len(per_test):.1f}%)"
    )

    # Top-5 NN to potent summary per test
    print("\nComputing top-5 NN to potent-46 per test compound...")
    potent_fps_valid = [f for f in potent_fps if f is not None]
    top5_means = np.zeros(len(test_fps))
    for i, qf in enumerate(test_fps):
        if qf is None:
            top5_means[i] = np.nan
            continue
        sims = DataStructs.BulkTanimotoSimilarity(qf, potent_fps_valid)
        k = min(5, len(sims))
        top5_means[i] = float(np.sort(sims)[-k:].mean()) if k else np.nan
    per_test["top5_tanimoto_potent"] = top5_means
    per_test.to_parquet(OUT_DIR.joinpath("test_analog_distance.parquet"), index=False)

    # Figure: three overlaid density histograms
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    bins = np.linspace(0, 1, 41)
    ax.hist(
        nn_all[~np.isnan(nn_all)],
        bins=bins,
        alpha=0.6,
        density=True,
        label=f"vs all train (n={len(train)})",
        color="#1f77b4",
    )
    ax.hist(
        nn_nonpotent[~np.isnan(nn_nonpotent)],
        bins=bins,
        alpha=0.5,
        density=True,
        label=f"vs non-potent (n={len(nonpotent_train)})",
        color="#ff7f0e",
    )
    ax.hist(
        nn_potent[~np.isnan(nn_potent)],
        bins=bins,
        alpha=0.7,
        density=True,
        label=f"vs potent-46",
        color="#2ca02c",
    )
    ax.set_xlabel("NN Tanimoto similarity (Morgan r=2, 2048 bits)")
    ax.set_ylabel("density")
    ax.set_title("Test → train NN similarity by subset")
    ax.legend(loc="upper right", fontsize=9)

    ax = axes[1]
    ax.scatter(
        nn_nonpotent, nn_potent, s=5, alpha=0.4, color="#555555"
    )
    diag = np.linspace(0, 1, 50)
    ax.plot(diag, diag, "k--", lw=1, alpha=0.5)
    ax.set_xlabel("NN Tanimoto to non-potent train")
    ax.set_ylabel("NN Tanimoto to potent-46 train")
    ax.set_title("Per-test compound: potent vs non-potent NN")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")

    plt.tight_layout()
    fig_path = FIG_DIR.joinpath("eda_cv_prep_01_test_analog_distance.png")
    plt.savefig(fig_path, dpi=130)
    print(f"\nSaved figure to {fig_path}")

    # A secondary figure: distribution of nn_potent split by test pEC50 is
    # impossible (test pEC50 is blinded). Instead, bin by nn_potent itself
    # and show count.
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(nn_potent[~np.isnan(nn_potent)], bins=bins, color="#2ca02c", alpha=0.85)
    ax.axvline(
        np.nanmedian(nn_potent), color="black", lw=1, ls="--", label="median"
    )
    ax.set_xlabel("NN Tanimoto to potent-46 train")
    ax.set_ylabel("# test compounds")
    ax.set_title(f"How close is each test compound to the 46 potent train seeds?")
    ax.legend()
    plt.tight_layout()
    fig2_path = FIG_DIR.joinpath("eda_cv_prep_01_test_vs_potent46_hist.png")
    plt.savefig(fig2_path, dpi=130)
    print(f"Saved figure to {fig2_path}")


if __name__ == "__main__":
    main()

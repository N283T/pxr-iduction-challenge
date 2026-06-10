#!/usr/bin/env -S pixi run python
"""Blend AS1-augmented Phase 2 predictions back toward the id55 anchor."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SUBMISSION_DIR = REPO_ROOT / "track1_activity" / "submissions"
ANCHOR_PATH = SUBMISSION_DIR / "ens_id51_top500_potent46_t40_soft_g35.csv"
DEFAULT_MODEL_PATH = (
    SUBMISSION_DIR
    / "phase2_as1_aug_cheme_2d_full_boltz_log2fc_pred_seed10ens_top500_tabpfnv3_ne8_t0p7_model_only.csv"
)


def load_as1_labels() -> pd.DataFrame:
    from data import get_engine

    return pd.read_sql(
        """
        SELECT c.molecule_name, l.pec50 AS as1_pec50
        FROM test_activity_phase1_labels l
        JOIN compounds c ON c.id = l.compound_id
        """,
        get_engine(),
    )


def load_submission(path: Path, column: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.rename(columns={"Molecule Name": "molecule_name", "pEC50": column})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--anchor", type=Path, default=ANCHOR_PATH)
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=[0.10, 0.20, 0.30, 0.40, 0.50],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import sys

    sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))

    anchor = load_submission(args.anchor, "anchor_pred")
    model = load_submission(args.model, "model_pred")
    as1 = load_as1_labels().set_index("molecule_name")["as1_pec50"]
    merged = anchor.merge(model[["molecule_name", "model_pred"]], on="molecule_name")
    is_as1 = merged["molecule_name"].isin(as1.index).to_numpy()
    delta = merged["model_pred"].to_numpy(dtype=np.float64) - merged[
        "anchor_pred"
    ].to_numpy(dtype=np.float64)

    rows = []
    for alpha in args.alphas:
        pred = merged["anchor_pred"].to_numpy(dtype=np.float64) + alpha * delta
        pred[is_as1] = merged.loc[is_as1, "molecule_name"].map(as1).to_numpy(
            dtype=np.float64
        )
        suffix = str(alpha).replace(".", "p")
        out_path = (
            SUBMISSION_DIR
            / f"phase2_as1_aug_top500_id55blend_a{suffix}_labels_as1.csv"
        )
        out = pd.DataFrame(
            {
                "SMILES": merged["SMILES"],
                "Molecule Name": merged["molecule_name"],
                "pEC50": pred,
            }
        )
        out.to_csv(out_path, index=False)
        rows.append(
            {
                "alpha": alpha,
                "path": str(out_path.relative_to(REPO_ROOT)),
                "as2_mean_abs_shift": float(np.abs(alpha * delta[~is_as1]).mean()),
                "as2_p90_abs_shift": float(np.quantile(np.abs(alpha * delta[~is_as1]), 0.90)),
                "as2_max_abs_shift": float(np.abs(alpha * delta[~is_as1]).max()),
            }
        )

    summary = pd.DataFrame(rows)
    summary_path = (
        REPO_ROOT
        / "track1_activity"
        / "analysis"
        / "phase2_as1_augmented"
        / "as1_aug_id55_blend_summary.csv"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()

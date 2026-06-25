#!/usr/bin/env -S pixi run python
"""Apply the selected Phase 2 pairrank high-activity gate to a submission."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SUBMISSION_DIR = REPO_ROOT / "track1_activity" / "submissions"
PAIRRANK_PATH = (
    REPO_ROOT
    / "track1_activity"
    / "analysis"
    / "phase2_classifier_gate"
    / "outputs"
    / "pairwise_assay_rank"
    / "all_pxr_chembl_htchem_single_conc_mpa1500_top64"
    / "test_pairrank_scores.csv"
)
DEFAULT_BASE = (
    SUBMISSION_DIR / "phase2_as1_aug_top500_id55blend_a0p4_labels_as1.csv"
)
DEFAULT_NAME = (
    "phase2_as1_aug_top500_id55blend_a0p4_"
    "pairrankchembl_q95_g0p15_labels_as1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--pairrank", type=Path, default=PAIRRANK_PATH)
    parser.add_argument("--threshold", type=float, default=0.58878)
    parser.add_argument("--shift", type=float, default=0.15)
    parser.add_argument("--name", default=DEFAULT_NAME)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = pd.read_csv(args.base)
    pairrank = pd.read_csv(args.pairrank)[
        ["test_id", "molecule_name", "split", "as1_pec50", "pairrank_chembl"]
    ].rename(columns={"molecule_name": "Molecule Name"})
    merged = base.merge(pairrank, on="Molecule Name", how="left", validate="one_to_one")
    if merged["pairrank_chembl"].isna().any():
        missing = merged.loc[merged["pairrank_chembl"].isna(), "Molecule Name"].head()
        raise ValueError(f"Missing pairrank rows for: {missing.tolist()}")

    is_as2 = merged["split"].eq("AS2").to_numpy()
    flag = is_as2 & merged["pairrank_chembl"].ge(args.threshold).to_numpy()
    pred = merged["pEC50"].to_numpy(dtype=np.float64).copy()
    pred[flag] += args.shift

    out = base.copy()
    out["pEC50"] = pred
    out_path = SUBMISSION_DIR / f"{args.name}.csv"
    out.to_csv(out_path, index=False)

    audit = merged[
        [
            "test_id",
            "Molecule Name",
            "split",
            "as1_pec50",
            "pEC50",
            "pairrank_chembl",
        ]
    ].copy()
    audit["pairrank_gate_flag"] = flag
    audit["pairrank_gate_shift"] = np.where(flag, args.shift, 0.0)
    audit["gated_pEC50"] = pred
    audit_dir = (
        REPO_ROOT
        / "track1_activity"
        / "analysis"
        / "phase2_pairrank_gate"
        / args.name
    )
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(audit_dir / "gate_audit.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "base": str(args.base.relative_to(REPO_ROOT)),
                "output": str(out_path.relative_to(REPO_ROOT)),
                "threshold": args.threshold,
                "shift": args.shift,
                "n_total": len(out),
                "n_as1": int(merged["split"].eq("AS1").sum()),
                "n_as2": int(is_as2.sum()),
                "n_flags": int(flag.sum()),
                "as2_flag_fraction": float(flag.sum() / max(is_as2.sum(), 1)),
                "mean_shift_all": float((pred - merged["pEC50"]).mean()),
                "mean_shift_as2": float((pred[is_as2] - merged.loc[is_as2, "pEC50"]).mean()),
                "max_shift": float(np.max(np.abs(pred - merged["pEC50"]))),
            }
        ]
    )
    summary.to_csv(audit_dir / "summary.csv", index=False)
    print(summary.to_string(index=False))
    print("\nFlagged AS2 compounds")
    print(
        audit.loc[flag]
        .sort_values("pairrank_chembl", ascending=False)
        .to_string(index=False)
    )
    print(f"\nWrote {out_path}")
    print(f"Wrote {audit_dir}")


if __name__ == "__main__":
    main()

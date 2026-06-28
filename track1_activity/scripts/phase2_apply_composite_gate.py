#!/usr/bin/env -S pixi run python
"""Apply selected Phase 2 composite scalar gates to a submission."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SUBMISSION_DIR = REPO_ROOT / "track1_activity" / "submissions"
DEFAULT_GATE_TABLE = (
    REPO_ROOT
    / "track1_activity"
    / "analysis"
    / "phase2_classifier_gate"
    / "outputs"
    / "composite_pairrank_chemprop_train_quantile_gates"
    / "combo_q98_lift02__cp_abs01_lowq98_drop02_shift.csv"
)
DEFAULT_BASE = SUBMISSION_DIR / "phase2_as1_aug_top500_id55blend_a0p4_labels_as1.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--gate-table", type=Path, default=DEFAULT_GATE_TABLE)
    parser.add_argument("--high-shift", type=float, default=0.20)
    parser.add_argument("--low-shift", type=float, default=-0.20)
    parser.add_argument("--apply-split", default="AS2")
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--exclude-high-if-shifted-from",
        type=Path,
        default=None,
        help="Exclude high flags on rows already moved in base relative to this CSV.",
    )
    parser.add_argument("--exclude-threshold", type=float, default=1e-8)
    return parser.parse_args()


def _load_prediction(path: Path, column: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"SMILES", "Molecule Name", "pEC50"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return df[["SMILES", "Molecule Name", "pEC50"]].rename(columns={"pEC50": column})


def main() -> None:
    args = parse_args()
    base = pd.read_csv(args.base)
    required_base = {"SMILES", "Molecule Name", "pEC50"}
    missing_base = required_base.difference(base.columns)
    if missing_base:
        raise ValueError(f"{args.base} missing columns: {sorted(missing_base)}")

    gates = pd.read_csv(args.gate_table)
    required_gates = {"molecule_name", "split", "high_flag", "low_flag"}
    missing_gates = required_gates.difference(gates.columns)
    if missing_gates:
        raise ValueError(f"{args.gate_table} missing columns: {sorted(missing_gates)}")

    gates = gates[list(required_gates)].rename(
        columns={"molecule_name": "Molecule Name"}
    )
    merged = base.merge(gates, on="Molecule Name", how="left", validate="one_to_one")
    if merged["split"].isna().any():
        missing = merged.loc[merged["split"].isna(), "Molecule Name"].head()
        raise ValueError(f"Missing gate rows for: {missing.tolist()}")

    apply_mask = merged["split"].eq(args.apply_split).to_numpy()
    high_flag = apply_mask & merged["high_flag"].astype(bool).to_numpy()
    low_flag = apply_mask & merged["low_flag"].astype(bool).to_numpy()

    already_shifted = np.zeros(len(merged), dtype=bool)
    if args.exclude_high_if_shifted_from is not None:
        reference = _load_prediction(
            args.exclude_high_if_shifted_from, "reference_pred"
        )
        aligned = merged[["SMILES", "Molecule Name", "pEC50"]].merge(
            reference, on=["SMILES", "Molecule Name"], how="left", validate="one_to_one"
        )
        if aligned["reference_pred"].isna().any():
            raise ValueError("Reference CSV did not align to base rows.")
        already_shifted = (
            np.abs(
                aligned["pEC50"].to_numpy(dtype=np.float64)
                - aligned["reference_pred"].to_numpy(dtype=np.float64)
            )
            > args.exclude_threshold
        )
        high_flag &= ~already_shifted

    base_pred = merged["pEC50"].to_numpy(dtype=np.float64)
    shift = np.zeros(len(merged), dtype=np.float64)
    shift[high_flag] += args.high_shift
    shift[low_flag] += args.low_shift

    out = base.copy()
    out["pEC50"] = base_pred + shift
    out_path = SUBMISSION_DIR / f"{args.name}.csv"
    out.to_csv(out_path, index=False)

    audit = merged[
        ["Molecule Name", "SMILES", "split", "pEC50", "high_flag", "low_flag"]
    ].copy()
    audit["already_shifted_from_reference"] = already_shifted
    audit["applied_high_flag"] = high_flag
    audit["applied_low_flag"] = low_flag
    audit["composite_gate_shift"] = shift
    audit["gated_pEC50"] = out["pEC50"]

    audit_dir = (
        REPO_ROOT / "track1_activity" / "analysis" / "phase2_composite_gate" / args.name
    )
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(audit_dir / "gate_audit.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "base": str(args.base.resolve().relative_to(REPO_ROOT)),
                "gate_table": str(args.gate_table.resolve().relative_to(REPO_ROOT)),
                "output": str(out_path.resolve().relative_to(REPO_ROOT)),
                "apply_split": args.apply_split,
                "high_shift": args.high_shift,
                "low_shift": args.low_shift,
                "n_total": len(out),
                "n_apply_split": int(apply_mask.sum()),
                "n_high_flags_raw": int(
                    (apply_mask & merged["high_flag"].astype(bool).to_numpy()).sum()
                ),
                "n_low_flags_raw": int(
                    (apply_mask & merged["low_flag"].astype(bool).to_numpy()).sum()
                ),
                "n_high_flags_applied": int(high_flag.sum()),
                "n_low_flags_applied": int(low_flag.sum()),
                "n_reference_shift_excluded": int(
                    (
                        apply_mask
                        & merged["high_flag"].astype(bool).to_numpy()
                        & already_shifted
                    ).sum()
                ),
                "mean_shift_all": float(shift.mean()),
                "mean_abs_shift_all": float(np.abs(shift).mean()),
                "max_abs_shift": float(np.abs(shift).max()),
            }
        ]
    )
    summary.to_csv(audit_dir / "summary.csv", index=False)

    print(summary.to_string(index=False))
    print("\nApplied AS2 shifts")
    print(
        audit.loc[np.abs(shift) > 0]
        .sort_values("composite_gate_shift", ascending=False)
        .to_string(index=False)
    )
    print(f"\nWrote {out_path}")
    print(f"Wrote {audit_dir}")


if __name__ == "__main__":
    main()

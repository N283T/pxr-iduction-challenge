#!/usr/bin/env -S pixi run python
"""Build the master DataFrame for Issue #52 re-EDA.

Joins compounds + descriptors + train/test/counter/single-conc activity +
Boltz-2 pose/affinity + PoseBusters summary into a single wide parquet.

Usage:
    pixi run python track1_activity/scripts/eda_redo_00_build_master.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.joinpath("src")))

from eda_redo import build_master_df, save_master


def _print_summary(df) -> None:
    n = len(df)
    print(f"[eda_redo] master: {n:>6,d} compounds")
    print("[eda_redo] split breakdown:")
    for label, count in df["split"].value_counts().sort_index().items():
        print(f"  {label:<12s} {count:>6,d}")
    print()
    print("[eda_redo] activity coverage:")
    print(f"  train pEC50 non-null: {df['train_pec50'].notna().sum():>6,d}")
    print(f"  counter pEC50 non-null: {df['counter_pec50'].notna().sum():>6,d}")
    print(f"  single_conc aggregated: {df['single_n_rows'].notna().sum():>6,d}")
    print(
        f"  train is_potent (pEC50 >= 6): {df['train_is_potent'].fillna(False).sum():>6,d}"
    )
    print()
    print("[eda_redo] Boltz-2 coverage:")
    b2 = df[df["b2_confidence"].notna() | df["b2_preprocessing_failed"].fillna(False)]
    print(f"  boltz2 rows: {len(b2):>6,d}")
    print(
        f"  preprocessing_failed: {df['b2_preprocessing_failed'].fillna(False).sum():>6,d}"
    )
    print(
        f"  ligand_oversize:       {df['b2_ligand_oversize'].fillna(False).sum():>6,d}"
    )
    print(f"  posebusters all_passed: {df['pb_all_passed'].fillna(False).sum():>6,d}")


def main() -> None:
    df = build_master_df()
    path = save_master(df)
    print(f"[eda_redo] wrote {path}")
    _print_summary(df)


if __name__ == "__main__":
    main()

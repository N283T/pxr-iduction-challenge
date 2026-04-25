#!/usr/bin/env -S pixi run python
"""Generate Track 2 Boltz-2 input YAMLs.

Produces two distinct input directories so we can invoke ``boltz predict``
on either independently:

  * ``inputs/apo/apo.yaml``   — ligand-free LBD prediction. Run this once
                                with ``--use_msa_server`` to populate the
                                ColabFold MSA cache and to give us an apo
                                LBD model for analysis.
  * ``inputs/holo/<id>.yaml`` — 184 protein-ligand YAMLs, one per Track 2
                                test compound. ``<id>`` matches the
                                ``structure`` column of the parquet, so the
                                downstream PDB output is named consistently
                                with the required submission format.

Usage:
    pixi run python track2_structure/scripts/build_inputs.py            # both
    pixi run python track2_structure/scripts/build_inputs.py --mode apo
    pixi run python track2_structure/scripts/build_inputs.py --mode holo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT.joinpath("track2_structure", "src")))

from track2.constants import TRACK2_INPUT_DIR  # noqa: E402
from track2.input_builder import write_yaml  # noqa: E402


def _build_apo(out_root: Path) -> None:
    apo_dir = out_root.joinpath("apo")
    write_yaml(apo_dir.joinpath("apo.yaml"), smiles=None)
    print(f"Wrote {apo_dir.joinpath('apo.yaml')}")


def _build_holo(out_root: Path, parquet_path: Path) -> None:
    df = pd.read_parquet(parquet_path)
    if df.empty:
        raise ValueError(f"no rows in {parquet_path}")
    holo_dir = out_root.joinpath("holo")
    holo_dir.mkdir(parents=True, exist_ok=True)
    for _, row in df.iterrows():
        sid = row["structure"]
        write_yaml(holo_dir.joinpath(f"{sid}.yaml"), smiles=row["smiles"])
    print(f"Wrote {len(df)} holo YAMLs under {holo_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--mode",
        choices=["apo", "holo", "all"],
        default="all",
        help="Which YAML set to build (default: all).",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=REPO_ROOT.joinpath("data", "structure_test.parquet"),
        help="Path to the Track 2 SMILES parquet.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=TRACK2_INPUT_DIR,
        help="Root output directory (creates apo/ and holo/ inside).",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.mode in ("apo", "all"):
        _build_apo(args.out_dir)
    if args.mode in ("holo", "all"):
        _build_holo(args.out_dir, args.data)


if __name__ == "__main__":
    main()

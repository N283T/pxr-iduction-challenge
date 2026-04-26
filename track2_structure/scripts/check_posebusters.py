#!/usr/bin/env -S pixi run python
"""Run PoseBusters validity checks on Track 2 Boltz-2 outputs.

For each compound, splits the Boltz model PDB (which contains both protein
chain A and ligand chain B with residue name LIG) into a protein-only PDB
and a ligand SDF (with bond orders re-assigned from the input SMILES via
``AllChem.AssignBondOrdersFromTemplate``), then runs the official
``PoseBusters(config="dock")`` battery and aggregates the 22 boolean
checks per compound.

By default scans only ``model_0`` (the highest-confidence Boltz pose).
Pass ``--all-models`` to score every model_0..4 — useful when we want to
re-rank pose selection by validity rather than confidence.

Usage:
    pixi run python track2_structure/scripts/check_posebusters.py
    pixi run python track2_structure/scripts/check_posebusters.py \\
        --workers 8 --all-models --out docs/track2_posebusters_all.csv
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT.joinpath("track2_structure", "src")))

import pandas as pd  # noqa: E402

from track2.constants import REPO_ROOT as PROJECT_ROOT  # noqa: E402

PRED_DIR = PROJECT_ROOT.joinpath(
    "structures",
    "boltz2_track2",
    "outputs",
    "holo",
    "boltz_results_holo",
    "predictions",
)
DEFAULT_DATA = PROJECT_ROOT.joinpath("data", "structure_test.parquet")


def _check_one(task: tuple[str, str, int]) -> dict[str, Any]:
    """Run PoseBusters on one (compound, model_idx) pair.

    Loaded inside the worker so the parent process stays light. Returns a
    flat dict with all PoseBusters columns plus compound id and model idx,
    or just an `error` key on failure.
    """
    import MDAnalysis as mda  # noqa: PLC0415
    from posebusters import PoseBusters  # noqa: PLC0415
    from rdkit import Chem, RDLogger  # noqa: PLC0415
    from rdkit.Chem import AllChem  # noqa: PLC0415

    RDLogger.DisableLog("rdApp.*")
    sid, smiles, model_idx = task
    pdb = PRED_DIR.joinpath(sid, f"{sid}_model_{model_idx}.pdb")
    base = {"compound": sid, "model": model_idx}
    if not pdb.exists():
        return {**base, "error": "pdb_missing"}
    try:
        with tempfile.TemporaryDirectory() as tdir:
            tdir = Path(tdir)
            u = mda.Universe(str(pdb))
            protein = u.select_atoms("protein")
            lig = u.select_atoms("resname LIG")
            if len(protein) == 0:
                return {**base, "error": "no_protein_atoms"}
            if len(lig) == 0:
                return {**base, "error": "no_ligand_atoms"}

            prot_path = tdir.joinpath("protein.pdb")
            lig_pdb_path = tdir.joinpath("ligand.pdb")
            lig_sdf_path = tdir.joinpath("ligand.sdf")
            protein.write(str(prot_path))
            lig.write(str(lig_pdb_path))

            ref_mol = Chem.MolFromSmiles(smiles)
            if ref_mol is None:
                return {**base, "error": "smiles_parse_failed"}
            lig_mol = Chem.MolFromPDBFile(
                str(lig_pdb_path), removeHs=True, sanitize=False
            )
            if lig_mol is None:
                return {**base, "error": "ligand_pdb_parse_failed"}
            try:
                lig_mol = AllChem.AssignBondOrdersFromTemplate(ref_mol, lig_mol)
            except Exception as exc:  # noqa: BLE001
                return {**base, "error": f"bond_order_failed: {exc}"}

            writer = Chem.SDWriter(str(lig_sdf_path))
            writer.write(lig_mol)
            writer.close()

            pb = PoseBusters(config="dock")
            result_df = pb.bust(
                mol_pred=str(lig_sdf_path),
                mol_cond=str(prot_path),
            )
            row = result_df.iloc[0].to_dict()
            row.update(base)
            row.pop("file", None)
            row.pop("molecule", None)
            return row
    except Exception as exc:  # noqa: BLE001
        return {**base, "error": f"failed: {type(exc).__name__}: {exc}"}


def _summarize(df: pd.DataFrame) -> None:
    if "error" in df.columns:
        n_err = df["error"].notna().sum()
        if n_err:
            print(f"\n[errors] {n_err} compounds failed:")
            for _, r in df[df["error"].notna()].iterrows():
                print(f"  {r['compound']:12s} model={r['model']}  {r['error']}")

    skip = {"compound", "model", "error", "position"}
    bool_cols = [
        c
        for c in df.columns
        if c not in skip
        and df[c].dropna().isin([True, False]).all()
        and df[c].notna().any()
    ]
    if not bool_cols:
        return
    print("\nPer-check pass rate (model_0 unless --all-models):")
    for c in sorted(bool_cols):
        n_pass = int(df[c].sum())
        n_total = int(df[c].notna().sum())
        pct = 100 * n_pass / n_total if n_total else 0
        bar = "#" * (n_pass * 30 // n_total) if n_total else ""
        print(f"  {c:42s}  {n_pass:3d}/{n_total:3d}  ({pct:5.1f}%) {bar}")

    df["all_passed"] = df[bool_cols].all(axis=1)
    df["n_passed"] = df[bool_cols].sum(axis=1)
    print(f"\nFully-passing compounds: {df['all_passed'].sum()} / {len(df)}")
    print(f"Mean passing checks: {df['n_passed'].mean():.1f} / {len(bool_cols)}")
    failed_any = df[~df["all_passed"]]
    if not failed_any.empty:
        print("\nCompounds with >=1 failure (top 20 by failures, model_0):")
        m0 = failed_any[failed_any["model"] == 0].sort_values("n_passed").head(20)
        for _, r in m0.iterrows():
            failing = [c for c in bool_cols if r[c] is False]
            print(
                f"  {r['compound']:12s}  passed={r['n_passed']}/{len(bool_cols)}  "
                f"failing={','.join(failing)}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help="SMILES parquet (compound id + smiles).",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Score every model_0..4 (5x more work). Default: only model_0.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit to first N compounds (for smoke testing).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Worker processes (default 8).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT.joinpath("docs", "track2_posebusters.csv"),
        help="Output CSV.",
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.data)
    if args.limit:
        df = df.head(args.limit)

    model_indices = list(range(5)) if args.all_models else [0]
    tasks: list[tuple[str, str, int]] = [
        (row.structure, row.smiles, mi)
        for _, row in df.iterrows()
        for mi in model_indices
    ]
    print(
        f"PoseBusters dock check: {len(tasks)} (compound, model) pairs, "
        f"workers={args.workers}"
    )

    results: list[dict[str, Any]] = []
    with mp.Pool(args.workers) as pool:
        for n, r in enumerate(pool.imap_unordered(_check_one, tasks, chunksize=2), 1):
            results.append(r)
            if n % 20 == 0 or n == len(tasks):
                print(f"  {n}/{len(tasks)}")

    out_df = (
        pd.DataFrame(results).sort_values(["compound", "model"]).reset_index(drop=True)
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"\nWrote {args.out} ({len(out_df)} rows)")
    _summarize(out_df)


if __name__ == "__main__":
    main()

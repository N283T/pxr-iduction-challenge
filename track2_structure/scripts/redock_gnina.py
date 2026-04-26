#!/usr/bin/env -S pixi run python
"""Locally refine each Boltz-2 ligand pose with gnina ``--minimize``.

For each compound, takes the highest-confidence Boltz pose (or a per-compound
selection from a CSV), splits the Boltz output PDB into protein-only +
ligand SDF (bond orders assigned from input SMILES), runs gnina with
``--minimize`` (local refinement + CNN rescoring), and assembles a new PDB
with the refined ligand pose grafted onto the Boltz protein chain. The
ligand residue is renamed back to ``LIG`` so the submission validator
accepts the result.

By default uses model_0 from every compound directory. Pass
``--selection-csv`` to use the per-compound model from a previous
``build_submission.py --strategy ...`` run (e.g. medoid-selected models).

Output goes to ``structures/boltz2_track2/redock_gnina/<id>/`` and a
combined refined PDB ready for submission lands in
``structures/boltz2_track2/redock_gnina/<id>/<id>_refined.pdb``.

Usage:
    pixi run python track2_structure/scripts/redock_gnina.py \\
        --workers 4
    pixi run python track2_structure/scripts/redock_gnina.py \\
        --workers 4 --selection-csv \\
        docs/track2_model_selection/track2_boltz2_medoid_v2_2026-04-26.csv
"""

from __future__ import annotations

import argparse
import gzip
import multiprocessing as mp
import os
import subprocess
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
REDOCK_DIR = PROJECT_ROOT.joinpath("structures", "boltz2_track2", "redock_gnina")
DEFAULT_DATA = PROJECT_ROOT.joinpath("data", "structure_test.parquet")
GNINA_BIN = Path.home().joinpath(".local", "bin", "gnina")
LD_PREFIX = PROJECT_ROOT.joinpath(".pixi", "envs", "default", "lib")


def _split_boltz_pdb(
    src_pdb: Path,
    smiles: str,
    out_protein_pdb: Path,
    out_ligand_sdf: Path,
) -> None:
    """Extract chain A protein PDB and ligand SDF (with bond orders) from a Boltz output."""
    import MDAnalysis as mda  # noqa: PLC0415
    from rdkit import Chem, RDLogger  # noqa: PLC0415
    from rdkit.Chem import AllChem  # noqa: PLC0415

    RDLogger.DisableLog("rdApp.*")
    u = mda.Universe(str(src_pdb))
    protein = u.select_atoms("protein")
    lig = u.select_atoms("resname LIG")
    out_protein_pdb.parent.mkdir(parents=True, exist_ok=True)
    protein.write(str(out_protein_pdb))

    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
        lig_pdb_path = Path(tmp.name)
    lig.write(str(lig_pdb_path))
    try:
        ref_mol = Chem.MolFromSmiles(smiles)
        if ref_mol is None:
            raise RuntimeError(f"smiles parse failed: {smiles}")
        lig_mol = Chem.MolFromPDBFile(str(lig_pdb_path), removeHs=True, sanitize=False)
        if lig_mol is None:
            raise RuntimeError("ligand pdb parse failed")
        lig_mol = AllChem.AssignBondOrdersFromTemplate(ref_mol, lig_mol)
        out_ligand_sdf.parent.mkdir(parents=True, exist_ok=True)
        writer = Chem.SDWriter(str(out_ligand_sdf))
        writer.write(lig_mol)
        writer.close()
    finally:
        lig_pdb_path.unlink(missing_ok=True)


def _run_gnina_minimize(
    receptor_pdb: Path,
    ligand_sdf: Path,
    out_sdf_gz: Path,
) -> dict[str, float]:
    """Invoke gnina --minimize. Returns parsed scores from stdout."""
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{LD_PREFIX}:{env.get('LD_LIBRARY_PATH', '')}"
    out_sdf_gz.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(GNINA_BIN),
        "--receptor",
        str(receptor_pdb),
        "--ligand",
        str(ligand_sdf),
        "--autobox_ligand",
        str(ligand_sdf),
        "--minimize",
        "--out",
        str(out_sdf_gz),
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"gnina exit {result.returncode}: {result.stderr[-500:]}")
    scores: dict[str, float] = {}
    for line in result.stdout.splitlines():
        line = line.strip()
        for prefix, key in [
            ("Affinity:", "vina_affinity"),
            ("RMSD:", "rmsd_to_input"),
            ("CNNscore:", "cnn_score"),
            ("CNNaffinity:", "cnn_affinity"),
            ("CNNvariance:", "cnn_variance"),
        ]:
            if line.startswith(prefix):
                parts = line.replace(prefix, "").split()
                if parts:
                    try:
                        scores[key] = float(parts[0])
                    except ValueError:
                        pass
                break
    return scores


def _assemble_refined_pdb(
    protein_pdb: Path,
    refined_sdf_gz: Path,
    out_pdb: Path,
) -> None:
    """Combine refined ligand SDF with protein PDB into a single PDB.

    Renames the ligand residue to LIG (chain B, resnum 1) so the official
    submission validator accepts the file. Preserves CONECT records from
    RDKit's PDB output and remaps their atom-serial references — without
    them RDKit's bond-perception in the validator infers bonds from
    distances and gets aromatic rings wrong, breaking the connectivity
    check on a handful of compounds.
    """
    from rdkit import Chem  # noqa: PLC0415

    with gzip.open(refined_sdf_gz, "rb") as f:
        sdf_bytes = f.read()
    sup = Chem.SDMolSupplier()
    sup.SetData(sdf_bytes.decode("utf-8"), removeHs=True, sanitize=True)
    mols = [m for m in sup if m is not None]
    if not mols:
        raise RuntimeError(f"no molecules in {refined_sdf_gz}")
    refined = mols[0]
    # Strip any remaining explicit hydrogens — gnina's --minimize sometimes
    # adds them, but the official validator parses the ligand with
    # ``removeHs=True`` and matches against a heavy-atom-only template, so
    # any extra H breaks AssignBondOrdersFromTemplate.
    refined = Chem.RemoveHs(refined)

    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    protein_text = protein_pdb.read_text()
    protein_lines = [ln for ln in protein_text.splitlines() if ln.strip() != "END"]
    last_serial = 0
    for line in protein_lines:
        if line.startswith(("ATOM", "HETATM")):
            try:
                last_serial = max(last_serial, int(line[6:11].strip()))
            except (ValueError, IndexError):
                pass

    refined_pdb_block = Chem.MolToPDBBlock(refined)
    serial_remap: dict[int, int] = {}
    next_serial = last_serial + 1
    lig_atom_lines: list[str] = []
    lig_conect_lines: list[str] = []
    for line in refined_pdb_block.splitlines():
        if line.startswith(("ATOM", "HETATM")):
            try:
                old_serial = int(line[6:11].strip())
            except ValueError:
                continue
            new_serial = next_serial
            next_serial += 1
            serial_remap[old_serial] = new_serial
            atom_name = line[12:16]
            if len(line) < 78:
                line = line.ljust(78)
            element = line[76:78]
            x = line[30:38]
            y = line[38:46]
            z = line[46:54]
            occ = line[54:60] if len(line) >= 60 else "  1.00"
            bfac = line[60:66] if len(line) >= 66 else "  0.00"
            new_line = (
                f"HETATM{new_serial:>5} {atom_name:<4} LIG B   1    "
                f"{x}{y}{z}{occ}{bfac}          {element}"
            )
            lig_atom_lines.append(new_line)
        elif line.startswith("CONECT"):
            rest = line[6:]
            old_atoms: list[int] = []
            for i in range(0, len(rest), 5):
                cell = rest[i : i + 5].strip()
                if not cell:
                    continue
                try:
                    old_atoms.append(int(cell))
                except ValueError:
                    pass
            new_atoms = [serial_remap[a] for a in old_atoms if a in serial_remap]
            if len(new_atoms) >= 2:
                conect_line = "CONECT" + "".join(f"{a:>5}" for a in new_atoms)
                lig_conect_lines.append(conect_line)

    if not lig_atom_lines:
        raise RuntimeError("no atoms extracted from refined ligand")

    last_atom_idx = max(
        (i for i, ln in enumerate(protein_lines) if ln.startswith("ATOM")),
        default=-1,
    )
    if last_atom_idx >= 0 and (
        last_atom_idx + 1 >= len(protein_lines)
        or not protein_lines[last_atom_idx + 1].startswith("TER")
    ):
        protein_lines.insert(last_atom_idx + 1, "TER")

    out_pdb.write_text(
        "\n".join(protein_lines)
        + "\n"
        + "\n".join(lig_atom_lines)
        + "\n"
        + "\n".join(lig_conect_lines)
        + ("\n" if lig_conect_lines else "")
        + "END\n"
    )


def _process_one(task: tuple[str, str, int, str]) -> dict[str, Any]:
    sid, smiles, model_idx, out_subdir = task
    base = {"compound": sid, "src_model": model_idx}
    src_pdb = PRED_DIR.joinpath(sid, f"{sid}_model_{model_idx}.pdb")
    if not src_pdb.exists():
        return {**base, "error": "src_pdb_missing"}
    out_dir = (
        REDOCK_DIR.joinpath(sid, out_subdir) if out_subdir else REDOCK_DIR.joinpath(sid)
    )
    protein_pdb = out_dir.joinpath("protein.pdb")
    ligand_sdf = out_dir.joinpath("ligand_input.sdf")
    refined_sdf_gz = out_dir.joinpath("ligand_refined.sdf.gz")
    final_pdb = out_dir.joinpath(f"{sid}_refined.pdb")
    try:
        _split_boltz_pdb(src_pdb, smiles, protein_pdb, ligand_sdf)
        scores = _run_gnina_minimize(protein_pdb, ligand_sdf, refined_sdf_gz)
        _assemble_refined_pdb(protein_pdb, refined_sdf_gz, final_pdb)
        return {**base, **scores, "refined_pdb": str(final_pdb)}
    except Exception as exc:  # noqa: BLE001
        return {**base, "error": f"{type(exc).__name__}: {exc}"}


def _resolve_tasks(
    df: pd.DataFrame,
    selection_csv: Path | None,
    model_idx_override: int | None,
    out_subdir: str,
) -> list[tuple[str, str, int, str]]:
    if model_idx_override is not None:
        return [
            (r.structure, r.smiles, model_idx_override, out_subdir)
            for _, r in df.iterrows()
        ]
    if selection_csv is None:
        return [(r.structure, r.smiles, 0, out_subdir) for _, r in df.iterrows()]
    sel = pd.read_csv(selection_csv)
    sel_map = dict(zip(sel["compound"], sel["selected_model"]))
    return [
        (r.structure, r.smiles, int(sel_map.get(r.structure, 0)), out_subdir)
        for _, r in df.iterrows()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--selection-csv",
        type=Path,
        default=None,
        help="Per-compound model selection CSV "
        "(e.g. docs/track2_model_selection/<tag>.csv from build_submission.py).",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--model-idx",
        type=int,
        default=None,
        help="If set, run gnina --minimize on this Boltz model index (0-4) for "
        "every compound. Overrides --selection-csv.",
    )
    parser.add_argument(
        "--out-subdir",
        type=str,
        default="",
        help="If set, write per-compound outputs to "
        "structures/.../redock_gnina/<id>/<out-subdir>/ instead of the "
        "compound root. Useful when running multiple model indices.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=PROJECT_ROOT.joinpath("docs", "track2_redock_gnina_scores.csv"),
    )
    args = parser.parse_args()

    if not GNINA_BIN.exists():
        sys.exit(f"gnina binary not found at {GNINA_BIN}")

    df = pd.read_parquet(args.data)
    if args.limit:
        df = df.head(args.limit)

    tasks = _resolve_tasks(df, args.selection_csv, args.model_idx, args.out_subdir)
    print(f"gnina --minimize redock: {len(tasks)} compounds, workers={args.workers}")

    REDOCK_DIR.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    with mp.Pool(args.workers) as pool:
        for n, r in enumerate(pool.imap_unordered(_process_one, tasks, chunksize=1), 1):
            results.append(r)
            if n % 10 == 0 or n == len(tasks):
                err_count = sum(1 for x in results if "error" in x)
                print(f"  {n}/{len(tasks)}  errors={err_count}")

    out_df = pd.DataFrame(results).sort_values("compound").reset_index(drop=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print(f"\nWrote {args.out_csv}")
    err = out_df[out_df.get("error").notna()] if "error" in out_df.columns else None
    if err is not None and not err.empty:
        print(f"Errors: {len(err)}")
        for _, r in err.iterrows():
            print(f"  {r['compound']}: {r['error']}")
    ok = out_df[out_df.get("error", pd.Series([None] * len(out_df))).isna()]
    if not ok.empty and "rmsd_to_input" in ok.columns:
        print(f"\nSucceeded: {len(ok)}")
        print(
            f"  RMSD vs Boltz input (Å): mean={ok['rmsd_to_input'].mean():.2f} "
            f"median={ok['rmsd_to_input'].median():.2f} "
            f"max={ok['rmsd_to_input'].max():.2f}"
        )
        print(
            f"  CNNscore: mean={ok['cnn_score'].mean():.3f} "
            f"median={ok['cnn_score'].median():.3f}"
        )
        print(
            f"  Vina affinity (kcal/mol): mean={ok['vina_affinity'].mean():.2f} "
            f"median={ok['vina_affinity'].median():.2f}"
        )


if __name__ == "__main__":
    main()

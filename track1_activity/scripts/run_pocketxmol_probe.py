"""Run PocketXMol docking/confidence probes on PXR Boltz structures.

This script keeps PocketXMol itself outside this repository. It generates one
PocketXMol sampling config per compound, runs the upstream `sample_use.py`, and
collects the confidence scores written to `gen_info.csv`.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from rdkit import Chem


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from track1_activity.src.data import get_engine  # noqa: E402

DEFAULT_POCKETXMOL_REPO = Path("/home/nagaet/ghq/github.com/pengxingang/PocketXMol")
DEFAULT_VENV_PYTHON = Path("/home/nagaet/.cache/codex/pocketxmol/.venv/bin/python")


@dataclass(frozen=True)
class CompoundJob:
    compound_id: int
    split: str
    smiles: str
    pec50: float | None

    @property
    def cid(self) -> str:
        return f"{self.compound_id:05d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default=time.strftime("pxm_%Y%m%d_%H%M%S"))
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "structures/pocketxmol_probe"
    )
    parser.add_argument("--pocketxmol-repo", type=Path, default=DEFAULT_POCKETXMOL_REPO)
    parser.add_argument("--venv-python", type=Path, default=DEFAULT_VENV_PYTHON)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--num-mols", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--radius", type=float, default=10.0)
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--compound-ids", nargs="*", type=int)
    parser.add_argument(
        "--all-compounds",
        action="store_true",
        help="Run every compound with available Boltz protein/ligand structures.",
    )
    parser.add_argument("--save-output-tensors", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def select_jobs(
    limit: int, compound_ids: list[int] | None, all_compounds: bool
) -> list[CompoundJob]:
    engine = get_engine()
    if all_compounds:
        query = """
        SELECT c.id AS compound_id,
               CASE WHEN t.compound_id IS NULL THEN 'test' ELSE 'train' END AS split,
               c.std_smiles AS smiles,
               t.pec50
        FROM compounds c
        LEFT JOIN train_activity t ON t.compound_id = c.id
        WHERE EXISTS (SELECT 1 FROM train_activity tx WHERE tx.compound_id = c.id)
           OR EXISTS (SELECT 1 FROM test_activity tx WHERE tx.compound_id = c.id)
        ORDER BY c.id
        """
        rows = pd.read_sql(query, engine)
    elif compound_ids:
        ids = ",".join(str(i) for i in compound_ids)
        query = f"""
        SELECT c.id AS compound_id,
               CASE WHEN t.compound_id IS NULL THEN 'test' ELSE 'train' END AS split,
               c.std_smiles AS smiles,
               t.pec50
        FROM compounds c
        LEFT JOIN train_activity t ON t.compound_id = c.id
        WHERE c.id IN ({ids})
        ORDER BY c.id
        """
        rows = pd.read_sql(query, engine)
    else:
        n_each = max(1, limit // 4)
        query = f"""
        (
          SELECT t.compound_id, 'train_low' AS split, c.std_smiles AS smiles, t.pec50
          FROM train_activity t
          JOIN compounds c ON c.id = t.compound_id
          ORDER BY t.pec50 ASC, t.compound_id
          LIMIT {n_each}
        )
        UNION ALL
        (
          SELECT t.compound_id, 'train_high' AS split, c.std_smiles AS smiles, t.pec50
          FROM train_activity t
          JOIN compounds c ON c.id = t.compound_id
          ORDER BY t.pec50 DESC, t.compound_id
          LIMIT {n_each}
        )
        UNION ALL
        (
          SELECT t.compound_id, 'train_random' AS split, c.std_smiles AS smiles, t.pec50
          FROM train_activity t
          JOIN compounds c ON c.id = t.compound_id
          ORDER BY md5(t.compound_id::text)
          LIMIT {n_each}
        )
        UNION ALL
        (
          SELECT ta.compound_id, 'test_random' AS split, c.std_smiles AS smiles,
                 NULL::double precision AS pec50
          FROM test_activity ta
          JOIN compounds c ON c.id = ta.compound_id
          ORDER BY md5(ta.compound_id::text)
          LIMIT {n_each}
        )
        """
        rows = pd.read_sql(query, engine).drop_duplicates("compound_id").head(limit)

    jobs = [
        CompoundJob(
            int(r.compound_id),
            str(r.split),
            str(r.smiles),
            None if pd.isna(r.pec50) else float(r.pec50),
        )
        for r in rows.itertuples(index=False)
    ]
    return [
        job
        for job in jobs
        if structure_paths(job.cid)[0].exists() and structure_paths(job.cid)[1].exists()
    ]


def structure_paths(cid: str) -> tuple[Path, Path]:
    protein = ROOT / f"structures/boltz2/outputs/posebusters_tmp_proteins/{cid}.pdb"
    ligand = ROOT / f"structures/boltz2/ligands/{cid}.sdf"
    return protein, ligand


def ligand_center(ligand: Path) -> tuple[float, float, float]:
    mol = Chem.SDMolSupplier(str(ligand), sanitize=False, removeHs=False)[0]
    if mol is None or not mol.GetNumConformers():
        raise ValueError(f"Cannot read 3D ligand: {ligand}")
    conf = mol.GetConformer()
    coords = [conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())]
    return tuple(
        sum(getattr(p, axis) for p in coords) / len(coords) for axis in ("x", "y", "z")
    )


def prepare_ligand(job: CompoundJob, run_dir: Path) -> Path:
    """Write a sanitizable SDF with DB bond orders and Boltz coordinates."""
    _, ligand = structure_paths(job.cid)
    out_dir = run_dir / "prepared_ligands"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{job.cid}.sdf"
    if out_path.exists():
        return out_path

    coord_mol = Chem.SDMolSupplier(str(ligand), sanitize=False, removeHs=False)[0]
    smiles_mol = Chem.MolFromSmiles(job.smiles)
    if coord_mol is None or smiles_mol is None:
        return ligand
    smiles_mol = Chem.RemoveHs(smiles_mol)
    if coord_mol.GetNumAtoms() != smiles_mol.GetNumAtoms():
        return ligand

    coord_conf = coord_mol.GetConformer()
    conf = Chem.Conformer(smiles_mol.GetNumAtoms())
    for i in range(smiles_mol.GetNumAtoms()):
        conf.SetAtomPosition(i, coord_conf.GetAtomPosition(i))
    smiles_mol.RemoveAllConformers()
    smiles_mol.AddConformer(conf, assignId=True)
    Chem.MolToMolFile(smiles_mol, str(out_path), kekulize=True)
    return out_path


def write_config(
    path: Path,
    job: CompoundJob,
    ligand: Path,
    num_steps: int,
    num_mols: int,
    batch_size: int,
    radius: float,
    save_output_tensors: bool,
) -> None:
    protein, _ = structure_paths(job.cid)
    center = ligand_center(ligand)
    center_text = ", ".join(f"{v:.4f}" for v in center)
    save_output_text = ""
    if save_output_tensors:
        save_output_text = """  save_output:
    - confidence_pos_traj
    - confidence_node
    - confidence_pos
    - confidence_halfedge
"""
    path.write_text(
        f"""sample:
  seed: 2024
  batch_size: {batch_size}
  num_mols: {num_mols}
  num_repeats: 1
  save_traj_prob: 0.0
{save_output_text}

data:
  protein_path: {protein}
  input_ligand: {ligand}
  is_pep: False
  pocket_args:
    ref_ligand_path: {ligand}
    radius: {radius:g}
  pocmol_args:
    data_id: pxr_{job.cid}
    pdbid: PXR

transforms:
  featurizer_pocket:
    center: [{center_text}]

task:
  name: dock
  transform:
    name: dock
    settings:
      free: 1
      flexible: 0

noise:
  name: dock
  num_steps: {num_steps}
  prior: from_train
  level:
    name: advance
    min: 0.
    max: 1.
    step2level:
      scale_start: 0.99999
      scale_end: 0.00001
      width: 3
""",
        encoding="utf-8",
    )


def newest_gen_info(outdir: Path) -> Path | None:
    matches = sorted(outdir.glob("*/gen_info.csv"), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def run_job(
    args: argparse.Namespace, job: CompoundJob, run_dir: Path
) -> dict[str, object]:
    config_dir = run_dir / "configs"
    raw_dir = run_dir / "raw" / job.cid
    config_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{job.cid}.yml"
    ligand = prepare_ligand(job, run_dir)
    write_config(
        config_path,
        job,
        ligand,
        args.num_steps,
        args.num_mols,
        args.batch_size,
        args.radius,
        args.save_output_tensors,
    )

    cmd = [
        str(args.venv_python),
        "scripts/sample_use.py",
        "--config_task",
        str(config_path),
        "--config_model",
        "configs/sample/pxm.yml",
        "--outdir",
        str(raw_dir),
        "--device",
        args.device,
        "--batch_size",
        str(args.batch_size),
        "--num_workers",
        str(args.num_workers),
    ]
    started = time.time()
    if args.dry_run:
        return {
            "compound_id": job.compound_id,
            "cid": job.cid,
            "split": job.split,
            "pec50": job.pec50,
            "status": "dry_run",
            "cmd": " ".join(cmd),
        }

    proc = subprocess.run(
        cmd,
        cwd=args.pocketxmol_repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    elapsed = time.time() - started
    log_path = run_dir / "logs" / f"{job.cid}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(proc.stdout, encoding="utf-8")

    row: dict[str, object] = {
        "compound_id": job.compound_id,
        "cid": job.cid,
        "split": job.split,
        "pec50": job.pec50,
        "status": "ok" if proc.returncode == 0 else "failed",
        "returncode": proc.returncode,
        "elapsed_sec": round(elapsed, 3),
    }
    gen_info = newest_gen_info(raw_dir)
    if proc.returncode == 0 and gen_info is not None:
        df = pd.read_csv(gen_info)
        for col in ["tag", "smiles", "cfd_traj", "cfd_pos", "cfd_node", "cfd_edge"]:
            row[col] = df[col].iloc[0] if col in df.columns and len(df) else None
        row["gen_info"] = str(gen_info)
        if args.save_output_tensors:
            row.update(read_output_tensor_stats(gen_info.parent))
    return row


def read_output_tensor_stats(log_dir: Path) -> dict[str, float]:
    pt_files = sorted((log_dir / "SDF").glob("*.pt"))
    if not pt_files:
        return {}
    import torch

    output = torch.load(pt_files[0], map_location="cpu", weights_only=False)
    traj = output.get("confidence_pos_traj")
    if traj is None:
        return {}
    cfd_steps = traj.float().mean(dim=0).flatten()
    n = len(cfd_steps)
    first = cfd_steps[: max(1, n // 4)]
    last = cfd_steps[-max(1, n // 4) :]
    return {
        "traj_mean": float(cfd_steps.mean()),
        "traj_std": float(cfd_steps.std(unbiased=False)),
        "traj_min": float(cfd_steps.min()),
        "traj_max": float(cfd_steps.max()),
        "traj_first": float(cfd_steps[0]),
        "traj_last": float(cfd_steps[-1]),
        "traj_first25_mean": float(first.mean()),
        "traj_last25_mean": float(last.mean()),
        "traj_delta": float(cfd_steps[-1] - cfd_steps[0]),
    }


def main() -> None:
    args = parse_args()
    run_dir = args.output_root / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    jobs = select_jobs(args.limit, args.compound_ids, args.all_compounds)
    summary_path = run_dir / "summary.csv"
    fields = [
        "compound_id",
        "cid",
        "split",
        "pec50",
        "status",
        "returncode",
        "elapsed_sec",
        "tag",
        "smiles",
        "cfd_traj",
        "cfd_pos",
        "cfd_node",
        "cfd_edge",
        "gen_info",
        "cmd",
        "traj_mean",
        "traj_std",
        "traj_min",
        "traj_max",
        "traj_first",
        "traj_last",
        "traj_first25_mean",
        "traj_last25_mean",
        "traj_delta",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for i, job in enumerate(jobs, 1):
            print(f"[{i}/{len(jobs)}] {job.cid} {job.split}", flush=True)
            try:
                row = run_job(args, job, run_dir)
            except Exception as exc:
                row = {
                    "compound_id": job.compound_id,
                    "cid": job.cid,
                    "split": job.split,
                    "pec50": job.pec50,
                    "status": "wrapper_failed",
                    "returncode": -1,
                    "tag": type(exc).__name__,
                    "smiles": str(exc),
                }
            writer.writerow(row)
            f.flush()
    print(summary_path)


if __name__ == "__main__":
    main()

"""Prepare and run PocketXMol hidden-state embedding extraction."""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

from run_pocketxmol_probe import (
    DEFAULT_POCKETXMOL_REPO,
    DEFAULT_VENV_PYTHON,
    ROOT,
    prepare_ligand,
    select_jobs,
    write_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "structures/pocketxmol_probe"
    )
    parser.add_argument("--pocketxmol-repo", type=Path, default=DEFAULT_POCKETXMOL_REPO)
    parser.add_argument("--venv-python", type=Path, default=DEFAULT_VENV_PYTHON)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--compound-ids", nargs="*", type=int)
    parser.add_argument("--all-compounds", action="store_true")
    parser.add_argument("--radius", type=float, default=10.0)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--timesteps", nargs="+", type=float, default=[1.0, 0.5, 0.05])
    parser.add_argument("--worker-limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def write_manifest(args: argparse.Namespace, run_dir: Path) -> Path:
    config_dir = run_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    jobs = select_jobs(args.limit, args.compound_ids, args.all_compounds)
    manifest_path = run_dir / "embedding_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["compound_id", "cid", "split", "pec50", "config_path"],
        )
        writer.writeheader()
        for job in jobs:
            ligand = prepare_ligand(job, run_dir)
            config_path = config_dir / f"{job.cid}.yml"
            write_config(
                config_path,
                job,
                ligand,
                args.num_steps,
                1,
                1,
                args.radius,
                False,
            )
            writer.writerow(
                {
                    "compound_id": job.compound_id,
                    "cid": job.cid,
                    "split": job.split,
                    "pec50": "" if job.pec50 is None else job.pec50,
                    "config_path": config_path,
                }
            )
    return manifest_path


def main() -> None:
    args = parse_args()
    run_dir = args.output_root / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = write_manifest(args, run_dir)
    raw_dir = run_dir / "raw_hidden"
    worker = ROOT / "track1_activity/scripts/pocketxmol_embedding_worker.py"
    cmd = [
        str(args.venv_python),
        str(worker),
        "--manifest",
        str(manifest_path),
        "--config-model",
        "configs/sample/pxm.yml",
        "--out-dir",
        str(raw_dir),
        "--device",
        args.device,
        "--timesteps",
        *[str(t) for t in args.timesteps],
    ]
    if args.worker_limit:
        cmd.extend(["--limit", str(args.worker_limit)])
    print(" ".join(cmd), flush=True)
    if args.dry_run:
        return
    subprocess.run(cmd, cwd=args.pocketxmol_repo, check=True)
    print(raw_dir)


if __name__ == "__main__":
    main()

#!/usr/bin/env -S pixi run python
"""Package the Boltz-2 best models into a Track 2 submission zip.

For each compound directory under
``structures/boltz2_track2/outputs/holo/boltz_results_holo/predictions/<id>/``,
copy ``<id>_model_0.pdb`` (Boltz orders models by confidence_score, so model_0
is the highest-confidence pose) to the staging directory as ``<id>.pdb`` and
package all 184 PDBs into a single zip.

Usage:
    pixi run python track2_structure/scripts/build_submission.py
    pixi run python track2_structure/scripts/build_submission.py \\
        --model 0 --tag boltz2_baseline_v1
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT.joinpath("track2_structure", "src")))

from track2.constants import REPO_ROOT as PROJECT_ROOT  # noqa: E402

PREDICTIONS_DIR = PROJECT_ROOT.joinpath(
    "structures",
    "boltz2_track2",
    "outputs",
    "holo",
    "boltz_results_holo",
    "predictions",
)
STAGING_DIR = PROJECT_ROOT.joinpath("track2_structure", "submissions", "_staging")
SUBMISSIONS_DIR = PROJECT_ROOT.joinpath("track2_structure", "submissions")
DEFAULT_DATA = PROJECT_ROOT.joinpath("data", "structure_test.parquet")


def _stage_pdbs(model_idx: int, expected_ids: set[str]) -> tuple[list[Path], list[str]]:
    """Copy <id>_model_<i>.pdb to STAGING_DIR/<id>.pdb. Returns (copied, missing)."""
    if STAGING_DIR.exists():
        shutil.rmtree(STAGING_DIR)
    STAGING_DIR.mkdir(parents=True)

    copied: list[Path] = []
    missing: list[str] = []
    for sid in sorted(expected_ids):
        src = PREDICTIONS_DIR.joinpath(sid, f"{sid}_model_{model_idx}.pdb")
        if not src.exists():
            missing.append(sid)
            continue
        dst = STAGING_DIR.joinpath(f"{sid}.pdb")
        shutil.copy2(src, dst)
        copied.append(dst)
    return copied, missing


def _zip_submission(pdbs: list[Path], zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for pdb in pdbs:
            zf.write(pdb, arcname=pdb.name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=int,
        default=0,
        help="Boltz-2 model index to submit (default: 0 = best confidence).",
    )
    parser.add_argument(
        "--tag",
        default=f"boltz2_model{0}_{date.today().isoformat()}",
        help="Tag included in the zip filename.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help="SMILES parquet (used to determine expected compound ids).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output zip path (default: track2_structure/submissions/<tag>.zip).",
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.data)
    expected_ids = set(df["structure"])
    if len(expected_ids) != 184:
        print(
            f"[warn] expected 184 ids in {args.data}, found {len(expected_ids)}",
            file=sys.stderr,
        )

    print(
        f"Staging model_{args.model} PDBs from {PREDICTIONS_DIR.relative_to(PROJECT_ROOT)}"
    )
    copied, missing = _stage_pdbs(args.model, expected_ids)
    if missing:
        print(f"[error] {len(missing)} PDBs missing:", file=sys.stderr)
        for sid in missing[:10]:
            print(f"  - {sid}", file=sys.stderr)
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more", file=sys.stderr)
        sys.exit(1)
    print(f"  copied {len(copied)} PDBs to {STAGING_DIR.relative_to(PROJECT_ROOT)}/")

    out = args.out or SUBMISSIONS_DIR.joinpath(f"{args.tag}.zip")
    _zip_submission(copied, out)
    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"  zipped → {out.relative_to(PROJECT_ROOT)} ({size_mb:.1f} MB)")
    print()
    print(
        "Next: run the official validation script before submitting:\n"
        f'  pixi run python -c "\\\n'
        f"import sys; sys.path.insert(0, "
        f"'/home/nagaet/ghq/github.com/OpenADMET/PXR-Challenge-Tutorial'); \\\n"
        f"from validation.structure_validation import validate_structure_submission; \\\n"
        f"import pandas as pd; \\\n"
        f"df = pd.read_parquet('{args.data}'); \\\n"
        f"ok, errs = validate_structure_submission(\\\n"
        f"    '{out}', expected_ids=set(df['structure']), \\\n"
        f"    expected_ligand_smiles=dict(zip(df['structure'], df['smiles']))); \\\n"
        f"print('OK' if ok else 'FAIL'); print('\\n'.join(errs[:20]))\""
    )


if __name__ == "__main__":
    main()

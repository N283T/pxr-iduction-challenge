#!/usr/bin/env -S pixi run python
"""Package Boltz-2 outputs into a Track 2 submission zip.

For each compound directory under
``structures/boltz2_track2/outputs/holo/boltz_results_holo/predictions/<id>/``,
select one of the five generated models per the chosen ``--strategy`` and copy
that PDB to the staging directory as ``<id>.pdb``. Then zip all 184 PDBs.

Strategies for per-compound model selection:

* ``confidence`` (default) — always model_0. Boltz orders its output models
  by the internal ``confidence_score``, so model_0 is the highest-confidence
  pose for every compound.
* ``medoid`` — after Cα-superposing each model on model_0, pick the model
  whose ligand centroid has the smallest mean distance to the other four.
  This is the "densest cluster" pose, which Deep Research literature
  recommends as more robust than single-best when the five poses disagree.
* ``iptm`` — per-compound, pick the model with the highest interface
  predicted-TM (``iptm`` from confidence JSON).
* ``low_ipde`` — per-compound, pick the model with the lowest interface
  predicted distance error (``complex_ipde``).

Usage:
    pixi run python track2_structure/scripts/build_submission.py \\
        --strategy medoid --tag boltz2_medoid_v2
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
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
SELECTION_LOG_DIR = PROJECT_ROOT.joinpath("docs", "track2_model_selection")


# ---------------------------------------------------------------------------
# Selection strategies
# ---------------------------------------------------------------------------


def _parse_atoms(pdb_text: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (Cα xyz, ligand xyz) from a Boltz-2 PDB string."""
    ca, lig = [], []
    for line in pdb_text.splitlines():
        if not line.startswith(("ATOM", "HETATM")) or len(line) < 54:
            continue
        try:
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
        except ValueError:
            continue
        atom = line[12:16].strip()
        resname = line[17:20].strip()
        if line.startswith("ATOM") and atom == "CA":
            ca.append((x, y, z))
        elif resname.startswith("LIG"):
            lig.append((x, y, z))
    return np.array(ca), np.array(lig)


def _kabsch(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Optimal rotation aligning centered P → centered Q."""
    H = P.T @ Q
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1, 1, d]) @ U.T


def _select_medoid(cid: str) -> int:
    """Return the medoid model index after Cα superposition.

    For each of the five models, Cα-superpose to model_0 then compute
    ligand centroid in the aligned frame. The medoid is the model whose
    centroid has the smallest mean pairwise distance to the others.
    """
    cdir = PREDICTIONS_DIR.joinpath(cid)
    centroids = []
    ca_ref = None
    for mi in range(5):
        pdb = cdir.joinpath(f"{cid}_model_{mi}.pdb")
        if not pdb.exists():
            return 0
        ca, lig = _parse_atoms(pdb.read_text())
        if mi == 0:
            ca_ref = ca
            centroids.append(lig.mean(0))
        else:
            if ca.shape != ca_ref.shape:
                return 0
            cs = ca.mean(0)
            cr = ca_ref.mean(0)
            R = _kabsch(ca - cs, ca_ref - cr)
            lig_t = (lig - cs) @ R.T + cr
            centroids.append(lig_t.mean(0))
    centroids = np.array(centroids)
    pairwise = np.linalg.norm(centroids[:, None] - centroids[None, :], axis=2)
    return int(pairwise.mean(axis=1).argmin())


def _select_by_confidence_field(cid: str, field: str, *, maximise: bool) -> int:
    """Pick the model with argmax / argmin of a confidence JSON field."""
    cdir = PREDICTIONS_DIR.joinpath(cid)
    values = []
    for mi in range(5):
        cj = cdir.joinpath(f"confidence_{cid}_model_{mi}.json")
        if not cj.exists():
            return 0
        d = json.loads(cj.read_text())
        v = d.get(field)
        if v is None:
            return 0
        values.append(v)
    arr = np.asarray(values)
    return int(arr.argmax() if maximise else arr.argmin())


def _resolve_model_indices(strategy: str, compound_ids: list[str]) -> dict[str, int]:
    """Return per-compound model index for the given strategy."""
    if strategy == "confidence":
        return {sid: 0 for sid in compound_ids}
    if strategy == "medoid":
        return {sid: _select_medoid(sid) for sid in compound_ids}
    if strategy == "iptm":
        return {
            sid: _select_by_confidence_field(sid, "iptm", maximise=True)
            for sid in compound_ids
        }
    if strategy == "low_ipde":
        return {
            sid: _select_by_confidence_field(sid, "complex_ipde", maximise=False)
            for sid in compound_ids
        }
    raise ValueError(f"unknown strategy: {strategy}")


# ---------------------------------------------------------------------------
# Staging + zip
# ---------------------------------------------------------------------------


def _stage_pdbs(
    model_per_compound: dict[str, int], expected_ids: set[str]
) -> tuple[list[Path], list[str]]:
    """Copy <id>_model_<i>.pdb to STAGING_DIR/<id>.pdb. Returns (copied, missing)."""
    if STAGING_DIR.exists():
        shutil.rmtree(STAGING_DIR)
    STAGING_DIR.mkdir(parents=True)

    copied: list[Path] = []
    missing: list[str] = []
    for sid in sorted(expected_ids):
        mi = model_per_compound.get(sid, 0)
        src = PREDICTIONS_DIR.joinpath(sid, f"{sid}_model_{mi}.pdb")
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


def _save_selection_log(
    strategy: str, model_per_compound: dict[str, int], tag: str
) -> Path:
    SELECTION_LOG_DIR.mkdir(parents=True, exist_ok=True)
    out = SELECTION_LOG_DIR.joinpath(f"{tag}.csv")
    df = pd.DataFrame(
        {
            "compound": list(model_per_compound.keys()),
            "selected_model": list(model_per_compound.values()),
            "strategy": strategy,
        }
    ).sort_values("compound")
    df.to_csv(out, index=False)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strategy",
        choices=["confidence", "medoid", "iptm", "low_ipde"],
        default="confidence",
        help="Per-compound model selection strategy (default: confidence).",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Tag included in the zip filename (default: <strategy>_<date>).",
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
    compound_ids = sorted(expected_ids)
    if len(expected_ids) != 184:
        print(
            f"[warn] expected 184 ids in {args.data}, found {len(expected_ids)}",
            file=sys.stderr,
        )

    print(f"Selecting models per compound with strategy={args.strategy}")
    model_per_compound = _resolve_model_indices(args.strategy, compound_ids)
    counts = pd.Series(list(model_per_compound.values())).value_counts().sort_index()
    print(f"  selected-model distribution: {dict(counts)}")
    n_nondefault = sum(1 for v in model_per_compound.values() if v != 0)
    print(f"  compounds with model != 0: {n_nondefault} / {len(compound_ids)}")

    tag = args.tag or f"boltz2_{args.strategy}_{date.today().isoformat()}"
    log = _save_selection_log(args.strategy, model_per_compound, tag)
    print(f"  selection log: {log.relative_to(PROJECT_ROOT)}")

    print(f"Staging selected PDBs from {PREDICTIONS_DIR.relative_to(PROJECT_ROOT)}")
    copied, missing = _stage_pdbs(model_per_compound, expected_ids)
    if missing:
        print(f"[error] {len(missing)} PDBs missing:", file=sys.stderr)
        for sid in missing[:10]:
            print(f"  - {sid}", file=sys.stderr)
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more", file=sys.stderr)
        sys.exit(1)
    print(f"  copied {len(copied)} PDBs to {STAGING_DIR.relative_to(PROJECT_ROOT)}/")

    out = args.out or SUBMISSIONS_DIR.joinpath(f"{tag}.zip")
    _zip_submission(copied, out)
    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"  zipped → {out.relative_to(PROJECT_ROOT)} ({size_mb:.1f} MB)")
    print()
    print(
        "Next: validate before submitting:\n"
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

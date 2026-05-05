#!/usr/bin/env -S pixi run python
"""Build a Z-anchored submission zip per FTMap consensus hotspot.

Strategy: for each query, look up the candidate pose source with the
SMALLEST distance to FTMap consensus hotspot Z (the canonical PXR
pocket — chain Z in our FTMap output, where all 16 probe types
converge). The choice comes from
``docs/track2/track2_ftmap_hotspot_scores.csv``.

Motivation: v3 (gnina --minimize) regressed by -0.0083 LDDT-PLI on
the LB after moving the ligand only ~0.03 Å further from hotspot Z
(5.15→5.20 template-RMSD on average, see PR #137). This calibration
strongly suggests that proximity to FTMap Z is the correct binding
mode for this ligand class. By pre-selecting per query the candidate
that's closest to Z, we expect to recover or improve over v1 (rank 7,
LDDT-PLI 0.4655).

Usage:
    pixi run python track2_structure/scripts/build_zanchored_submission.py \\
        --include boltz                # v2b: Boltz models only
    pixi run python track2_structure/scripts/build_zanchored_submission.py \\
        --include boltz,gnina          # v6: Boltz + gnina_refined
    pixi run python track2_structure/scripts/build_zanchored_submission.py \\
        --tag track2_boltz2_zanchored_v2b_2026-04-26 --include boltz
"""

from __future__ import annotations

import argparse
import shutil
import sys
import warnings
import zipfile
from datetime import date
from pathlib import Path

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
GNINA_DIR = PROJECT_ROOT.joinpath("structures", "boltz2_track2", "redock_gnina")
TEMPLATE_DIR = PROJECT_ROOT.joinpath("structures", "boltz2_track2", "redock_template")
POSIT_DIR = PROJECT_ROOT.joinpath("structures", "boltz2_track2", "redock_posit")
HOTSPOT_CSV = PROJECT_ROOT.joinpath("docs", "track2", "track2_ftmap_hotspot_scores.csv")
DEFAULT_DATA = PROJECT_ROOT.joinpath("data", "structure_test.parquet")
SUBMISSIONS_DIR = PROJECT_ROOT.joinpath("track2_structure", "submissions")
STAGING_DIR = SUBMISSIONS_DIR.joinpath("_staging")
SELECTION_LOG_DIR = PROJECT_ROOT.joinpath("docs", "track2", "model_selection")


SOURCE_PDB_RESOLVERS = {
    **{
        f"boltz_model_{i}": (
            lambda sid, i=i: PRED_DIR.joinpath(sid, f"{sid}_model_{i}.pdb")
        )
        for i in range(5)
    },
    "gnina_refined": (lambda sid: GNINA_DIR.joinpath(sid, f"{sid}_refined.pdb")),
    "template_transferred": (
        lambda sid: TEMPLATE_DIR.joinpath(sid, f"{sid}_template.pdb")
    ),
    "posit": (lambda sid: POSIT_DIR.joinpath(sid, f"{sid}_posit.pdb")),
}


def _resolve_pdb(sid: str, source: str) -> Path:
    return SOURCE_PDB_RESOLVERS[source](sid)


def _load_posebusters_pass(pb_csv: Path) -> set[tuple[str, str]]:
    """Return {(compound, 'boltz_model_<i>')} pairs that pass all PoseBusters
    validity checks. Only Boltz model rows are considered (model 0..4)."""
    df = pd.read_csv(pb_csv)
    bool_cols = [
        c
        for c in df.columns
        if c not in {"compound", "model", "error"}
        and df[c].dropna().isin([True, False]).all()
        and df[c].notna().any()
    ]
    df = df.copy()
    df["all_passed"] = df[bool_cols].all(axis=1)
    passing = df[df["all_passed"]]
    return {
        (row["compound"], f"boltz_model_{int(row['model'])}")
        for _, row in passing.iterrows()
    }


def _select_z_anchored(
    hotspot_df: pd.DataFrame,
    allowed_sources: set[str],
    *,
    posebusters_pass: set[tuple[str, str]] | None = None,
    min_z_improvement: float = 0.0,
    baseline_source: str = "boltz_model_0",
) -> dict[str, str]:
    """Per compound, pick the candidate source with the smallest dist_to_Z,
    subject to two optional safety gates.

    Gate A (validity): if ``posebusters_pass`` is provided, restrict every
    boltz_model_<i> candidate to (compound, source) pairs found in the set.
    Other source families are not gated (they have their own validity steps).

    Gate B (improvement margin): drop any non-baseline candidate whose
    dist_to_Z improvement vs ``baseline_source`` is below
    ``min_z_improvement``. The compound's baseline candidate is always
    retained (so we fall back to it when no swap meets the bar).
    """
    sub = hotspot_df[
        hotspot_df["source"].isin(allowed_sources) & hotspot_df["dist_to_Z"].notna()
    ].copy()
    if sub.empty:
        return {}

    if posebusters_pass is not None:
        boltz_mask = sub["source"].str.startswith("boltz_model_")
        keep_boltz = sub[boltz_mask].apply(
            lambda r: (r["compound"], r["source"]) in posebusters_pass, axis=1
        )
        sub = pd.concat([sub[~boltz_mask], sub[boltz_mask][keep_boltz]])

    if min_z_improvement > 0.0:
        baseline = (
            sub[sub["source"] == baseline_source]
            .set_index("compound")["dist_to_Z"]
            .to_dict()
        )
        sub = sub.assign(
            _baseline_dist=sub["compound"].map(baseline),
        )
        sub["_improvement"] = sub["_baseline_dist"] - sub["dist_to_Z"]
        keep = (sub["source"] == baseline_source) | (
            sub["_improvement"] >= min_z_improvement
        )
        sub = sub[keep].drop(columns=["_baseline_dist", "_improvement"])

    if sub.empty:
        return {}
    sub_sorted = sub.sort_values("dist_to_Z").groupby("compound").first().reset_index()
    return dict(zip(sub_sorted["compound"], sub_sorted["source"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include",
        default="boltz",
        help=(
            "Comma-separated source families to consider: "
            "boltz (= boltz_model_0..4), gnina, template, posit. "
            "Default: boltz (= 'v2b'). For 'v6' use 'boltz,gnina'."
        ),
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--hotspot-csv", type=Path, default=HOTSPOT_CSV)
    parser.add_argument("--tag", default=None)
    parser.add_argument(
        "--fallback",
        default="boltz_model_0",
        choices=list(SOURCE_PDB_RESOLVERS.keys()),
        help="Source to use for queries without a Z-distance score.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output zip path (default: track2_structure/submissions/<tag>.zip).",
    )
    parser.add_argument(
        "--require-posebusters-passed",
        type=Path,
        default=None,
        help=(
            "Path to PoseBusters CSV (e.g. docs/track2/track2_posebusters_all.csv). "
            "If set, restricts each boltz_model_<i> candidate to (compound, "
            "model) pairs that pass all validity checks. Other source "
            "families are not gated."
        ),
    )
    parser.add_argument(
        "--min-z-improvement",
        type=float,
        default=0.0,
        help=(
            "Minimum required dist_to_Z improvement (Å) for any non-baseline "
            "swap. Below this threshold the compound stays on --fallback. "
            "Use 0.0 (default) to disable. 0.5 is a safe value to keep only "
            "high-impact Z-anchored swaps."
        ),
    )
    args = parser.parse_args()

    requested = {f.strip() for f in args.include.split(",") if f.strip()}
    expanded: set[str] = set()
    for fam in requested:
        if fam == "boltz":
            expanded.update(f"boltz_model_{i}" for i in range(5))
        elif fam == "gnina":
            expanded.add("gnina_refined")
        elif fam == "template":
            expanded.add("template_transferred")
        elif fam == "posit":
            expanded.add("posit")
        else:
            sys.exit(f"unknown family: {fam}")
    print(f"Allowed sources: {sorted(expanded)}")

    hotspot_df = pd.read_csv(args.hotspot_csv)
    pb_pass: set[tuple[str, str]] | None = None
    if args.require_posebusters_passed is not None:
        pb_pass = _load_posebusters_pass(args.require_posebusters_passed)
        print(f"PoseBusters gate: {len(pb_pass)} valid (compound, boltz_model) pairs")
    selection = _select_z_anchored(
        hotspot_df,
        expanded,
        posebusters_pass=pb_pass,
        min_z_improvement=args.min_z_improvement,
        baseline_source=args.fallback,
    )

    df = pd.read_parquet(args.data)
    expected_ids = sorted(df["structure"])
    print(f"\nQueries: {len(expected_ids)} | with Z-distance score: {len(selection)}")

    if STAGING_DIR.exists():
        shutil.rmtree(STAGING_DIR)
    STAGING_DIR.mkdir(parents=True)
    selection_rows: list[dict] = []
    for sid in expected_ids:
        chosen = selection.get(sid, args.fallback)
        src = _resolve_pdb(sid, chosen)
        if not src.exists():
            sys.exit(f"missing pdb for {sid}: {src}")
        shutil.copy2(src, STAGING_DIR.joinpath(f"{sid}.pdb"))
        selection_rows.append({"compound": sid, "selected_source": chosen})

    counts = pd.Series([r["selected_source"] for r in selection_rows]).value_counts()
    print("\nSelected-source distribution:")
    for src, c in counts.items():
        print(f"  {src}: {c}")

    tag = (
        args.tag
        or f"boltz2_zanchored_{args.include.replace(',', '+')}_{date.today().isoformat()}"
    )
    out = args.out or SUBMISSIONS_DIR.joinpath(f"{tag}.zip")
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for pdb in sorted(STAGING_DIR.glob("*.pdb")):
            zf.write(pdb, arcname=pdb.name)
    print(
        f"\nZip: {out.relative_to(PROJECT_ROOT)} ({out.stat().st_size / 1024 / 1024:.1f} MB)"
    )

    SELECTION_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = SELECTION_LOG_DIR.joinpath(f"{tag}.csv")
    pd.DataFrame(selection_rows).to_csv(log_path, index=False)
    print(f"Selection log: {log_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

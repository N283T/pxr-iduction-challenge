"""Inventory Boltz full-run and fast embeddings-only trunk coverage.

This is an audit script, not a model-training script. It establishes the
actual data layers available before choosing the next Boltz-family candidate:

* compound_boltz2: full rcycle=3 structure/pose/confidence run.
* compound_boltz2_trunk_fast: 13k pooled trunk table with rcycle flags and
  source embeddings_*.npz paths.
* existing Boltz-family experiment rows.

Output:
  track1_activity/analysis/boltz_trunk_fast_inventory/outputs/report.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
sys.path.insert(
    0, str(REPO_ROOT.joinpath("track1_activity", "boltz2", "src", "boltz2"))
)

from constants import PXR_SEQUENCE  # noqa: E402
from data import get_engine  # noqa: E402

OUT_DIR = REPO_ROOT.joinpath(
    "track1_activity", "analysis", "boltz_trunk_fast_inventory", "outputs"
)
REPORT_PATH = OUT_DIR.joinpath("report.md")

BOLTZ_NAME_TOKENS = (
    "boltz",
    "pooled_boltz",
    "trunk_pretrain",
    "mordred3d",
    "contact",
)


def missing_ids(all_ids: list[int], present_ids: list[int]) -> list[int]:
    """Return sorted IDs present in all_ids but absent from present_ids."""
    return sorted(set(int(x) for x in all_ids) - set(int(x) for x in present_ids))


def format_recycling_counts(counts: dict[int, int]) -> str:
    """Format recycling step counts in deterministic order."""
    return "\n".join(f"- rcycle={k}: {counts[k]}" for k in sorted(counts))


def summarize_npz(npz_path: Path, protein_n_res: int) -> dict[str, Any]:
    """Read one Boltz embeddings NPZ and summarize shapes/token counts."""
    summary: dict[str, Any] = {
        "path": str(npz_path),
        "readable": False,
        "size_mb": float(npz_path.stat().st_size / 1_000_000),
    }
    try:
        data = np.load(npz_path, allow_pickle=False)
        s = data["s"]
        z = data["z"]
    except Exception as exc:  # noqa: BLE001
        summary["error"] = f"{type(exc).__name__}: {exc}"
        return summary

    token_count = int(s.shape[1]) if s.ndim >= 3 else None
    ligand_tokens = token_count - protein_n_res if token_count is not None else None
    summary.update(
        {
            "readable": True,
            "s_shape": list(s.shape),
            "z_shape": list(z.shape),
            "token_count": token_count,
            "ligand_tokens": ligand_tokens,
            "finite": bool(np.isfinite(s).all() and np.isfinite(z).all()),
        }
    )
    return summary


def is_boltz_experiment_name(name: str) -> bool:
    """Return True for Boltz-family experiment names."""
    lname = name.lower()
    return any(token in lname for token in BOLTZ_NAME_TOKENS)


def classify_boltz_experiment(name: str) -> str:
    """Classify Boltz-family experiments by direct trunk/structure usage."""
    lname = name.lower()
    if (
        "pooled_boltz" in lname
        or "boltz_trunk_pretrain" in lname
        or "boltz_raw_plus_pretrain" in lname
    ):
        return "trunk_only"
    if (
        "boltz2_tabular" in lname
        or "mordred3d" in lname
        or "contact" in lname
        or "ifp" in lname
    ):
        return "structure_tabular"
    if "boltz" in lname:
        return "descriptor_mix"
    return "other"


def fetch_one_value(engine, sql: str, column: str) -> int:
    return int(pd.read_sql(sql, engine)[column].iloc[0])


def fetch_full_run_summary(conn) -> dict[str, Any]:
    full_rows = pd.read_sql(
        """
        SELECT compound_id, embeddings_npz_path, preprocessing_failed,
               ligand_atom_count, confidence_score, affinity_pred_value,
               ligand_to_pocket_distance_a
          FROM compound_boltz2
         ORDER BY compound_id
        """,
        conn,
    )
    return {
        "n_rows": int(len(full_rows)),
        "n_embedding_paths": int(full_rows["embeddings_npz_path"].notna().sum()),
        "n_failed": int(full_rows["preprocessing_failed"].sum()),
        "n_pose_confidence": int(full_rows["confidence_score"].notna().sum()),
        "n_affinity": int(full_rows["affinity_pred_value"].notna().sum()),
        "n_ligand_distance": int(
            full_rows["ligand_to_pocket_distance_a"].notna().sum()
        ),
        "ids": full_rows["compound_id"].astype(int).tolist(),
    }


def fetch_trunk_fast(conn) -> pd.DataFrame:
    return pd.read_sql(
        """
        SELECT compound_id, recycling_steps, source_npz_path,
               cardinality(s_prot_mean) AS s_prot_dim,
               cardinality(s_lig_mean) AS s_lig_dim,
               cardinality(z_if_mean) AS z_if_mean_dim,
               cardinality(z_if_max) AS z_if_max_dim
          FROM compound_boltz2_trunk_fast
         ORDER BY compound_id
        """,
        conn,
    )


def fetch_boltz_experiments(conn, limit: int) -> pd.DataFrame:
    df = pd.read_sql(
        """
        SELECT id, name, mae_mean, rae_mean, spearman_mean, created_at
          FROM experiment_summary
         ORDER BY mae_mean ASC NULLS LAST, created_at DESC
        """,
        conn,
    )
    mask = df["name"].map(is_boltz_experiment_name)
    out = df.loc[mask].copy()
    out["category"] = out["name"].map(classify_boltz_experiment)
    return out.head(limit)


def sample_npz_summaries(
    trunk_df: pd.DataFrame,
    per_rcycle: int,
    protein_n_res: int,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for rcycle, group in trunk_df.groupby("recycling_steps", sort=True):
        picked = group.dropna(subset=["source_npz_path"]).head(per_rcycle)
        for row in picked.itertuples(index=False):
            summary = summarize_npz(Path(row.source_npz_path), protein_n_res)
            summary["compound_id"] = int(row.compound_id)
            summary["recycling_steps"] = int(rcycle)
            samples.append(summary)
    return samples


def render_report(
    *,
    n_compounds: int,
    full: dict[str, Any],
    trunk_df: pd.DataFrame,
    missing: list[int],
    npz_samples: list[dict[str, Any]],
    boltz_experiments: pd.DataFrame,
) -> str:
    counts = (
        trunk_df.groupby("recycling_steps")["compound_id"].count().astype(int).to_dict()
    )
    npz_existing = trunk_df["source_npz_path"].map(lambda p: Path(p).exists()).sum()
    dim_counts = trunk_df[
        ["s_prot_dim", "s_lig_dim", "z_if_mean_dim", "z_if_max_dim"]
    ].drop_duplicates()

    lines: list[str] = [
        "# Boltz Trunk Fast Inventory",
        "",
        "## Coverage",
        "",
        f"- compounds table rows: {n_compounds}",
        f"- compound_boltz2 rows: {full['n_rows']}",
        f"- compound_boltz2 embedding paths: {full['n_embedding_paths']}",
        f"- compound_boltz2 preprocessing_failed rows: {full['n_failed']}",
        f"- compound_boltz2 confidence rows: {full['n_pose_confidence']}",
        f"- compound_boltz2 affinity rows: {full['n_affinity']}",
        f"- compound_boltz2 ligand distance rows: {full['n_ligand_distance']}",
        f"- compound_boltz2_trunk_fast rows: {len(trunk_df)}",
        f"- compound_boltz2_trunk_fast source paths existing: {int(npz_existing)}",
        f"- missing from trunk_fast: {missing if missing else 'none'}",
        "",
        "## Recycling Split",
        "",
        format_recycling_counts({int(k): int(v) for k, v in counts.items()}),
        "",
        "## Stored Vector Dimensions",
        "",
        dim_counts.to_markdown(index=False),
        "",
        "## Sample Raw NPZ Shapes",
        "",
    ]
    sample_rows = []
    for sample in npz_samples:
        sample_rows.append(
            {
                "compound_id": sample.get("compound_id"),
                "rcycle": sample.get("recycling_steps"),
                "readable": sample.get("readable"),
                "s_shape": sample.get("s_shape"),
                "z_shape": sample.get("z_shape"),
                "ligand_tokens": sample.get("ligand_tokens"),
                "finite": sample.get("finite"),
                "size_mb": round(float(sample.get("size_mb", 0.0)), 2),
            }
        )
    lines.extend(
        [
            pd.DataFrame(sample_rows).to_markdown(index=False),
            "",
            "## Existing Boltz-Family Experiments",
            "",
        ]
    )
    if boltz_experiments.empty:
        lines.append("No Boltz-family experiments found.")
    else:
        display = boltz_experiments.copy()
        display["created_at"] = display["created_at"].astype(str)
        lines.append(display.to_markdown(index=False))
        for category in ("trunk_only", "structure_tabular", "descriptor_mix"):
            subset = display.loc[display["category"] == category].head(20)
            if subset.empty:
                continue
            lines.extend(
                [
                    "",
                    f"### {category}",
                    "",
                    subset.to_markdown(index=False),
                ]
            )
    lines.extend(
        [
            "",
            "## Read",
            "",
            "- The 13k trunk-fast layer is available for weak-label pretraining and raw `s/z` re-pooling.",
            "- Only the 4652 rcycle=3 full-run rows should be used for pose, confidence, affinity, or contact-gated structure diagnostics.",
            "- Any next Boltz candidate should preserve the existing allpairs reserve member unless replacement evidence is strong.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-per-rcycle", type=int, default=3)
    parser.add_argument("--experiment-limit", type=int, default=40)
    parser.add_argument("--out", type=Path, default=REPORT_PATH)
    args = parser.parse_args()

    engine = get_engine()
    n_compounds = fetch_one_value(
        engine,
        "SELECT COUNT(*) AS n_compounds FROM compounds",
        "n_compounds",
    )
    with engine.connect() as conn:
        compound_ids = (
            pd.read_sql(
                "SELECT id FROM compounds ORDER BY id",
                conn,
            )["id"]
            .astype(int)
            .tolist()
        )
        full = fetch_full_run_summary(conn)
        trunk_df = fetch_trunk_fast(conn)
        boltz_experiments = fetch_boltz_experiments(conn, args.experiment_limit)

    missing = missing_ids(
        compound_ids,
        trunk_df["compound_id"].astype(int).tolist(),
    )
    npz_samples = sample_npz_summaries(
        trunk_df,
        per_rcycle=args.sample_per_rcycle,
        protein_n_res=len(PXR_SEQUENCE),
    )
    report = render_report(
        n_compounds=n_compounds,
        full=full,
        trunk_df=trunk_df,
        missing=missing,
        npz_samples=npz_samples,
        boltz_experiments=boltz_experiments,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

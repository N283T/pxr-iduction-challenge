#!/usr/bin/env -S pixi run python
"""Rebuild and save the canonical Track 1 UMAP CV split.

The output is a local generated artifact under ``data/cv_splits``. It includes
the ordered database identifiers, fold and cluster assignments, and all ten
UMAP coordinates. A sibling JSON file records parameters, versions, and hashes
needed to audit a best-effort reproduction.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "track1_activity" / "src"))

from data import get_engine  # noqa: E402
from splits import build_umap_split  # noqa: E402

N_SPLITS = 5
N_CLUSTERS = 50
SEED = 42
MORGAN_RADIUS = 2
MORGAN_BITS = 2048
UMAP_COMPONENTS = 10
UMAP_N_NEIGHBORS = 30
UMAP_METRIC = "jaccard"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_ordered_inputs(df: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for row in df.itertuples(index=False):
        digest.update(
            f"{row.train_activity_id}\t{row.compound_id}\t{row.smiles}\n".encode()
        )
    return digest.hexdigest()


def package_versions() -> dict[str, str]:
    names = ["numpy", "pandas", "pyarrow", "rdkit", "scikit-learn", "umap-learn"]
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unknown"
    return versions


def git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "cv_splits",
        help="Output directory (default: data/cv_splits)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source = pd.read_sql(
        """
        SELECT
            t.id AS train_activity_id,
            t.compound_id,
            c.molecule_name,
            c.smiles AS raw_smiles,
            c.std_smiles AS smiles,
            t.pec50
        FROM train_activity t
        JOIN compounds c ON c.id = t.compound_id
        ORDER BY t.id
        """,
        get_engine(),
    )
    if source["smiles"].isna().any():
        missing = source.loc[source["smiles"].isna(), "train_activity_id"].tolist()
        raise ValueError(
            f"Missing standardized SMILES for train_activity IDs: {missing[:10]}"
        )

    result = build_umap_split(
        smiles_list=source["smiles"].tolist(),
        n_splits=N_SPLITS,
        n_clusters=N_CLUSTERS,
        seed=SEED,
        metric=UMAP_METRIC,
    )

    artifact = source.copy()
    artifact.insert(0, "train_idx", np.arange(len(artifact), dtype=np.int64))
    artifact["fold"] = result.fold_labels
    artifact["cluster_id"] = result.cluster_labels
    for component in range(UMAP_COMPONENTS):
        artifact[f"umap_{component}"] = result.embedding[:, component]

    parquet_path = output_dir / "canonical_umap_split.parquet"
    metadata_path = output_dir / "canonical_umap_split_meta.json"
    artifact.to_parquet(parquet_path, index=False)

    fold_sizes = artifact.groupby("fold", sort=True).size()
    cluster_sizes = artifact.groupby("cluster_id", sort=True).size()
    metadata = {
        "description": "Best-effort reproduction of the canonical Track 1 training CV split",
        "source": {
            "database": "pxr_challenge",
            "query_order": "train_activity.id ASC",
            "row_count": len(artifact),
            "ordered_input_sha256": hash_ordered_inputs(source),
        },
        "parameters": {
            "morgan_radius": MORGAN_RADIUS,
            "morgan_bits": MORGAN_BITS,
            "umap_components": UMAP_COMPONENTS,
            "umap_n_neighbors": UMAP_N_NEIGHBORS,
            "umap_metric": UMAP_METRIC,
            "kmeans_clusters": N_CLUSTERS,
            "kmeans_n_init": 10,
            "n_splits": N_SPLITS,
            "random_seed": SEED,
            "fold_assignment": "clusters sorted by descending size, greedily assigned to the smallest fold",
        },
        "summary": {
            "fold_sizes": {str(k): int(v) for k, v in fold_sizes.items()},
            "cluster_count": int(cluster_sizes.size),
            "cluster_size_min": int(cluster_sizes.min()),
            "cluster_size_max": int(cluster_sizes.max()),
        },
        "environment": {
            "python": platform.python_version(),
            "packages": package_versions(),
            "git_revision_before_generation": git_revision(),
        },
        "implementation": {
            "generator": {
                "path": "track1_activity/scripts/save_canonical_umap_split.py",
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "split_module": {
                "path": "track1_activity/src/splits.py",
                "sha256": sha256_file(
                    REPO_ROOT / "track1_activity" / "src" / "splits.py"
                ),
            },
        },
        "artifact": {
            "path": str(parquet_path),
            "sha256": sha256_file(parquet_path),
            "columns": artifact.columns.tolist(),
        },
        "reproduction_note": (
            "UMAP/KMeans output can change across dependency versions or input snapshots; "
            "use the hashes and recorded versions when comparing reproductions."
        ),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"Saved {len(artifact):,} rows to {parquet_path}")
    print(f"Saved metadata to {metadata_path}")
    print(f"Fold sizes: {metadata['summary']['fold_sizes']}")
    print(f"Parquet SHA-256: {metadata['artifact']['sha256']}")


if __name__ == "__main__":
    main()

"""Pool variable-length PocketXMol hidden states into fixed-length features."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-manifest", type=Path, required=True)
    parser.add_argument("--out-npz", type=Path, required=True)
    parser.add_argument("--out-features", type=Path, required=True)
    parser.add_argument(
        "--blocks",
        nargs="+",
        default=["pocket", "node", "edge"],
        choices=["pocket", "node", "edge"],
    )
    parser.add_argument(
        "--stats",
        nargs="+",
        default=["mean", "std"],
        choices=["mean", "std", "max", "min", "q10", "q25", "q50", "q75", "q90"],
    )
    parser.add_argument(
        "--timesteps",
        nargs="+",
        help="Optional timestep labels or numeric values, e.g. t1p000 or 1.0.",
    )
    parser.add_argument("--include-deltas", action="store_true")
    parser.add_argument("--include-sizes", action="store_true")
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def stat_vector(array: np.ndarray, stat: str) -> np.ndarray:
    array = array.astype(np.float32, copy=False)
    if stat == "mean":
        return array.mean(axis=0)
    if stat == "std":
        return array.std(axis=0)
    if stat == "max":
        return array.max(axis=0)
    if stat == "min":
        return array.min(axis=0)
    if stat == "q10":
        return np.quantile(array, 0.10, axis=0).astype(np.float32)
    if stat == "q25":
        return np.quantile(array, 0.25, axis=0).astype(np.float32)
    if stat == "q50":
        return np.quantile(array, 0.50, axis=0).astype(np.float32)
    if stat == "q75":
        return np.quantile(array, 0.75, axis=0).astype(np.float32)
    if stat == "q90":
        return np.quantile(array, 0.90, axis=0).astype(np.float32)
    raise ValueError(stat)


def timestep_label(value: str) -> str:
    try:
        return f"t{float(value):.3f}".replace(".", "p")
    except ValueError:
        return value


def sorted_timesteps(keys: list[str]) -> list[str]:
    return sorted({key.rsplit("_", 1)[0] for key in keys})


def pool_file(
    path: Path,
    blocks: list[str],
    stats: list[str],
    selected_timesteps: set[str] | None,
    include_sizes: bool,
) -> tuple[np.ndarray, list[str]]:
    data = np.load(path)
    timesteps = sorted_timesteps(list(data.files))
    if selected_timesteps is not None:
        timesteps = [
            timestep for timestep in timesteps if timestep in selected_timesteps
        ]
    vectors = []
    names = []
    for timestep in timesteps:
        for block in blocks:
            key = f"{timestep}_{block}"
            if key not in data:
                continue
            array = data[key]
            dim = array.shape[1] if array.ndim > 1 else 1
            if include_sizes:
                vectors.append(np.array([array.shape[0]], dtype=np.float32))
                names.append(f"{timestep}_{block}_size")
            for stat in stats:
                vectors.append(stat_vector(array, stat))
                names.extend(f"{timestep}_{block}_{stat}_{i:03d}" for i in range(dim))
    return np.concatenate(vectors).astype(np.float32), names


def add_delta_features(
    path: Path,
    blocks: list[str],
    stats: list[str],
    selected_timesteps: set[str] | None,
    include_sizes: bool,
) -> tuple[np.ndarray, list[str]]:
    data = np.load(path)
    timesteps = sorted_timesteps(list(data.files))
    if selected_timesteps is not None:
        timesteps = [
            timestep for timestep in timesteps if timestep in selected_timesteps
        ]
    if len(timesteps) < 2:
        return np.array([], dtype=np.float32), []
    first = timesteps[0]
    last = timesteps[-1]
    vectors = []
    names = []
    for block in blocks:
        first_key = f"{first}_{block}"
        last_key = f"{last}_{block}"
        if first_key not in data or last_key not in data:
            continue
        first_vec = {stat: stat_vector(data[first_key], stat) for stat in stats}
        last_vec = {stat: stat_vector(data[last_key], stat) for stat in stats}
        dim = data[first_key].shape[1] if data[first_key].ndim > 1 else 1
        if include_sizes:
            vectors.append(
                np.array(
                    [data[last_key].shape[0] - data[first_key].shape[0]],
                    dtype=np.float32,
                )
            )
            names.append(f"{last}_minus_{first}_{block}_size")
        for stat in stats:
            vectors.append(last_vec[stat] - first_vec[stat])
            names.extend(
                f"{last}_minus_{first}_{block}_{stat}_{i:03d}" for i in range(dim)
            )
    if not vectors:
        return np.array([], dtype=np.float32), []
    return np.concatenate(vectors).astype(np.float32), names


def main() -> None:
    args = parse_args()
    rows = read_manifest(args.raw_manifest)
    compound_ids = []
    splits = []
    pec50 = []
    embeddings = []
    feature_names = None
    selected_timesteps = (
        {timestep_label(value) for value in args.timesteps} if args.timesteps else None
    )
    for row in rows:
        hidden_path = Path(row["hidden_path"])
        vec, names = pool_file(
            hidden_path,
            args.blocks,
            args.stats,
            selected_timesteps,
            args.include_sizes,
        )
        if args.include_deltas:
            delta_vec, delta_names = add_delta_features(
                hidden_path,
                args.blocks,
                args.stats,
                selected_timesteps,
                args.include_sizes,
            )
            if len(delta_vec):
                vec = np.concatenate([vec, delta_vec]).astype(np.float32)
                names = names + delta_names
        if len(vec) == 0:
            raise ValueError(f"No features pooled from {hidden_path}")
        if feature_names is None:
            feature_names = names
        elif names != feature_names:
            raise ValueError(
                f"Feature names changed at compound_id={row['compound_id']}"
            )
        embeddings.append(vec)
        compound_ids.append(int(row["compound_id"]))
        splits.append(row["split"])
        pec50.append(np.nan if row["pec50"] == "" else float(row["pec50"]))

    args.out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_npz,
        compound_id=np.array(compound_ids, dtype=np.int32),
        split=np.array(splits, dtype=object),
        pec50=np.array(pec50, dtype=np.float32),
        embedding=np.vstack(embeddings).astype(np.float32),
    )
    args.out_features.write_text(
        "\n".join(feature_names or []) + "\n", encoding="utf-8"
    )
    print(args.out_npz)


if __name__ == "__main__":
    main()

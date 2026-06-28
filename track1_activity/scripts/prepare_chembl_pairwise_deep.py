#!/usr/bin/env -S pixi run python
"""Prepare ChEMBL36 same-assay pair data for ChemProp pairwise pretraining.

The output is intentionally model-agnostic:

* ``molecules.parquet``: one standardized molecule per row.
* ``activities.parquet``: median pChEMBL value per assay/molecule.
* ``pairs.parquet``: sampled same-assay pairs with ``delta = y_a - y_b``.

External ChEMBL structures are standardized with the ChEMBL structure pipeline
before exact challenge InChIKey exclusion. This mirrors the local compound
standardization policy and keeps generated artifacts under ``data/chembl/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from chembl_structure_pipeline import get_parent_mol, standardize_mol
from rdkit import Chem, RDLogger
from sqlalchemy import create_engine

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))

from data import get_engine  # noqa: E402

CHEMBL_URL = "postgresql+psycopg2:///chembl_36?host=/tmp&port=5433"
DEFAULT_OUT_DIR = REPO_ROOT.joinpath("data", "chembl", "pairwise_deep")

RDLogger.DisableLog("rdApp.warning")


@dataclass(frozen=True)
class StandardizedMol:
    smiles: str | None
    inchikey: str | None
    status: str


def standardize_external_smiles(smiles: str) -> StandardizedMol:
    """Apply the ChEMBL structure pipeline and return parent canonical SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return StandardizedMol(None, None, "invalid_input")
    try:
        std_mol = standardize_mol(mol)
    except Exception:
        return StandardizedMol(None, None, "failed_standardize")
    if std_mol is None:
        return StandardizedMol(None, None, "failed_standardize")
    try:
        parent_mol, _exclude = get_parent_mol(std_mol)
    except Exception:
        parent_mol = std_mol
    if parent_mol is None:
        parent_mol = std_mol
    std_smiles = Chem.MolToSmiles(parent_mol)
    if not std_smiles:
        return StandardizedMol(None, None, "failed_parent")
    try:
        inchikey = Chem.MolToInchiKey(parent_mol)
    except Exception:
        inchikey = None
    if not inchikey:
        return StandardizedMol(std_smiles, None, "missing_inchikey")
    raw_canon = Chem.MolToSmiles(mol)
    status = "ok" if std_smiles == raw_canon else "parent_changed"
    return StandardizedMol(std_smiles, inchikey, status)


def load_challenge_inchikeys() -> set[str]:
    """Return exact standardized challenge InChIKeys from the PXR database."""
    rows = pd.read_sql(
        """
        SELECT DISTINCT d.inchikey, c.std_smiles
        FROM compounds c
        LEFT JOIN compound_descriptors d ON d.compound_id = c.id
        WHERE c.id IN (
            SELECT compound_id FROM train_activity
            UNION
            SELECT compound_id FROM test_activity
        )
        ORDER BY d.inchikey NULLS LAST
        """,
        get_engine(),
    )
    keys = set(rows["inchikey"].dropna().astype(str))
    missing = rows[rows["inchikey"].isna() & rows["std_smiles"].notna()]
    for smi in missing["std_smiles"].astype(str):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            key = Chem.MolToInchiKey(mol)
            if key:
                keys.add(key)
    return keys


def parse_csv_arg(value: str) -> tuple[str, ...]:
    return tuple(x.strip() for x in value.split(",") if x.strip())


def build_chembl_sql(args: argparse.Namespace) -> tuple[str, dict[str, object]]:
    endpoint_types = parse_csv_arg(args.standard_types)
    assay_types = parse_csv_arg(args.assay_types)
    params: dict[str, object] = {
        "min_confidence": args.min_confidence,
        "standard_types": endpoint_types,
        "assay_types": assay_types,
    }
    where = [
        "act.pchembl_value IS NOT NULL",
        "act.standard_relation = '='",
        "s.canonical_smiles IS NOT NULL",
        "a.confidence_score >= %(min_confidence)s",
        "act.standard_type IN %(standard_types)s",
    ]
    if assay_types:
        where.append("a.assay_type IN %(assay_types)s")
    order_expr = {
        "assay": "a.assay_id, m.molregno",
        "random": "md5(act.activity_id::text)",
    }[args.order_by]
    limit_sql = ""
    if args.max_activity_rows > 0:
        limit_sql = "LIMIT %(max_activity_rows)s"
        params["max_activity_rows"] = args.max_activity_rows
    sql = f"""
        SELECT
            s.canonical_smiles,
            s.standard_inchi_key AS chembl_inchikey,
            m.chembl_id AS mol_chembl_id,
            a.chembl_id AS assay_chembl_id,
            a.assay_type,
            a.confidence_score,
            a.description AS assay_description,
            t.chembl_id AS target_chembl_id,
            t.target_type,
            act.standard_type,
            act.pchembl_value
        FROM activities act
        JOIN molecule_dictionary m ON m.molregno = act.molregno
        JOIN compound_structures s ON s.molregno = act.molregno
        JOIN assays a ON a.assay_id = act.assay_id
        JOIN target_dictionary t ON t.tid = a.tid
        WHERE {" AND ".join(where)}
        ORDER BY {order_expr}
        {limit_sql}
    """
    return sql, params


def load_raw_chembl(args: argparse.Namespace) -> pd.DataFrame:
    chembl = create_engine(CHEMBL_URL)
    sql, params = build_chembl_sql(args)
    return pd.read_sql(sql, chembl, params=params)


def standardize_rows(
    raw: pd.DataFrame, cache_path: Path | None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique_smiles = raw["canonical_smiles"].dropna().drop_duplicates().tolist()
    cached = pd.DataFrame()
    if cache_path is not None and cache_path.exists():
        cached = pd.read_parquet(cache_path)
        cached = cached[cached["canonical_smiles"].isin(set(unique_smiles))].copy()
    cached_smiles = set(cached["canonical_smiles"]) if not cached.empty else set()
    todo = [smi for smi in unique_smiles if smi not in cached_smiles]
    records = []
    for i, smiles in enumerate(todo, start=1):
        std = standardize_external_smiles(smiles)
        records.append(
            {
                "canonical_smiles": smiles,
                "std_smiles": std.smiles,
                "std_inchikey": std.inchikey,
                "standardize_status": std.status,
            }
        )
        if i % 25000 == 0:
            print(f"  standardized {i:,}/{len(todo):,} uncached unique SMILES")
    new_df = pd.DataFrame(records)
    std_df = pd.concat([cached, new_df], ignore_index=True)
    std_df = std_df.drop_duplicates("canonical_smiles", keep="last")
    if cache_path is not None and not new_df.empty:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():
            full_cache = pd.read_parquet(cache_path)
            full_cache = pd.concat([full_cache, new_df], ignore_index=True)
            full_cache = full_cache.drop_duplicates("canonical_smiles", keep="last")
        else:
            full_cache = new_df
        full_cache.to_parquet(cache_path, index=False)
    rows = raw.merge(std_df, on="canonical_smiles", how="left")
    rows = rows.dropna(subset=["std_smiles", "std_inchikey", "pchembl_value"]).copy()
    return rows, std_df


def collapse_activities(
    rows: pd.DataFrame, challenge_keys: set[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = rows[~rows["std_inchikey"].isin(challenge_keys)].copy()
    rows["assay_id"] = rows["assay_chembl_id"].astype(str)
    rows["assay_endpoint"] = rows["standard_type"].astype(str)
    activities = (
        rows.groupby(
            [
                "assay_id",
                "std_inchikey",
                "std_smiles",
                "assay_type",
                "assay_endpoint",
                "confidence_score",
                "target_chembl_id",
                "target_type",
            ],
            as_index=False,
        )
        .agg(
            value=("pchembl_value", "median"),
            n_measurements=("pchembl_value", "size"),
            mol_chembl_id=("mol_chembl_id", "first"),
        )
        .reset_index(drop=True)
    )
    molecules = (
        activities[["std_inchikey", "std_smiles"]]
        .drop_duplicates()
        .sort_values("std_inchikey")
        .reset_index(drop=True)
    )
    molecules.insert(0, "mol_id", np.arange(len(molecules), dtype=np.int64))
    activities = activities.merge(
        molecules[["mol_id", "std_inchikey"]], on="std_inchikey", how="left"
    )
    activities.insert(0, "activity_id", np.arange(len(activities), dtype=np.int64))
    return molecules, activities


def filter_assays(
    activities: pd.DataFrame, args: argparse.Namespace
) -> tuple[pd.DataFrame, pd.DataFrame]:
    assay_summary = (
        activities.groupby("assay_id")
        .agg(
            n_molecules=("mol_id", "nunique"),
            n_activities=("activity_id", "size"),
            value_mean=("value", "mean"),
            value_std=("value", "std"),
            value_min=("value", "min"),
            value_max=("value", "max"),
            value_nunique=("value", "nunique"),
            assay_type=("assay_type", "first"),
            endpoint=("assay_endpoint", "first"),
            confidence_score=("confidence_score", "first"),
            target_chembl_id=("target_chembl_id", "first"),
            target_type=("target_type", "first"),
        )
        .reset_index()
    )
    keep = assay_summary[
        assay_summary["n_molecules"].ge(args.min_assay_n)
        & assay_summary["value_std"].ge(args.min_assay_std)
        & assay_summary["value_nunique"].ge(args.min_assay_unique)
    ].copy()
    filtered = activities[activities["assay_id"].isin(set(keep["assay_id"]))].copy()
    filtered = filtered.reset_index(drop=True)
    filtered["activity_id"] = np.arange(len(filtered), dtype=np.int64)
    return filtered, keep


def sample_pairs(activities: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rng = np.random.default_rng(args.seed)
    assay_ids = np.array(sorted(activities["assay_id"].unique()))
    rng.shuffle(assay_ids)
    n_val = (
        max(1, int(round(len(assay_ids) * args.val_assay_frac)))
        if len(assay_ids) > 1
        else 0
    )
    val_assays = set(assay_ids[:n_val])

    pair_rows: list[dict[str, object]] = []
    for assay_id, sub in activities.groupby("assay_id", sort=False):
        sub = sub.sort_values("activity_id").reset_index(drop=True)
        n = len(sub)
        values = sub["value"].to_numpy(dtype=np.float32)
        all_possible = n * (n - 1) // 2
        local_pairs: list[tuple[int, int, float]] = []
        if all_possible <= args.max_pairs_per_assay:
            for i in range(n):
                for j in range(i + 1, n):
                    delta = float(values[i] - values[j])
                    if abs(delta) >= args.min_pair_delta:
                        local_pairs.append((i, j, delta))
        else:
            seen: set[tuple[int, int]] = set()
            max_trials = args.max_pairs_per_assay * 30
            trials = 0
            while len(local_pairs) < args.max_pairs_per_assay and trials < max_trials:
                i, j = rng.choice(n, size=2, replace=False)
                if i > j:
                    i, j = j, i
                if (i, j) in seen:
                    trials += 1
                    continue
                seen.add((int(i), int(j)))
                delta = float(values[i] - values[j])
                if abs(delta) >= args.min_pair_delta:
                    local_pairs.append((int(i), int(j), delta))
                trials += 1
        if len(local_pairs) > args.max_pairs_per_assay:
            keep_idx = rng.choice(
                len(local_pairs), size=args.max_pairs_per_assay, replace=False
            )
            local_pairs = [local_pairs[i] for i in keep_idx]
        split = "val" if assay_id in val_assays else "train"
        for i, j, delta in local_pairs:
            row_i = sub.iloc[i]
            row_j = sub.iloc[j]
            pair_rows.append(
                {
                    "assay_id": assay_id,
                    "activity_id_a": int(row_i["activity_id"]),
                    "activity_id_b": int(row_j["activity_id"]),
                    "mol_id_a": int(row_i["mol_id"]),
                    "mol_id_b": int(row_j["mol_id"]),
                    "value_a": float(row_i["value"]),
                    "value_b": float(row_j["value"]),
                    "delta": delta,
                    "weight": min(abs(delta), args.max_pair_weight),
                    "split": split,
                }
            )
            if args.add_reverse_pairs:
                pair_rows.append(
                    {
                        "assay_id": assay_id,
                        "activity_id_a": int(row_j["activity_id"]),
                        "activity_id_b": int(row_i["activity_id"]),
                        "mol_id_a": int(row_j["mol_id"]),
                        "mol_id_b": int(row_i["mol_id"]),
                        "value_a": float(row_j["value"]),
                        "value_b": float(row_i["value"]),
                        "delta": -delta,
                        "weight": min(abs(delta), args.max_pair_weight),
                        "split": split,
                    }
                )
    pairs = pd.DataFrame(pair_rows)
    if pairs.empty:
        raise RuntimeError("No same-assay pairs survived the filters.")
    pairs.insert(0, "pair_id", np.arange(len(pairs), dtype=np.int64))
    return pairs


def write_report(
    out_dir: Path,
    args: argparse.Namespace,
    raw: pd.DataFrame,
    std_df: pd.DataFrame,
    molecules: pd.DataFrame,
    activities: pd.DataFrame,
    assay_summary: pd.DataFrame,
    pairs: pd.DataFrame,
) -> None:
    report = {
        "config": {key: str(value) for key, value in vars(args).items()},
        "raw_rows": int(len(raw)),
        "unique_raw_smiles": int(raw["canonical_smiles"].nunique()),
        "standardize_status": std_df["standardize_status"]
        .value_counts(dropna=False)
        .to_dict(),
        "molecules": int(len(molecules)),
        "activities": int(len(activities)),
        "assays": int(activities["assay_id"].nunique()),
        "pairs": int(len(pairs)),
        "pairs_by_split": pairs["split"].value_counts().to_dict(),
        "assay_summary": {
            "mean_n_molecules": float(assay_summary["n_molecules"].mean()),
            "median_n_molecules": float(assay_summary["n_molecules"].median()),
            "mean_value_std": float(assay_summary["value_std"].mean()),
        },
    }
    out_dir.joinpath("prepare_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--standard-types", default="IC50,Ki,Kd,EC50,AC50")
    parser.add_argument("--assay-types", default="B,F")
    parser.add_argument("--min-confidence", type=int, default=8)
    parser.add_argument("--max-activity-rows", type=int, default=250000)
    parser.add_argument("--order-by", choices=["assay", "random"], default="assay")
    parser.add_argument(
        "--standardize-cache",
        type=Path,
        default=REPO_ROOT.joinpath(
            "data", "chembl", "pairwise_deep_standardized_smiles.parquet"
        ),
    )
    parser.add_argument("--min-assay-n", type=int, default=20)
    parser.add_argument("--min-assay-std", type=float, default=0.25)
    parser.add_argument("--min-assay-unique", type=int, default=8)
    parser.add_argument("--min-pair-delta", type=float, default=0.25)
    parser.add_argument("--max-pair-weight", type=float, default=2.0)
    parser.add_argument("--max-pairs-per-assay", type=int, default=1000)
    parser.add_argument("--val-assay-frac", type=float, default=0.05)
    parser.add_argument(
        "--add-reverse-pairs", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading ChEMBL36 activity rows")
    raw = load_raw_chembl(args)
    print(f"  raw rows: {len(raw):,}")

    print("Standardizing external SMILES with ChEMBL structure pipeline")
    standardized, std_df = standardize_rows(raw, args.standardize_cache)
    print(f"  standardized valid rows: {len(standardized):,}")

    print("Excluding exact challenge InChIKeys and collapsing assay/molecule values")
    challenge_keys = load_challenge_inchikeys()
    molecules, activities = collapse_activities(standardized, challenge_keys)
    print(f"  molecules after exclusion: {len(molecules):,}")
    print(f"  assay/molecule activities: {len(activities):,}")

    print("Filtering assays and sampling same-assay pairs")
    activities, assay_summary = filter_assays(activities, args)
    pairs = sample_pairs(activities, args)
    print(f"  kept assays: {activities['assay_id'].nunique():,}")
    print(f"  pairs: {len(pairs):,} ({pairs['split'].value_counts().to_dict()})")

    molecules.to_parquet(args.out_dir.joinpath("molecules.parquet"), index=False)
    activities.to_parquet(args.out_dir.joinpath("activities.parquet"), index=False)
    assay_summary.to_csv(args.out_dir.joinpath("assay_summary.csv"), index=False)
    pairs.to_parquet(args.out_dir.joinpath("pairs.parquet"), index=False)
    write_report(
        args.out_dir, args, raw, std_df, molecules, activities, assay_summary, pairs
    )
    print(f"Saved pairwise data to {args.out_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Export exact molecule matches between Track 1 train and ChEMBL 36."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sqlalchemy import Engine, create_engine, text

PXR_URL = "postgresql+psycopg2:///pxr_challenge?host=/tmp&port=5433"
CHEMBL_URL = "postgresql+psycopg2:///chembl_36?host=/tmp&port=5433"
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "docs" / "data" / "chembl"


def load_train_keys(engine: Engine) -> pd.DataFrame:
    """Load raw and standardized InChIKeys for every Track 1 train row."""
    return pd.read_sql(
        text(
            """
            SELECT
                t.id AS train_id,
                c.id AS compound_id,
                c.molecule_name AS train_molecule_name,
                mol_inchikey(c.mol)::text AS raw_inchi_key,
                mol_inchikey(c.std_mol)::text AS standard_inchi_key
            FROM train_activity t
            JOIN compounds c ON c.id = t.compound_id
            ORDER BY t.id
            """
        ),
        engine,
    )


def load_chembl_molecules(engine: Engine, keys: list[str]) -> pd.DataFrame:
    """Load ChEMBL molecules whose standard InChIKey exactly matches a key."""
    return pd.read_sql(
        text(
            """
            SELECT
                cs.standard_inchi_key AS match_inchi_key,
                md.chembl_id,
                md.pref_name AS chembl_pref_name
            FROM compound_structures cs
            JOIN molecule_dictionary md ON md.molregno = cs.molregno
            WHERE cs.standard_inchi_key = ANY(:keys)
            ORDER BY cs.standard_inchi_key, md.chembl_id
            """
        ),
        engine,
        params={"keys": keys},
    )


def exact_matches(
    train: pd.DataFrame, chembl: pd.DataFrame, key_column: str
) -> pd.DataFrame:
    """Return train rows with an exact ChEMBL match for the selected key."""
    result = train.merge(
        chembl,
        left_on=key_column,
        right_on="match_inchi_key",
        how="inner",
        validate="many_to_many",
    )
    return result[
        [
            "train_id",
            "compound_id",
            "train_molecule_name",
            key_column,
            "chembl_id",
            "chembl_pref_name",
        ]
    ].sort_values(["train_id", "chembl_id"], ignore_index=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train = load_train_keys(create_engine(PXR_URL))
    all_keys = sorted(
        set(train["standard_inchi_key"].dropna()) | set(train["raw_inchi_key"].dropna())
    )
    chembl = load_chembl_molecules(create_engine(CHEMBL_URL), all_keys)

    standard = exact_matches(train, chembl, "standard_inchi_key")
    raw = exact_matches(train, chembl, "raw_inchi_key")

    standard_path = args.output_dir / "train_chembl36_exact_matches.csv"
    raw_path = args.output_dir / "train_chembl36_exact_matches_raw.csv"
    standard.to_csv(standard_path, index=False)
    raw.to_csv(raw_path, index=False)

    standard_ids = set(standard["train_id"])
    raw_ids = set(raw["train_id"])
    print(f"train rows: {len(train)}")
    print(f"standard exact matches: {len(standard_ids)}")
    print(f"raw exact matches: {len(raw_ids)}")
    print(f"standard-only matches: {len(standard_ids - raw_ids)}")
    print(f"raw-only matches: {len(raw_ids - standard_ids)}")
    print(
        f"standard matches with ChEMBL pref_name: {standard['chembl_pref_name'].notna().sum()}"
    )
    print(f"wrote: {standard_path}")
    print(f"wrote: {raw_path}")


if __name__ == "__main__":
    main()

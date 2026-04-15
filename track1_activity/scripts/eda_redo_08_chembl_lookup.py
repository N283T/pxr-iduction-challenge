#!/usr/bin/env -S pixi run python
"""Cross-reference drop candidates against a local ChEMBL 36 instance.

For every compound in 07_drop_candidates.parquet, find its ChEMBL entry
(chembl_id, pref_name, max_phase, molecule_type) by InChIKey match
against chembl_36.compound_structures.

Also checks whether ChEMBL has any recorded activity against human PXR
(NR1I2, UniProt O75469) so we know which drop candidates are already
documented PXR ligands.

Outputs:
  - 08_drop_candidates_chembl.parquet   - drop list enriched with ChEMBL metadata
  - 08_pxr_activities.parquet           - any ChEMBL activity rows on PXR for drop cpds
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.joinpath("src")))

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT.joinpath("data", "eda_redo")

CHEMBL_URL = "postgresql+psycopg2:///chembl_36?host=/tmp&port=5433"


def main() -> None:
    drops = pd.read_parquet(DATA_DIR.joinpath("07_drop_candidates.parquet"))
    print(f"[08] drop candidates: {len(drops)}")

    keys = drops["inchikey"].dropna().unique().tolist()
    print(f"[08] unique InChIKeys to resolve: {len(keys)}")

    engine = create_engine(CHEMBL_URL)
    with engine.connect() as conn:
        # 1. Resolve InChIKey -> chembl_id + pref_name
        meta_q = text(
            """
            SELECT
                s.standard_inchi_key AS inchikey,
                m.chembl_id,
                m.pref_name,
                m.max_phase,
                m.molecule_type,
                m.therapeutic_flag,
                m.natural_product,
                m.inorganic_flag,
                m.first_approval
            FROM compound_structures s
            JOIN molecule_dictionary m ON m.molregno = s.molregno
            WHERE s.standard_inchi_key = ANY(:keys)
            """
        )
        meta = pd.read_sql(meta_q, conn, params={"keys": keys})
        print(f"[08] resolved in ChEMBL: {len(meta)}  / {len(keys)}")

        # 2. Any activity on human PXR (NR1I2, UniProt O75469)? ChEMBL target chembl_id
        #    for PXR is CHEMBL2034. Pull any reported activity for the drop cpds
        #    against this target.
        pxr_q = text(
            """
            SELECT
                s.standard_inchi_key AS inchikey,
                m.chembl_id            AS mol_chembl_id,
                m.pref_name,
                a.chembl_id            AS assay_chembl_id,
                a.description          AS assay_description,
                act.standard_type,
                act.standard_value,
                act.standard_units,
                act.standard_relation,
                act.pchembl_value,
                t.pref_name            AS target_name,
                t.chembl_id            AS target_chembl_id
            FROM compound_structures s
            JOIN molecule_dictionary m ON m.molregno = s.molregno
            JOIN activities act        ON act.molregno = m.molregno
            JOIN assays a              ON a.assay_id = act.assay_id
            JOIN target_dictionary t   ON t.tid = a.tid
            WHERE s.standard_inchi_key = ANY(:keys)
              AND t.chembl_id = 'CHEMBL2034'
            """
        )
        pxr = pd.read_sql(pxr_q, conn, params={"keys": keys})
        print(
            f"[08] ChEMBL rows recording activity on PXR (CHEMBL2034): {len(pxr)}"
            f"  for {pxr['inchikey'].nunique() if len(pxr) else 0} unique compounds"
        )

    enriched = drops.merge(meta, on="inchikey", how="left")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    enriched_path = DATA_DIR.joinpath("08_drop_candidates_chembl.parquet")
    enriched.to_parquet(enriched_path, index=False)
    print(f"[08] wrote {enriched_path}")
    if len(pxr):
        pxr_path = DATA_DIR.joinpath("08_pxr_activities.parquet")
        pxr.to_parquet(pxr_path, index=False)
        print(f"[08] wrote {pxr_path}")

    # ------------------------------------------------------------------
    # Summary reporting
    # ------------------------------------------------------------------
    print()
    print("[08] drop candidates with ChEMBL hits:")
    in_chembl = enriched[enriched["chembl_id"].notna()]
    print(f"  total:                   {len(in_chembl)} / {len(enriched)}")
    print(
        f"  big_tail in ChEMBL:      {(in_chembl['drop_reason'] == 'big_tail').sum()}"
    )
    print(
        f"  small_tail in ChEMBL:    {(in_chembl['drop_reason'] == 'small_tail').sum()}"
    )
    print(f"  both in ChEMBL:          {(in_chembl['drop_reason'] == 'both').sum()}")

    print()
    print("[08] approved drugs among drop candidates (max_phase = 4):")
    approved = in_chembl[in_chembl["max_phase"] == 4]
    print(
        approved[
            [
                "compound_id",
                "drop_reason",
                "chembl_id",
                "pref_name",
                "molecule_type",
                "first_approval",
                "num_heavy_atoms",
                "amw",
                "train_pec50",
            ]
        ].to_string(index=False, max_colwidth=50)
    )

    print()
    print("[08] any drugs in later phases (max_phase >= 2):")
    phased = in_chembl[in_chembl["max_phase"] >= 2].sort_values(
        "max_phase", ascending=False
    )
    print(f"  N={len(phased)}")

    print()
    if len(pxr):
        print("[08] confirmed PXR ligands in ChEMBL (drop candidates):")
        hits = (
            pxr.groupby(["mol_chembl_id", "pref_name"])
            .agg(
                n_rows=("inchikey", "size"),
                types=(
                    "standard_type",
                    lambda s: ",".join(sorted(set(s.dropna())))[:80],
                ),
                pchembl_max=("pchembl_value", "max"),
            )
            .reset_index()
        )
        print(hits.to_string(index=False, max_colwidth=70))


if __name__ == "__main__":
    main()

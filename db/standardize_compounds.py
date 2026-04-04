"""Standardize all compounds using ChEMBL structure pipeline."""

import logging

import psycopg2
from chembl_structure_pipeline import standardize_mol, get_parent_mol
from rdkit import Chem

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

DB_PARAMS = {"dbname": "pxr_challenge", "host": "/tmp", "port": 5433}


def standardize_smiles(smiles: str) -> tuple[str | None, str]:
    """Standardize a SMILES string using ChEMBL pipeline.

    Returns (standardized_smiles, status).
    Status: 'ok', 'parent_changed', 'failed_standardize', 'failed_parent', 'invalid_input'
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "invalid_input"

    # Step 1: Standardize (normalize charges, functional groups, etc.)
    try:
        std_mol = standardize_mol(mol)
    except Exception as e:
        logger.warning(f"Standardization failed for {smiles}: {e}")
        return None, "failed_standardize"

    if std_mol is None:
        return None, "failed_standardize"

    # Step 2: Get parent molecule (strip salts, solvents)
    try:
        parent_mol, exclude_flag = get_parent_mol(std_mol)
    except Exception as e:
        logger.warning(f"Parent extraction failed for {smiles}: {e}")
        # Fall back to standardized mol without parent extraction
        parent_mol = std_mol

    if parent_mol is None:
        parent_mol = std_mol

    # Generate canonical SMILES
    std_smiles = Chem.MolToSmiles(parent_mol)
    if not std_smiles:
        return None, "failed_parent"

    status = "ok" if std_smiles == Chem.MolToSmiles(mol) else "parent_changed"
    return std_smiles, status


def main():
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()

    # Fetch all compounds
    cur.execute("SELECT id, smiles FROM compounds ORDER BY id")
    compounds = cur.fetchall()
    total = len(compounds)
    logger.info(f"Processing {total} compounds...")

    stats = {"ok": 0, "parent_changed": 0, "failed_standardize": 0,
             "failed_parent": 0, "invalid_input": 0}
    failed_ids = []

    for i, (cid, smiles) in enumerate(compounds):
        std_smiles, status = standardize_smiles(smiles)
        stats[status] += 1

        if std_smiles is None:
            failed_ids.append((cid, smiles, status))
            # Keep original as fallback
            std_smiles = smiles

        cur.execute(
            "UPDATE compounds SET std_smiles = %s, std_mol = mol_from_smiles(%s::cstring) WHERE id = %s",
            (std_smiles, std_smiles, cid),
        )

        if (i + 1) % 2000 == 0:
            conn.commit()
            logger.info(f"  Processed {i + 1}/{total}")

    conn.commit()

    # Summary
    logger.info(f"\nStandardization complete:")
    for status, count in stats.items():
        logger.info(f"  {status}: {count}")

    if failed_ids:
        logger.warning(f"\nFailed compounds ({len(failed_ids)}):")
        for cid, smiles, status in failed_ids:
            logger.warning(f"  id={cid} status={status} smiles={smiles[:80]}")

    # Check how many SMILES actually changed
    cur.execute("SELECT count(*) FROM compounds WHERE smiles != std_smiles")
    changed = cur.fetchone()[0]
    logger.info(f"\nSMILES changed after standardization: {changed}/{total}")

    # Check for duplicates in std_smiles
    cur.execute("""
        SELECT std_smiles, count(*) as cnt
        FROM compounds
        GROUP BY std_smiles
        HAVING count(*) > 1
        ORDER BY cnt DESC
        LIMIT 20
    """)
    dupes = cur.fetchall()
    if dupes:
        logger.warning(f"\nDuplicate std_smiles found: {len(dupes)} groups")
        for std_smi, cnt in dupes[:5]:
            logger.warning(f"  {cnt}x: {std_smi[:80]}")
    else:
        logger.info("\nNo duplicate std_smiles found.")

    # Verify std_mol column
    cur.execute("SELECT count(*) FROM compounds WHERE std_mol IS NOT NULL")
    valid_mols = cur.fetchone()[0]
    logger.info(f"Valid std_mol: {valid_mols}/{total}")

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()

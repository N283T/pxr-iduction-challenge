"""Render the 46 potent seed compounds as a grid image."""

from pathlib import Path

import psycopg2
from rdkit import Chem
from rdkit.Chem import Draw


def main() -> None:
    conn = psycopg2.connect(host="/tmp", port=5433, dbname="pxr_challenge")
    cur = conn.cursor()
    cur.execute(
        """
        SELECT c.molecule_name, c.std_smiles, t.pec50 AS train_pec50,
               ca.pec50 AS counter_pec50,
               (t.pec50 - ca.pec50) AS delta
        FROM train_activity t
        JOIN compounds c ON c.id = t.compound_id
        JOIN counter_assay ca ON ca.compound_id = t.compound_id
        WHERE t.pec50 >= 6.0
          AND (t.pec50 - ca.pec50) >= 1.5
        ORDER BY (t.pec50 - ca.pec50) DESC
        """
    )
    rows = cur.fetchall()
    conn.close()

    print(f"potent-46 count: {len(rows)}")

    mols = []
    legends = []
    for name, smi, pec50, counter, delta in rows:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        mols.append(mol)
        legends.append(f"{name}\npEC50={pec50:.2f}  Δ={delta:.2f}")

    out_dir = Path("docs/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir.joinpath("potent-46.png")

    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=6,
        subImgSize=(280, 260),
        legends=legends,
        useSVG=False,
    )
    img.save(str(out_path))
    print(f"Saved: {out_path}  ({out_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()

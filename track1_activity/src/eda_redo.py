"""Issue #52 re-EDA helpers.

Builds a single wide master DataFrame that joins every per-compound signal we
care about for noise-removal analysis: structural descriptors, train/test/aux
split membership, dose-response activity (pEC50, Emax), counter-assay
selectivity, single-concentration screening response, and Boltz-2 pose /
affinity / PoseBusters outputs.

Downstream scripts (eda_redo_01_* ... eda_redo_05_*) consume the parquet this
module produces so that every plot and every drop-list candidate is joined
with activity values by default -- we never look at pure structure in
isolation.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

DB_URL = "postgresql+psycopg2:///pxr_challenge?host=/tmp&port=5433"

# pEC50 cutoff used in the OpenADMET blog post to define "potent PXR inducers"
# (EC50 <= 1 uM). Used as a flag, not a hard threshold.
POTENT_PEC50 = 6.0

MASTER_PARQUET_PATH = (
    Path(__file__)
    .resolve()
    .parent.parent.parent.joinpath("data", "eda_redo", "master.parquet")
)


def _master_query() -> str:
    """SQL for the master join.

    One row per compound (LEFT JOIN keeps every row in `compounds`), with:
      - structural descriptors
      - activity from train_activity (dose-response, one row per cid)
      - counter_assay (selectivity)
      - single_concentration aggregated per cid (strongest response)
      - boltz2 pose / affinity / confidence
      - boltz2_posebusters summary
      - split membership flags (in_train / in_test / in_counter / in_single)
    """
    return """
    WITH single_agg AS (
        SELECT
            compound_id,
            COUNT(*)                         AS single_n_rows,
            MAX(median_log2_fc)              AS single_max_log2_fc,
            MIN(p_value)                     AS single_min_p,
            MAX(neg_log10_fdr)               AS single_max_neg_log10_fdr,
            MODE() WITHIN GROUP (ORDER BY compound_class) AS single_dominant_class
        FROM single_concentration
        GROUP BY compound_id
    )
    SELECT
        c.id                                 AS compound_id,
        c.smiles,
        c.std_smiles,
        (ta.compound_id IS NOT NULL)         AS in_train,
        (te.compound_id IS NOT NULL)         AS in_test,
        (ca.compound_id IS NOT NULL)         AS in_counter,
        (sa.compound_id IS NOT NULL)         AS in_single,
        -- structural descriptors (selected)
        d.num_heavy_atoms,
        d.amw,
        d.logp,
        d.tpsa,
        d.fractioncsp3,
        d.num_rings,
        d.num_aromatic_rings,
        d.num_rotatable_bonds,
        d.num_heteroatoms,
        d.hba,
        d.hbd,
        d.murcko_scaffold,
        d.formula,
        d.inchikey,
        -- dose-response activity (train)
        ta.pec50                             AS train_pec50,
        ta.pec50_ci_lower                    AS train_pec50_ci_lower,
        ta.pec50_ci_upper                    AS train_pec50_ci_upper,
        ta.pec50_std_error                   AS train_pec50_std_error,
        ta.emax_estimate                     AS train_emax,
        ta.emax_vs_pos_ctrl                  AS train_emax_vs_pos_ctrl,
        (ta.pec50 IS NOT NULL AND ta.pec50 >= :potent) AS train_is_potent,
        -- counter-assay (selectivity; same compounds as train)
        ca.pec50                             AS counter_pec50,
        ca.emax_vs_pos_ctrl                  AS counter_emax_vs_pos_ctrl,
        (ca.pec50 - ta.pec50)                AS counter_minus_train_pec50,
        -- single-concentration screen (aggregated)
        sa.single_n_rows,
        sa.single_max_log2_fc,
        sa.single_min_p,
        sa.single_max_neg_log10_fdr,
        sa.single_dominant_class,
        -- boltz2 pose / affinity / confidence
        b.preprocessing_failed               AS b2_preprocessing_failed,
        b.failure_reason                     AS b2_failure_reason,
        b.ligand_oversize                    AS b2_ligand_oversize,
        b.ligand_atom_count                  AS b2_ligand_atom_count,
        b.ligand_to_pocket_distance_a        AS b2_pocket_distance_a,
        b.affinity_pred_value                AS b2_affinity_pred,
        b.affinity_probability_binary        AS b2_affinity_prob,
        b.confidence_score                   AS b2_confidence,
        b.iptm                               AS b2_iptm,
        b.ligand_iptm                        AS b2_ligand_iptm,
        b.complex_plddt                      AS b2_complex_plddt,
        -- posebusters summary
        pb.num_checks                        AS pb_num_checks,
        pb.num_passed                        AS pb_num_passed,
        pb.all_passed                        AS pb_all_passed,
        pb.intramol_passed                   AS pb_intramol_passed,
        pb.intermol_passed                   AS pb_intermol_passed,
        pb.minimum_distance_to_protein       AS pb_no_protein_clash,
        pb.internal_energy                   AS pb_internal_energy_ok
    FROM compounds c
    LEFT JOIN compound_descriptors d  ON d.compound_id  = c.id
    LEFT JOIN train_activity        ta  ON ta.compound_id = c.id
    LEFT JOIN test_activity         te  ON te.compound_id = c.id
    LEFT JOIN counter_assay         ca  ON ca.compound_id = c.id
    LEFT JOIN single_agg            sa  ON sa.compound_id = c.id
    LEFT JOIN compound_boltz2       b   ON b.compound_id  = c.id
    LEFT JOIN compound_boltz2_posebusters pb ON pb.compound_id = c.id
    ORDER BY c.id
    """


def build_master_df() -> pd.DataFrame:
    """Run the master join and return a wide DataFrame (one row per compound)."""
    engine = create_engine(DB_URL)
    with engine.connect() as conn:
        df = pd.read_sql(text(_master_query()), conn, params={"potent": POTENT_PEC50})
    # Derive a single categorical split column for convenient group-bys while
    # keeping the multi-hot booleans for precise membership queries.
    df["split"] = _derive_split(df)
    return df


def _derive_split(df: pd.DataFrame) -> pd.Series:
    """Assign a single split label with an obvious priority order.

    test > train > counter-only > single-only > other. Compounds in both
    train and counter end up as 'train' because that is where the model
    target lives; the `in_counter` flag still says they have counter data.
    """

    def tag(row: pd.Series) -> str:
        if row["in_test"]:
            return "test"
        if row["in_train"]:
            return "train"
        if row["in_counter"]:
            return "counter_only"
        if row["in_single"]:
            return "single_only"
        return "other"

    return df.apply(tag, axis=1).astype("category")


def save_master(df: pd.DataFrame, path: Path = MASTER_PARQUET_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def load_master(path: Path = MASTER_PARQUET_PATH) -> pd.DataFrame:
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# Molecule rendering helpers (for static PNG grids and marimo SVG tables)
# ---------------------------------------------------------------------------


def mol_to_svg(smiles: str, width: int = 250, height: int = 200) -> str:
    """Render a SMILES to an SVG string.

    Matches the pattern in https://github.com/N283T/cheminfo-in-marimo and
    chemspace-marimo so notebooks can wrap the output in `mo.Html`.
    """
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D

    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return f"<svg width='{width}' height='{height}'><text x='10' y='20'>invalid SMILES</text></svg>"
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def draw_mol_grid_png(
    smiles_list: list[str],
    legends: list[str],
    out_path: Path,
    mols_per_row: int = 5,
    sub_img_size: tuple[int, int] = (280, 220),
) -> Path:
    """Render a grid of 2D structures to PNG with custom legends.

    Used by eda_redo scripts to visually confirm that flagged compounds
    really are the chemotypes we think they are (cardenolides, peptide
    dimers, etc.) rather than trusting descriptor numbers alone.
    """
    from rdkit import Chem
    from rdkit.Chem import Draw

    mols = [Chem.MolFromSmiles(s) if s else None for s in smiles_list]
    keep = [(m, lg) for m, lg in zip(mols, legends) if m is not None]
    if not keep:
        out_path.write_bytes(b"")
        return out_path
    mols_ok, legends_ok = zip(*keep)
    img = Draw.MolsToGridImage(
        list(mols_ok),
        molsPerRow=mols_per_row,
        subImgSize=sub_img_size,
        legends=list(legends_ok),
        useSVG=False,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return out_path

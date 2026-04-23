-- OpenEye oemedchem / oemolprop 2D descriptors (per compound).
--
-- 16 scalar descriptors plus Bemis-Murcko scaffold SMILES. OE's
-- implementations differ from RDKit (XLogP vs Crippen logP, 2dPSA vs
-- RDKit TPSA, etc.) so these add decorrelated signal even when names
-- overlap.

CREATE TABLE IF NOT EXISTS compound_oemedchem (
    compound_id                   INTEGER PRIMARY KEY REFERENCES compounds(id),
    -- Physicochemical (oemolprop)
    xlogp                         REAL,
    psa_2d                        REAL,
    mw                            REAL,
    hba                           INTEGER,
    hbd                           INTEGER,
    lipinski_hba                  INTEGER,
    lipinski_hbd                  INTEGER,
    aromatic_ring_count           INTEGER,
    rotatable_bond_count          INTEGER,
    fraction_csp3                 REAL,
    halide_fraction               REAL,
    longest_unbranched_c_chain    INTEGER,
    longest_unbranched_heavy_chain INTEGER,
    anionic_carbon_count          INTEGER,
    num_unspecified_atom_stereo   INTEGER,
    num_unspecified_bond_stereo   INTEGER,
    -- Structural (oemedchem)
    bemis_murcko_scaffold_smiles  TEXT,
    computed_at                   TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_compound_oemedchem_scaffold
    ON compound_oemedchem(bemis_murcko_scaffold_smiles);

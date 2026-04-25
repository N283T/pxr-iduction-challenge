-- PoseBusters pose quality checks for Boltz-2 predictions.
--
-- 19 individual checks + summary counts per compound. A row exists only
-- for compounds that have a Boltz-2 prediction (ie. compound_boltz2.
-- preprocessing_failed = FALSE). Populated by
-- track1_activity/boltz2/scripts/boltz2_posebusters.py.
--
-- PoseBusters column names use ``-`` which is invalid in SQL, so they
-- are rewritten with underscores. The ordering below matches the output
-- order of PoseBusters ``config="dock"``.

CREATE TABLE IF NOT EXISTS compound_boltz2_posebusters (
    compound_id INT PRIMARY KEY REFERENCES compounds(id),

    -- Aggregate summary
    num_checks      INT,
    num_passed      INT,
    all_passed      BOOLEAN,     -- all 19 checks passed
    intramol_passed BOOLEAN,     -- 13 intramolecular checks all passed
    intermol_passed BOOLEAN,     -- 6 protein-ligand checks all passed

    -- Intramolecular checks (ligand-only)
    mol_pred_loaded                 BOOLEAN,
    mol_cond_loaded                 BOOLEAN,
    sanitization                    BOOLEAN,
    inchi_convertible               BOOLEAN,
    all_atoms_connected             BOOLEAN,
    no_radicals                     BOOLEAN,
    bond_lengths                    BOOLEAN,
    bond_angles                     BOOLEAN,
    internal_steric_clash           BOOLEAN,
    aromatic_ring_flatness          BOOLEAN,
    non_aromatic_ring_non_flatness  BOOLEAN,
    double_bond_flatness            BOOLEAN,
    internal_energy                 BOOLEAN,

    -- Intermolecular checks (protein-ligand)
    protein_ligand_maximum_distance           BOOLEAN,
    minimum_distance_to_protein               BOOLEAN,   -- True = no clash
    minimum_distance_to_organic_cofactors     BOOLEAN,
    minimum_distance_to_inorganic_cofactors   BOOLEAN,
    minimum_distance_to_waters                BOOLEAN,
    volume_overlap_with_protein               BOOLEAN,
    volume_overlap_with_organic_cofactors     BOOLEAN,
    volume_overlap_with_inorganic_cofactors   BOOLEAN,
    volume_overlap_with_waters                BOOLEAN,

    -- Provenance
    posebusters_version TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS compound_boltz2_posebusters_all_passed_idx
    ON compound_boltz2_posebusters (all_passed);

CREATE INDEX IF NOT EXISTS compound_boltz2_posebusters_clash_idx
    ON compound_boltz2_posebusters (minimum_distance_to_protein);

-- OpenEye Omega 3D conformer ensemble.
--
-- SDF file paths are recorded in DB; actual conformers live on disk
-- under ``structures/omega/<compound_id>.sdf`` (gitignored).

CREATE TABLE IF NOT EXISTS compound_omega3d (
    compound_id      INTEGER PRIMARY KEY REFERENCES compounds(id),
    input_smiles     TEXT,              -- pH 7.4 protonated SMILES used as input
    sdf_path         TEXT,              -- absolute path to SDF
    n_confs          INTEGER,
    min_energy       REAL,
    max_energy       REAL,
    status           TEXT,              -- 'ok' | 'omega_failed' | 'parse_failed' | 'too_large'
    computed_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_compound_omega3d_status
    ON compound_omega3d(status);

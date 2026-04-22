-- OpenEye Quacpac pH 7.4 protonation + tautomer enumeration.

CREATE TABLE IF NOT EXISTS compound_quacpac (
    compound_id    INTEGER PRIMARY KEY REFERENCES compounds(id),
    ph74_smiles    TEXT,
    formal_charge  INTEGER,
    computed_at    TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_compound_quacpac_smiles
    ON compound_quacpac(ph74_smiles);


CREATE TABLE IF NOT EXISTS compound_tautomers (
    compound_id       INTEGER PRIMARY KEY REFERENCES compounds(id),
    input_smiles      TEXT,
    n_tautomers       INTEGER,
    tautomer_smiles   JSONB,       -- list[str] of all reasonable tautomers
    computed_at       TIMESTAMPTZ DEFAULT now()
);

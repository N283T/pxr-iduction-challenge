-- Full RDKit 2D descriptor table (all 217 descriptors from Descriptors._descList)
-- Computed via Python RDKit (not PostgreSQL cartridge, which only exposes ~40).
-- JSONB storage mirrors compound_mordred pattern for consistency.

CREATE TABLE IF NOT EXISTS compound_descriptors_full (
    compound_id integer PRIMARY KEY REFERENCES compounds(id),
    descriptors jsonb NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_descriptors_full_compound
    ON compound_descriptors_full (compound_id);

CREATE INDEX IF NOT EXISTS idx_descriptors_full_descriptors
    ON compound_descriptors_full USING GIN (descriptors);

-- Add standardized SMILES and mol columns to compounds table
-- Keep original smiles/mol for reference

ALTER TABLE compounds ADD COLUMN IF NOT EXISTS std_smiles TEXT;
ALTER TABLE compounds ADD COLUMN IF NOT EXISTS std_mol MOL;

-- Index for standardized mol
CREATE INDEX IF NOT EXISTS idx_compounds_std_mol ON compounds USING gist(std_mol);
CREATE INDEX IF NOT EXISTS idx_compounds_std_mfp2 ON compounds USING gist(morganbv_fp(std_mol));

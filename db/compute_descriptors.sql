-- Pre-compute molecular descriptors for all compounds using RDKit cartridge

-- Drop if exists for re-runnability
DROP TABLE IF EXISTS compound_descriptors;

CREATE TABLE compound_descriptors (
    compound_id INTEGER PRIMARY KEY REFERENCES compounds(id),

    -- Basic physicochemical
    amw DOUBLE PRECISION,              -- Average molecular weight
    exactmw DOUBLE PRECISION,          -- Exact molecular weight
    logp DOUBLE PRECISION,             -- Wildman-Crippen LogP
    tpsa DOUBLE PRECISION,             -- Topological polar surface area
    labuteasa DOUBLE PRECISION,        -- Labute ASA
    fractioncsp3 DOUBLE PRECISION,     -- Fraction of SP3 carbons

    -- Hydrogen bond
    hba INTEGER,                       -- H-bond acceptors
    hbd INTEGER,                       -- H-bond donors

    -- Atom/bond counts
    num_atoms INTEGER,                 -- Total atoms (incl. H)
    num_heavy_atoms INTEGER,           -- Heavy atoms
    num_heteroatoms INTEGER,           -- Heteroatoms
    num_rotatable_bonds INTEGER,       -- Rotatable bonds
    num_amide_bonds INTEGER,           -- Amide bonds

    -- Ring counts
    num_rings INTEGER,                 -- Total rings
    num_aromatic_rings INTEGER,        -- Aromatic rings
    num_aliphatic_rings INTEGER,       -- Aliphatic rings
    num_aromatic_carbocycles INTEGER,  -- Aromatic carbocycles
    num_aromatic_heterocycles INTEGER, -- Aromatic heterocycles
    num_aliphatic_carbocycles INTEGER, -- Aliphatic carbocycles
    num_aliphatic_heterocycles INTEGER,-- Aliphatic heterocycles
    num_saturated_rings INTEGER,       -- Saturated rings
    num_saturated_carbocycles INTEGER, -- Saturated carbocycles
    num_saturated_heterocycles INTEGER,-- Saturated heterocycles
    num_heterocycles INTEGER,          -- Total heterocycles
    num_spiro_atoms INTEGER,           -- Spiro atoms
    num_bridgehead_atoms INTEGER,      -- Bridgehead atoms

    -- Topological / connectivity
    chi0v DOUBLE PRECISION,
    chi1v DOUBLE PRECISION,
    chi2v DOUBLE PRECISION,
    chi3v DOUBLE PRECISION,
    chi4v DOUBLE PRECISION,
    chi0n DOUBLE PRECISION,
    chi1n DOUBLE PRECISION,
    chi2n DOUBLE PRECISION,
    chi3n DOUBLE PRECISION,
    chi4n DOUBLE PRECISION,
    kappa1 DOUBLE PRECISION,
    kappa2 DOUBLE PRECISION,
    kappa3 DOUBLE PRECISION,
    phi DOUBLE PRECISION,
    hallkieralpha DOUBLE PRECISION,

    -- Scaffold
    murcko_scaffold TEXT,

    -- Identifiers
    formula TEXT,
    inchikey TEXT
);

INSERT INTO compound_descriptors
SELECT
    c.id AS compound_id,

    -- Basic physicochemical
    mol_amw(c.mol),
    mol_exactmw(c.mol),
    mol_logp(c.mol),
    mol_tpsa(c.mol),
    mol_labuteasa(c.mol),
    mol_fractioncsp3(c.mol),

    -- Hydrogen bond
    mol_hba(c.mol),
    mol_hbd(c.mol),

    -- Atom/bond counts
    mol_numatoms(c.mol),
    mol_numheavyatoms(c.mol),
    mol_numheteroatoms(c.mol),
    mol_numrotatablebonds(c.mol),
    mol_numamidebonds(c.mol),

    -- Ring counts
    mol_numrings(c.mol),
    mol_numaromaticrings(c.mol),
    mol_numaliphaticrings(c.mol),
    mol_numaromaticcarbocycles(c.mol),
    mol_numaromaticheterocycles(c.mol),
    mol_numaliphaticcarbocycles(c.mol),
    mol_numaliphaticheterocycles(c.mol),
    mol_numsaturatedrings(c.mol),
    mol_numsaturatedcarbocycles(c.mol),
    mol_numsaturatedheterocycles(c.mol),
    mol_numheterocycles(c.mol),
    mol_numspiroatoms(c.mol),
    mol_numbridgeheadatoms(c.mol),

    -- Topological / connectivity
    mol_chi0v(c.mol),
    mol_chi1v(c.mol),
    mol_chi2v(c.mol),
    mol_chi3v(c.mol),
    mol_chi4v(c.mol),
    mol_chi0n(c.mol),
    mol_chi1n(c.mol),
    mol_chi2n(c.mol),
    mol_chi3n(c.mol),
    mol_chi4n(c.mol),
    mol_kappa1(c.mol),
    mol_kappa2(c.mol),
    mol_kappa3(c.mol),
    mol_phi(c.mol),
    mol_hallkieralpha(c.mol),

    -- Scaffold
    mol_to_smiles(mol_murckoscaffold(c.mol)),

    -- Identifiers
    mol_formula(c.mol),
    mol_inchikey(c.mol)
FROM compounds c
WHERE c.mol IS NOT NULL;

-- Index for joins
CREATE INDEX idx_descriptors_compound ON compound_descriptors(compound_id);

-- Fingerprints stored separately (binary, for similarity search)
DROP TABLE IF EXISTS compound_fingerprints;

CREATE TABLE compound_fingerprints (
    compound_id INTEGER PRIMARY KEY REFERENCES compounds(id),
    morgan_fp BFP,         -- Morgan (ECFP4, radius=2, 2048 bits)
    featmorgan_fp BFP,     -- Feature Morgan (FCFP4)
    maccs_fp BFP,          -- MACCS keys (166 bits)
    atompair_fp BFP,       -- Atom pair fingerprint
    avalon_fp BFP          -- Avalon fingerprint
);

INSERT INTO compound_fingerprints
SELECT
    c.id AS compound_id,
    morganbv_fp(c.mol) AS morgan_fp,
    featmorganbv_fp(c.mol) AS featmorgan_fp,
    maccs_fp(c.mol) AS maccs_fp,
    atompairbv_fp(c.mol) AS atompair_fp,
    avalon_fp(c.mol) AS avalon_fp
FROM compounds c
WHERE c.mol IS NOT NULL;

CREATE INDEX idx_fp_compound ON compound_fingerprints(compound_id);
CREATE INDEX idx_fp_morgan ON compound_fingerprints USING gist(morgan_fp);
CREATE INDEX idx_fp_maccs ON compound_fingerprints USING gist(maccs_fp);

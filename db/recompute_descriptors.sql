-- Recompute descriptors and fingerprints using standardized molecules (std_mol)

-- Clear and repopulate descriptors
TRUNCATE compound_descriptors;

INSERT INTO compound_descriptors
SELECT
    c.id AS compound_id,
    mol_amw(c.std_mol),
    mol_exactmw(c.std_mol),
    mol_logp(c.std_mol),
    mol_tpsa(c.std_mol),
    mol_labuteasa(c.std_mol),
    mol_fractioncsp3(c.std_mol),
    mol_hba(c.std_mol),
    mol_hbd(c.std_mol),
    mol_numatoms(c.std_mol),
    mol_numheavyatoms(c.std_mol),
    mol_numheteroatoms(c.std_mol),
    mol_numrotatablebonds(c.std_mol),
    mol_numamidebonds(c.std_mol),
    mol_numrings(c.std_mol),
    mol_numaromaticrings(c.std_mol),
    mol_numaliphaticrings(c.std_mol),
    mol_numaromaticcarbocycles(c.std_mol),
    mol_numaromaticheterocycles(c.std_mol),
    mol_numaliphaticcarbocycles(c.std_mol),
    mol_numaliphaticheterocycles(c.std_mol),
    mol_numsaturatedrings(c.std_mol),
    mol_numsaturatedcarbocycles(c.std_mol),
    mol_numsaturatedheterocycles(c.std_mol),
    mol_numheterocycles(c.std_mol),
    mol_numspiroatoms(c.std_mol),
    mol_numbridgeheadatoms(c.std_mol),
    mol_chi0v(c.std_mol),
    mol_chi1v(c.std_mol),
    mol_chi2v(c.std_mol),
    mol_chi3v(c.std_mol),
    mol_chi4v(c.std_mol),
    mol_chi0n(c.std_mol),
    mol_chi1n(c.std_mol),
    mol_chi2n(c.std_mol),
    mol_chi3n(c.std_mol),
    mol_chi4n(c.std_mol),
    mol_kappa1(c.std_mol),
    mol_kappa2(c.std_mol),
    mol_kappa3(c.std_mol),
    mol_phi(c.std_mol),
    mol_hallkieralpha(c.std_mol),
    mol_to_smiles(mol_murckoscaffold(c.std_mol)),
    mol_formula(c.std_mol),
    mol_inchikey(c.std_mol)
FROM compounds c
WHERE c.std_mol IS NOT NULL;

-- Clear and repopulate fingerprints
TRUNCATE compound_fingerprints;

INSERT INTO compound_fingerprints
SELECT
    c.id AS compound_id,
    morganbv_fp(c.std_mol) AS morgan_fp,
    featmorganbv_fp(c.std_mol) AS featmorgan_fp,
    maccs_fp(c.std_mol) AS maccs_fp,
    atompairbv_fp(c.std_mol) AS atompair_fp,
    avalon_fp(c.std_mol) AS avalon_fp
FROM compounds c
WHERE c.std_mol IS NOT NULL;

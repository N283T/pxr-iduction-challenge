-- PXR Challenge Database Schema

-- Master compound table with RDKit mol column
CREATE TABLE compounds (
    id SERIAL PRIMARY KEY,
    molecule_name TEXT NOT NULL,
    smiles TEXT NOT NULL UNIQUE,
    mol MOL GENERATED ALWAYS AS (mol_from_smiles(smiles::cstring)) STORED
);

-- GiST index for substructure/similarity search
CREATE INDEX idx_compounds_mol ON compounds USING gist(mol);
-- Fingerprint index for similarity search
CREATE INDEX idx_compounds_mfp2 ON compounds USING gist(morganbv_fp(mol));

-- Train activity data (dose-response)
CREATE TABLE train_activity (
    id SERIAL PRIMARY KEY,
    compound_id INTEGER NOT NULL REFERENCES compounds(id),
    ocnt_batch TEXT,
    pec50 DOUBLE PRECISION NOT NULL,
    pec50_ci_lower DOUBLE PRECISION,
    pec50_ci_upper DOUBLE PRECISION,
    pec50_std_error DOUBLE PRECISION,
    emax_estimate DOUBLE PRECISION,
    emax_ci_lower DOUBLE PRECISION,
    emax_ci_upper DOUBLE PRECISION,
    emax_std_error DOUBLE PRECISION,
    emax_vs_pos_ctrl DOUBLE PRECISION,
    emax_vs_pos_ctrl_ci_lower DOUBLE PRECISION,
    emax_vs_pos_ctrl_ci_upper DOUBLE PRECISION,
    emax_vs_pos_ctrl_std_error DOUBLE PRECISION
);

CREATE INDEX idx_train_activity_compound ON train_activity(compound_id);

-- Test compounds (blinded, pEC50 to predict)
CREATE TABLE test_activity (
    id SERIAL PRIMARY KEY,
    compound_id INTEGER NOT NULL REFERENCES compounds(id)
);

CREATE INDEX idx_test_activity_compound ON test_activity(compound_id);

-- Phase 1 unblinded labels for a subset of test_activity (Analog Set 1).
-- Keep separate from train_activity so Phase 1 vs Phase 2 training choices are explicit.
CREATE TABLE test_activity_phase1_labels (
    id SERIAL PRIMARY KEY,
    compound_id INTEGER NOT NULL REFERENCES compounds(id),
    phase INTEGER NOT NULL DEFAULT 1,
    ocnt_batch TEXT,
    pec50 DOUBLE PRECISION NOT NULL,
    pec50_ci_lower DOUBLE PRECISION,
    pec50_ci_upper DOUBLE PRECISION,
    pec50_std_error DOUBLE PRECISION,
    emax_estimate DOUBLE PRECISION,
    emax_ci_lower DOUBLE PRECISION,
    emax_ci_upper DOUBLE PRECISION,
    emax_std_error DOUBLE PRECISION,
    emax_vs_pos_ctrl DOUBLE PRECISION,
    emax_vs_pos_ctrl_ci_lower DOUBLE PRECISION,
    emax_vs_pos_ctrl_ci_upper DOUBLE PRECISION,
    emax_vs_pos_ctrl_std_error DOUBLE PRECISION,
    source_split TEXT,
    loaded_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (compound_id, phase)
);

CREATE INDEX idx_test_activity_phase1_labels_compound
    ON test_activity_phase1_labels(compound_id);

-- Counter-assay data (PXR-null control)
CREATE TABLE counter_assay (
    id SERIAL PRIMARY KEY,
    compound_id INTEGER NOT NULL REFERENCES compounds(id),
    ocnt_batch TEXT,
    pec50 DOUBLE PRECISION,
    pec50_ci_lower DOUBLE PRECISION,
    pec50_ci_upper DOUBLE PRECISION,
    pec50_std_error DOUBLE PRECISION,
    emax_estimate DOUBLE PRECISION,
    emax_ci_lower DOUBLE PRECISION,
    emax_ci_upper DOUBLE PRECISION,
    emax_std_error DOUBLE PRECISION,
    emax_vs_pos_ctrl DOUBLE PRECISION,
    emax_vs_pos_ctrl_ci_lower DOUBLE PRECISION,
    emax_vs_pos_ctrl_ci_upper DOUBLE PRECISION,
    emax_vs_pos_ctrl_std_error DOUBLE PRECISION
);

CREATE INDEX idx_counter_assay_compound ON counter_assay(compound_id);

-- Single-concentration screening data
CREATE TABLE single_concentration (
    id SERIAL PRIMARY KEY,
    compound_id INTEGER NOT NULL REFERENCES compounds(id),
    ocnt_batch TEXT,
    plate_id TEXT,
    compound_class TEXT,
    concentration_m DOUBLE PRECISION NOT NULL,
    log2_fc_estimate DOUBLE PRECISION,
    log2_fc_stderr DOUBLE PRECISION,
    t_statistic DOUBLE PRECISION,
    p_value DOUBLE PRECISION,
    fdr_bh DOUBLE PRECISION,
    neg_log10_fdr DOUBLE PRECISION,
    median_log2_fc DOUBLE PRECISION,
    cohens_d DOUBLE PRECISION,
    n_replicates INTEGER,
    experiment_name TEXT,
    ocnt_id TEXT
);

CREATE INDEX idx_single_conc_compound ON single_concentration(compound_id);
CREATE INDEX idx_single_conc_experiment ON single_concentration(experiment_name);

-- HTChem dose-response activity data released for Track 1 Phase 2.
CREATE TABLE htchem_activity (
    id SERIAL PRIMARY KEY,
    compound_id INTEGER NOT NULL REFERENCES compounds(id),
    source_type TEXT NOT NULL CHECK (source_type IN ('crude', 'semi_pure')),
    ocnt_id TEXT NOT NULL,
    batch_id TEXT NOT NULL,
    ec50_um DOUBLE PRECISION,
    pec50 DOUBLE PRECISION,
    emax_normalized DOUBLE PRECISION,
    emax_raw DOUBLE PRECISION,
    corrected_ec50_um DOUBLE PRECISION,
    corrected_pec50 DOUBLE PRECISION,
    drc_pec50_se DOUBLE PRECISION,
    corrected_pec50_se DOUBLE PRECISION,
    pec50_ci95 DOUBLE PRECISION,
    corrected_pec50_ci95 DOUBLE PRECISION,
    volatility TEXT,
    cad_yield_volatility_note TEXT,
    evapt_c DOUBLE PRECISION,
    theoretical_mass_on_column_ng DOUBLE PRECISION,
    peak_area_pa_min DOUBLE PRECISION,
    actual_mass_on_column_ng DOUBLE PRECISION,
    product_yield_percent DOUBLE PRECISION,
    correction_factor DOUBLE PRECISION,
    cad_peak_area_cv_percent DOUBLE PRECISION,
    cad_slope_cv_percent DOUBLE PRECISION,
    cad_yield_se_log10 DOUBLE PRECISION,
    raw_record JSONB NOT NULL,
    loaded_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (source_type, ocnt_id, batch_id)
);

CREATE INDEX idx_htchem_activity_compound ON htchem_activity(compound_id);
CREATE INDEX idx_htchem_activity_source_type ON htchem_activity(source_type);

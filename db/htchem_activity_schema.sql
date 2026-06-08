-- HTChem dose-response activity data released for Track 1 Phase 2.
--
-- Applies both crude and semi-pure HTChem releases into one parsed table.
-- Numeric columns contain coercible assay/CAD values; raw_record preserves the
-- original HF row, including nonnumeric qualifiers such as "no data" or
-- "Undetected".
--
-- Apply:
--   pixi run psql -h /tmp -p 5433 -d pxr_challenge -f db/htchem_activity_schema.sql
--
-- Idempotent: safe to re-run.

CREATE TABLE IF NOT EXISTS htchem_activity (
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

CREATE INDEX IF NOT EXISTS idx_htchem_activity_compound
    ON htchem_activity(compound_id);

CREATE INDEX IF NOT EXISTS idx_htchem_activity_source_type
    ON htchem_activity(source_type);

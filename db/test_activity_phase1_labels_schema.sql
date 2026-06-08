-- Phase 1 unblinded labels for Track 1 Activity test compounds.
--
-- These are the Analog Set 1 pEC50 labels released at the start of Phase 2.
-- Keep them separate from train_activity so Phase 1 OOF and Phase 2 retraining
-- are explicit choices in modeling code.
--
-- Apply:
--   pixi run psql -h /tmp -p 5433 -d pxr_challenge -f db/test_activity_phase1_labels_schema.sql
--
-- Idempotent: safe to re-run.

CREATE TABLE IF NOT EXISTS test_activity_phase1_labels (
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

CREATE INDEX IF NOT EXISTS idx_test_activity_phase1_labels_compound
    ON test_activity_phase1_labels(compound_id);

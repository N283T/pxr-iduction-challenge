-- Boltz-2 affinity predictions rerun from saved official-Boltz trunk embeddings.
--
-- This is intentionally separate from compound_boltz2 because the structures
-- and affinity values come from a resume/revalidation run rather than the
-- original full Boltz production pass.

CREATE TABLE IF NOT EXISTS compound_boltz2_affinity_reuse (
    compound_id INT PRIMARY KEY REFERENCES compounds(id),

    -- File paths relative to repository root.
    affinity_json_path TEXT NOT NULL,
    affinity_embeddings_npz_path TEXT NOT NULL,

    -- Boltz-2 affinity head outputs (mean + 2 ensemble members).
    affinity_pred_value           DOUBLE PRECISION NOT NULL,
    affinity_probability_binary   DOUBLE PRECISION NOT NULL,
    affinity_pred_value_1         DOUBLE PRECISION NOT NULL,
    affinity_probability_binary_1 DOUBLE PRECISION NOT NULL,
    affinity_pred_value_2         DOUBLE PRECISION NOT NULL,
    affinity_probability_binary_2 DOUBLE PRECISION NOT NULL,

    -- Final affinity-module embeddings saved after the affinity cross-attention
    -- stack. Each ensemble member is currently 384 dimensions.
    affinity_g1 DOUBLE PRECISION[] NOT NULL,
    affinity_g2 DOUBLE PRECISION[] NOT NULL,
    affinity_token_count INT NOT NULL,

    source_predictions_root TEXT NOT NULL,
    boltz_version TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS compound_boltz2_affinity_reuse_value_idx
    ON compound_boltz2_affinity_reuse (affinity_pred_value);

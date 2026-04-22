-- OpenEye FastROCS shape similarity to the training potent-active set.
--
-- Queries = 67 training compounds with pEC50 >= 6.0 (multi-conformer from Omega).
-- Targets = all 13136 compounds (best-energy conformer from Omega).
-- Output per compound: max / mean scores over queries + the most-similar query id.

CREATE TABLE IF NOT EXISTS compound_rocs (
    compound_id               INTEGER PRIMARY KEY REFERENCES compounds(id),
    -- Summary over all 67 queries
    max_shape_tanimoto        REAL,
    max_color_tanimoto        REAL,
    max_combo_tanimoto        REAL,
    mean_shape_tanimoto       REAL,
    mean_color_tanimoto       REAL,
    mean_combo_tanimoto       REAL,
    -- Identity of the best-matching query compound
    nearest_query_compound_id INTEGER,
    nearest_query_combo       REAL,
    -- Full score vector (JSONB map: {query_compound_id: [shape, color, combo]})
    all_query_scores          JSONB,
    computed_at               TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_compound_rocs_max_combo
    ON compound_rocs(max_combo_tanimoto DESC);

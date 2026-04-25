-- ChemFM-1B (TheLuoFengLab) causal-LM embeddings.
-- Stored as two poolings because ChemFM is a LlamaForCausalLM:
--   embedding_last: last non-padding token hidden state (conventional for causal LM readout)
--   embedding_mean: attention-mask-weighted mean over last hidden state (BERT-style)
-- hidden_size = 2048.

CREATE TABLE IF NOT EXISTS compound_chemfm_1b (
    compound_id      INTEGER PRIMARY KEY REFERENCES compounds(id),
    embedding_last   DOUBLE PRECISION[] NOT NULL,
    embedding_mean   DOUBLE PRECISION[] NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_compound_chemfm_1b_cid
    ON compound_chemfm_1b(compound_id);

-- Views so the existing single-column `load_embeddings()` helper in
-- run_train.py can consume each pooling as a separate feature.
CREATE OR REPLACE VIEW compound_chemfm_1b_last AS
    SELECT compound_id, embedding_last AS embedding
    FROM compound_chemfm_1b;

CREATE OR REPLACE VIEW compound_chemfm_1b_mean AS
    SELECT compound_id, embedding_mean AS embedding
    FROM compound_chemfm_1b;

-- ChemFM-3B: same Llama architecture at 3B scale.
-- hidden_size = 3072 (config verified 2026-04-25).
CREATE TABLE IF NOT EXISTS compound_chemfm_3b (
    compound_id      INTEGER PRIMARY KEY REFERENCES compounds(id),
    embedding_last   DOUBLE PRECISION[] NOT NULL,
    embedding_mean   DOUBLE PRECISION[] NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_compound_chemfm_3b_cid
    ON compound_chemfm_3b(compound_id);

CREATE OR REPLACE VIEW compound_chemfm_3b_last AS
    SELECT compound_id, embedding_last AS embedding
    FROM compound_chemfm_3b;

CREATE OR REPLACE VIEW compound_chemfm_3b_mean AS
    SELECT compound_id, embedding_mean AS embedding
    FROM compound_chemfm_3b;

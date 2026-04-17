-- Add the embeddings_npz_path column to compound_boltz2 for existing
-- databases that pre-date the trunk s/z embeddings run (issue #57). Fresh
-- clones pick the column up directly from db/boltz2_schema.sql.
--
-- Safe to run multiple times; ADD COLUMN IF NOT EXISTS is a no-op when
-- the column already exists.

ALTER TABLE compound_boltz2
    ADD COLUMN IF NOT EXISTS embeddings_npz_path TEXT;

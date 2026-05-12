-- Boltz-2 trunk pooled embeddings, unified across rcycle settings.
--
-- Populated by track1_activity/scripts/boltz_affhead/08_pool_and_upsert_fast.py
-- which pools embeddings_*.npz files (s/z trunk tensors) into a fixed
-- 1024-dim vector via the Boltz-2 paper's all-pairs interface mask
-- variant, then upserts both the existing 4,653 rcycle=3 outputs and the
-- new 8,482 rcycle=1 fast outputs into one table. The rcycle=1 rows can be
-- progressively upgraded in place with rcycle=3 resumed outputs from
-- structures/boltz2/outputs_fast_rcycle3/.
--
-- Two recycling settings coexist (recycling_steps column flags which);
-- consumers should treat the union as a single 13,135-row corpus for
-- pretrain MLP input, but may filter on recycling_steps for analysis.
--
-- See docs/superpowers/specs/2026-04-21-boltz-trunk-log2fc-pretrain-design.md
-- (Buterez strategy-3 with Boltz-2 backbone, issue #109).

CREATE TABLE IF NOT EXISTS compound_boltz2_trunk_fast (
    compound_id INT PRIMARY KEY REFERENCES compounds(id),

    -- Pooled trunk vectors (concatenation totals 1024 dims)
    s_prot_mean REAL[] NOT NULL,    -- 384d, mean over 434 PXR residue tokens
    s_lig_mean  REAL[] NOT NULL,    -- 384d, mean over ligand atom tokens
    z_if_mean   REAL[] NOT NULL,    -- 128d, mean over (PXR residues x ligand atoms) pairs
    z_if_max    REAL[] NOT NULL,    -- 128d, max  over the same pairs

    recycling_steps INT NOT NULL,   -- 1 = fast variant, 3 = high-quality main run
    source_npz_path TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS compound_boltz2_trunk_fast_rcycle_idx
    ON compound_boltz2_trunk_fast (recycling_steps);

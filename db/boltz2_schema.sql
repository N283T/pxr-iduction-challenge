-- Boltz-2 prediction outputs (one row per compound).
--
-- Populated by track2_structure/scripts/boltz2_postprocess.py after the
-- full Boltz-2 inference run completes. Failed compounds get
-- preprocessing_failed=TRUE and a failure_reason; the per-compound files
-- are then NULL.

CREATE TABLE IF NOT EXISTS compound_boltz2 (
    compound_id INT PRIMARY KEY REFERENCES compounds(id),

    -- File paths (relative to repository root, see structures/boltz2/)
    pose_cif_path        TEXT,    -- predicted complex (protein + ligand)
    ligand_pkl_path      TEXT,    -- lossless RDKit Mol pickle (preferred for downstream)
    ligand_sdf_path      TEXT,    -- viewer / external docking tool format
    confidence_json_path TEXT,
    affinity_json_path   TEXT,
    plddt_npz_path       TEXT,
    pae_npz_path         TEXT,
    pde_npz_path         TEXT,

    -- Status flags
    preprocessing_failed BOOLEAN NOT NULL DEFAULT FALSE,
    failure_reason       TEXT,
    ligand_oversize      BOOLEAN NOT NULL DEFAULT FALSE,  -- > 56 heavy atoms (Boltz-2 affinity train cap)

    -- Boltz-2 affinity head outputs (mean + 2 ensemble members)
    affinity_pred_value           DOUBLE PRECISION,  -- mean log10(IC50) μM
    affinity_probability_binary   DOUBLE PRECISION,  -- mean binder probability [0,1]
    affinity_pred_value_1         DOUBLE PRECISION,
    affinity_probability_binary_1 DOUBLE PRECISION,
    affinity_pred_value_2         DOUBLE PRECISION,
    affinity_probability_binary_2 DOUBLE PRECISION,

    -- Boltz-2 confidence metrics (from confidence_*_model_0.json)
    confidence_score DOUBLE PRECISION,
    ptm              DOUBLE PRECISION,
    iptm             DOUBLE PRECISION,
    ligand_iptm      DOUBLE PRECISION,
    protein_iptm     DOUBLE PRECISION,
    complex_plddt    DOUBLE PRECISION,
    complex_iplddt   DOUBLE PRECISION,
    complex_pde      DOUBLE PRECISION,
    complex_ipde     DOUBLE PRECISION,

    -- Geometry sanity checks
    ligand_atom_count           INT,
    ligand_to_pocket_distance_a DOUBLE PRECISION,

    -- Provenance
    boltz_version TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS compound_boltz2_status_idx
    ON compound_boltz2 (preprocessing_failed, ligand_oversize);

CREATE INDEX IF NOT EXISTS compound_boltz2_affinity_idx
    ON compound_boltz2 (affinity_pred_value)
    WHERE preprocessing_failed = FALSE;

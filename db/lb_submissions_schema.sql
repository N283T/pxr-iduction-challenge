-- Leaderboard submission tracking table.
--
-- One row per `api.py submit` call. The row is created at submit time with
-- NULL LB fields; the LB columns are back-filled by `api.py fetch` once the
-- Hugging Face Space has processed the submission (usually < 2 hours).
--
-- Purpose: keep strategy / ablation notes locally (never sent via model_tag
-- to the public leaderboard) and link each submission back to the
-- `experiments` row that produced the CSV.
--
-- Apply:
--   pixi run psql -h /tmp -p 5433 -d pxr_challenge -f db/lb_submissions_schema.sql

CREATE TABLE IF NOT EXISTS lb_submissions (
    id SERIAL PRIMARY KEY,

    -- Submission identity
    submission_name TEXT NOT NULL,      -- e.g. 'ens_l2_a0.05'
    file_path TEXT NOT NULL,            -- relative path under repo root
    experiment_name TEXT,               -- FK (soft) to experiments.name
    submitted_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    notes TEXT,                         -- LOCAL ONLY; never sent to LB
    track TEXT NOT NULL DEFAULT 'activity'
        CHECK (track IN ('activity', 'structure')),

    -- Leaderboard response (NULL until fetched)
    lb_rank INT,
    -- Track 1 (activity) metrics
    lb_mae DOUBLE PRECISION,
    lb_mae_std DOUBLE PRECISION,
    lb_rae DOUBLE PRECISION,
    lb_rae_std DOUBLE PRECISION,
    lb_r2 DOUBLE PRECISION,
    lb_r2_std DOUBLE PRECISION,
    lb_spearman DOUBLE PRECISION,
    lb_spearman_std DOUBLE PRECISION,
    lb_kendall DOUBLE PRECISION,
    lb_kendall_std DOUBLE PRECISION,
    -- Track 2 (structure) metrics
    lb_lddt_pli DOUBLE PRECISION,
    lb_lddt_pli_std DOUBLE PRECISION,
    lb_bisyrmsd DOUBLE PRECISION,
    lb_bisyrmsd_std DOUBLE PRECISION,
    lb_lddt_lp DOUBLE PRECISION,
    lb_lddt_lp_std DOUBLE PRECISION,
    lb_coverage DOUBLE PRECISION,
    lb_submitted_utc TEXT,              -- matches LB CSV 'Submitted' column
    lb_fetched_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_lb_submissions_name
    ON lb_submissions(submission_name);
CREATE INDEX IF NOT EXISTS idx_lb_submissions_submitted_at
    ON lb_submissions(submitted_at DESC);
CREATE INDEX IF NOT EXISTS idx_lb_submissions_pending_fetch
    ON lb_submissions(lb_fetched_at) WHERE lb_fetched_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_lb_submissions_track
    ON lb_submissions(track, submitted_at DESC);

-- Convenient read view: most-recent submissions with LB status. Track 1 (activity)
-- and Track 2 (structure) metric columns are exposed side-by-side; only one set
-- is populated per row depending on `track`.
DROP VIEW IF EXISTS lb_submission_history;
CREATE VIEW lb_submission_history AS
SELECT
    id,
    track,
    submission_name,
    experiment_name,
    submitted_at,
    lb_fetched_at,
    lb_rank,
    -- Track 1 (activity) metrics
    lb_rae,
    lb_rae_std,
    lb_mae,
    lb_r2,
    lb_spearman,
    -- Track 2 (structure) metrics
    lb_lddt_pli,
    lb_lddt_pli_std,
    lb_bisyrmsd,
    lb_bisyrmsd_std,
    lb_lddt_lp,
    lb_coverage,
    notes
FROM lb_submissions
ORDER BY submitted_at DESC;

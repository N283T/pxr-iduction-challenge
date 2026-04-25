-- Migration: extend lb_submissions to support Track 2 (Structure Prediction).
--
-- Adds:
--   - `track` column to distinguish Track 1 (activity) from Track 2 (structure).
--     Existing rows are all activity submissions, so DEFAULT 'activity' is safe.
--   - Track 2 leaderboard metric columns (LDDT-PLI / BiSyRMSD / LDDT-LP / Coverage).
--
-- Apply:
--   pixi run psql -h /tmp -p 5433 -d pxr_challenge -f db/lb_submissions_track2_migration.sql
--
-- Idempotent: every column add uses IF NOT EXISTS, so re-running is safe.

ALTER TABLE lb_submissions
    ADD COLUMN IF NOT EXISTS track TEXT NOT NULL DEFAULT 'activity'
        CHECK (track IN ('activity', 'structure'));

ALTER TABLE lb_submissions ADD COLUMN IF NOT EXISTS lb_lddt_pli      DOUBLE PRECISION;
ALTER TABLE lb_submissions ADD COLUMN IF NOT EXISTS lb_lddt_pli_std  DOUBLE PRECISION;
ALTER TABLE lb_submissions ADD COLUMN IF NOT EXISTS lb_bisyrmsd      DOUBLE PRECISION;
ALTER TABLE lb_submissions ADD COLUMN IF NOT EXISTS lb_bisyrmsd_std  DOUBLE PRECISION;
ALTER TABLE lb_submissions ADD COLUMN IF NOT EXISTS lb_lddt_lp       DOUBLE PRECISION;
ALTER TABLE lb_submissions ADD COLUMN IF NOT EXISTS lb_lddt_lp_std   DOUBLE PRECISION;
ALTER TABLE lb_submissions ADD COLUMN IF NOT EXISTS lb_coverage      DOUBLE PRECISION;

CREATE INDEX IF NOT EXISTS idx_lb_submissions_track
    ON lb_submissions(track, submitted_at DESC);

-- Refresh the read view to expose the new columns. CREATE OR REPLACE cannot
-- reorder/rename columns of an existing view, so we drop first.
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

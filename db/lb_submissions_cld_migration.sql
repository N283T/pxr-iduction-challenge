-- Migration: add Compact Letter Display (CLD) storage for Activity LB rows.
--
-- The Hugging Face leaderboard added a `CLD` column in Phase 2. It is a text
-- compact-letter display string such as `b,d,i,j,m,n`.
--
-- Apply:
--   pixi run psql -h /tmp -p 5433 -d pxr_challenge -f db/lb_submissions_cld_migration.sql
--
-- Idempotent: safe to re-run.

ALTER TABLE lb_submissions ADD COLUMN IF NOT EXISTS lb_cld TEXT;

-- Refresh the read view to expose the new column. CREATE OR REPLACE cannot
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
    lb_cld,
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

# Leaderboard Snapshots

This directory stores public leaderboard CSV snapshots used for retrospective
Track 1 and Track 2 analysis.

## Layout

| Directory | Contents |
|---|---|
| `activity/` | Track 1 activity leaderboard snapshots named `leaderboard_<date>.csv`. Use `leaderboard_<date>_<time>.csv` if preserving multiple same-day fetches. |
| `structure/` | Track 2 structure leaderboard snapshots named `leaderboard_structure_<date>.csv`. |

Keep filenames date-stamped and append-only. Do not overwrite an older snapshot
with a newer fetch from the same day unless the older file was clearly corrupt.

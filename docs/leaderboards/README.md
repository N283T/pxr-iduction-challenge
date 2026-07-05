# Leaderboard Snapshots

This directory stores public leaderboard CSV snapshots used for retrospective
Track 1 analysis.

## Layout

| Directory | Contents |
|---|---|
| `activity/` | Track 1 activity leaderboard snapshots named `leaderboard_<date>_<time>JST.csv`. |

`api.py fetch` writes timestamped files here by default. Keep filenames
append-only; do not overwrite an older snapshot unless the older file was
clearly corrupt.

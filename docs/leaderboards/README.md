# Leaderboard Snapshots

This directory stores public leaderboard CSV snapshots used for retrospective
Track 1 and Track 2 analysis.

## Layout

| Directory | Contents |
|---|---|
| `activity/` | Track 1 activity leaderboard snapshots named `leaderboard_<date>_<time>JST.csv`. |
| `structure/` | Track 2 structure leaderboard snapshots named `leaderboard_structure_<date>_<time>JST.csv`. |

`api.py fetch` writes timestamped files here by default. Keep filenames
append-only; do not overwrite an older snapshot unless the older file was
clearly corrupt.

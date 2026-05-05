# Track 2 Documents

This directory stores Track 2 structure-prediction documents and small scoring
tables. Large generated structures, input YAMLs, submissions, and raw Boltz-2
outputs stay under `structures/` or `track2_structure/` and are gitignored.

## Layout

| Path | Contents |
|---|---|
| `submission_spec.md` | Track 2 submission-format notes. |
| `track2_holo_ligand_db.csv` | Holo-template ligand database summary. |
| `track2_template_rmsd_scores.csv` | Candidate pose scores against holo templates. |
| `track2_ftmap_hotspot_scores.csv` | Candidate pose scores against FTMap hotspots. |
| `track2_posebusters_all.csv` | PoseBusters validity checks for Boltz model candidates. |
| `track2_redock_*_scores.csv` | Redocking/template-transfer score summaries. |
| `model_selection/` | Per-submission pose-selection logs. |

The historical `docs/track2_compound_grids/` image directory is generated and
ignored. If those grids are regenerated, prefer `docs/track2/compound_grids/`.

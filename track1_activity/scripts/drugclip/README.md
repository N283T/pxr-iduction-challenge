# DrugCLIP Axis

Status: external-data feature axis; not part of no-external-data submissions.

This directory contains the DrugCLIP molecule encoding path used to test whether
DrugCLIP embeddings add a useful orthogonal signal. The generated artifacts live
under `data/` and are intentionally not tracked.

## Scripts

| Script | Purpose |
|---|---|
| `01_smiles_to_lmdb.py` | Convert repository compound SMILES into the LMDB format expected by the external DrugCLIP codebase. |
| `02_encode_drugclip.sh` | Run the DrugCLIP molecule encoder from the external checkout and write embeddings under `data/drugclip_embed/`. |

## Cleanup Stance

Keep this directory as an external-data reference. Do not use outputs from this
axis in a submission that is intended to be no-external-data. If this axis is
reopened, record the external-data decision explicitly in the submission notes.

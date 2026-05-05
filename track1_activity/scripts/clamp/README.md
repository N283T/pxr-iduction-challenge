# CLAMP Axis

Status: closed negative result for the current Track 1 pool.

This directory contains the CLAMP raw encoder extraction and leak-check scripts
from PR #159. The axis failed the first gate by a large margin, but the scripts
are useful provenance for why CLAMP was not added to the production ensemble.

## Scripts

| Script | Purpose |
|---|---|
| `01_extract_clamp_embed.py` | Extract CLAMP molecule embeddings for challenge compounds into `data/clamp_embed.parquet`. |
| `02_leak_check.py` | Probe whether CLAMP has PXR-specific prior knowledge from assay-text prompts. |

## Cleanup Stance

Keep these scripts as a compact closed-axis record. Do not move them back into
the top-level workflow unless CLAMP is reopened with a new prompt, model
checkpoint, or leakage policy.

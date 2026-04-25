"""Constants for the Track 2 (Structure Prediction) Boltz-2 pipeline.

Track 2 deliberately uses the LBD-only PXR sequence (293 aa, residues 142-434
of UniProt O75469) to match the official PXR-Challenge-Tutorial reference
exactly, rather than the full-length 434-aa sequence used by the Track 1
pipeline. The LBD is the only domain relevant to ligand binding; OST scoring
operates on protein-ligand contacts only, so excluding the disordered N-terminal
DBD/hinge gives equivalent scoring at lower compute cost.
"""

from __future__ import annotations

from pathlib import Path

# PXR Ligand-Binding Domain (residues 142-434, 1-indexed, UniProt O75469).
# Identical to the FASTA shipped with the official tutorial repo.
PXR_LBD_SEQUENCE = (
    "GLTEEQRMMIRELMDAQMKTFDTTFSHFKNFRLPGVLSSGCELPESLQAPSREEAAKWSQVRKDLCSLKVS"
    "LQLRGEDGSVWNYKPPADSGGKEIFSLLPHMADMSTYMFKGIISFAKVISYFRDLPIEDQISLLKGAAFEL"
    "CQLRFNTVFNAETGTWECGRLSYCLEDTAGGFQQLLLEPMLKFHYMLKKLQLHEEEYVLMQAISLFSPDRP"
    "GVLQHRVVDQLQEQFAITLKSYIECNRPQPAHRFLFLKIMAMLTELRSINAQHTQRLLRIQDIHPFATPLM"
    "QELFGITGS"
)
assert len(PXR_LBD_SEQUENCE) == 293, f"expected 293 aa LBD, got {len(PXR_LBD_SEQUENCE)}"

PROTEIN_CHAIN_ID = "A"
LIGAND_CHAIN_ID = "B"

# Repo-relative paths (resolved at import time so call sites can use them
# directly without recomputing).
REPO_ROOT = Path(__file__).resolve().parents[3]
TRACK2_INPUT_DIR = REPO_ROOT.joinpath("track2_structure", "inputs")
TRACK2_OUTPUT_DIR = REPO_ROOT.joinpath("structures", "boltz2_track2", "outputs")
LOG_DIR = REPO_ROOT.joinpath("logs")

# Cached PXR LBD MSA, produced by run_apo.sh (which copies the
# ColabFold-generated CSV out of the apo run output dir to this stable
# location). Holo YAMLs reference this path so the 184 production runs
# share a single preprocessing step instead of hammering ColabFold 184x.
MSA_PATH = REPO_ROOT.joinpath("structures", "boltz2_track2", "msa", "pxr_lbd.csv")

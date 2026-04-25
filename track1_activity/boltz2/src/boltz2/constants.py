"""Constants for the Boltz-2 PXR pipeline.

All paths, the PXR full-length sequence, pocket residue numbering, and chain
id conventions are centralised here so that input building, post-processing,
and analysis scripts agree on a single source of truth.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Filesystem layout
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]
STRUCTURES_DIR = PROJECT_ROOT.joinpath("structures")
BOLTZ2_DIR = STRUCTURES_DIR.joinpath("boltz2")

MSA_DIR = BOLTZ2_DIR.joinpath("msa")
MSA_PATH = MSA_DIR.joinpath("pxr.a3m")

INPUTS_DIR = BOLTZ2_DIR.joinpath("inputs")
OUTPUTS_DIR = BOLTZ2_DIR.joinpath("outputs")

INPUTS_SMOKE_DIR = BOLTZ2_DIR.joinpath("inputs_smoke")
OUTPUTS_SMOKE_DIR = BOLTZ2_DIR.joinpath("outputs_smoke")

MANIFEST_PATH = BOLTZ2_DIR.joinpath("manifest.csv")
MANIFEST_SMOKE_PATH = BOLTZ2_DIR.joinpath("manifest_smoke.csv")


# ---------------------------------------------------------------------------
# PXR target (UniProt O75469, full-length 434 aa)
# Source: AFDB AF-O75469-F1-msa_v6.a3m (query line)
# ---------------------------------------------------------------------------

PXR_SEQUENCE = (
    "MEVRPKESWNHADFVHCEDTESVPGKPSVNADEEVGGPQICRVCGDKATGYHFNVMTCEGCKGFFRRAMKR"
    "NARLRCPFRKGACEITRKTRRQCQACRLRKCLESGMKKEMIMSDEAVEERRALIKRKKSERTGTQPLGVQG"
    "LTEEQRMMIRELMDAQMKTFDTTFSHFKNFRLPGVLSSGCELPESLQAPSREEAAKWSQVRKDLCSLKVSL"
    "QLRGEDGSVWNYKPPADSGGKEIFSLLPHMADMSTYMFKGIISFAKVISYFRDLPIEDQISLLKGAAFELC"
    "QLRFNTVFNAETGTWECGRLSYCLEDTAGGFQQLLLEPMLKFHYMLKKLQLHEEEYVLMQAISLFSPDRPG"
    "VLQHRVVDQLQEQFAITLKSYIECNRPQPAHRFLFLKIMAMLTELRSINAQHTQRLLRIQDIHPFATPLMQ"
    "ELFGITGS"
)
assert len(PXR_SEQUENCE) == 434, (
    f"PXR full-length sequence should be 434 residues, got {len(PXR_SEQUENCE)}"
)


# ---------------------------------------------------------------------------
# Binding pocket
# Source: structures/pxr_pocket_residues.json (issue #13)
# Numbering follows UniProt O75469.
# ---------------------------------------------------------------------------

# Core pocket: residues contacted in >=50/72 holo PDB structures.
PXR_CORE_POCKET_RESIDUES: tuple[int, ...] = (
    209,
    211,
    240,
    243,
    247,
    281,
    285,
    288,
    299,
    306,
    323,
    407,
    411,
)

# Soft constraint distance for the Boltz pocket directive (Angstroms).
POCKET_MAX_DISTANCE: float = 6.0


# ---------------------------------------------------------------------------
# Chain id conventions
# ---------------------------------------------------------------------------

PROTEIN_CHAIN_ID = "A"
LIGAND_CHAIN_ID = "B"

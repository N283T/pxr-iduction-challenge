"""Fingerprint and feature generation for Track 1."""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, rdFingerprintGenerator
from rdkit.Avalon import pyAvalonTools


def smiles_to_mols(smiles_series: pd.Series) -> list:
    """Convert SMILES to RDKit mol objects."""
    return [Chem.MolFromSmiles(s) for s in smiles_series]


def morgan_fp(mols: list, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """Morgan (ECFP) fingerprint."""
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    return np.array([gen.GetFingerprintAsNumPy(m) for m in mols], dtype=np.uint8)


def morgan_fp_r3(mols: list, n_bits: int = 2048) -> np.ndarray:
    """Morgan fingerprint with radius=3 (ECFP6)."""
    return morgan_fp(mols, radius=3, n_bits=n_bits)


def feat_morgan_fp(mols: list, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """Feature Morgan (FCFP) fingerprint."""
    gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=n_bits,
        atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen(),
    )
    return np.array([gen.GetFingerprintAsNumPy(m) for m in mols], dtype=np.uint8)


def maccs_fp(mols: list) -> np.ndarray:
    """MACCS keys (167 bits)."""
    return np.array([np.array(MACCSkeys.GenMACCSKeys(m), dtype=np.uint8) for m in mols])


def atompair_fp(mols: list, n_bits: int = 2048) -> np.ndarray:
    """Atom pair fingerprint."""
    gen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=n_bits)
    return np.array([gen.GetFingerprintAsNumPy(m) for m in mols], dtype=np.uint8)


def topological_torsion_fp(mols: list, n_bits: int = 2048) -> np.ndarray:
    """Topological torsion fingerprint."""
    gen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=n_bits)
    return np.array([gen.GetFingerprintAsNumPy(m) for m in mols], dtype=np.uint8)


def rdkit_fp(mols: list, n_bits: int = 2048) -> np.ndarray:
    """RDKit fingerprint (daylight-like)."""
    gen = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=n_bits)
    return np.array([gen.GetFingerprintAsNumPy(m) for m in mols], dtype=np.uint8)


def avalon_fp(mols: list, n_bits: int = 2048) -> np.ndarray:
    """Avalon fingerprint."""
    return np.array(
        [
            np.array(pyAvalonTools.GetAvalonFP(m, nBits=n_bits), dtype=np.uint8)
            for m in mols
        ]
    )


def count_morgan_fp(mols: list, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """Count-based Morgan fingerprint."""
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    return np.array([gen.GetCountFingerprintAsNumPy(m) for m in mols], dtype=np.float32)


def count_atompair_fp(mols: list, n_bits: int = 2048) -> np.ndarray:
    """Count-based Atom pair fingerprint."""
    gen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=n_bits)
    return np.array([gen.GetCountFingerprintAsNumPy(m) for m in mols], dtype=np.float32)


def count_rdkit_fp(mols: list, n_bits: int = 2048) -> np.ndarray:
    """Count-based RDKit fingerprint."""
    gen = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=n_bits)
    return np.array([gen.GetCountFingerprintAsNumPy(m) for m in mols], dtype=np.float32)


# Registry for easy iteration
FP_REGISTRY = {
    "morgan_r2_2048": lambda mols: morgan_fp(mols, radius=2, n_bits=2048),
    "morgan_r3_2048": lambda mols: morgan_fp(mols, radius=3, n_bits=2048),
    "feat_morgan_r2_2048": lambda mols: feat_morgan_fp(mols, radius=2, n_bits=2048),
    "maccs": maccs_fp,
    "atompair_2048": lambda mols: atompair_fp(mols, n_bits=2048),
    "topo_torsion_2048": lambda mols: topological_torsion_fp(mols, n_bits=2048),
    "rdkit_fp_2048": lambda mols: rdkit_fp(mols, n_bits=2048),
    "avalon_2048": lambda mols: avalon_fp(mols, n_bits=2048),
    "count_morgan_r2_2048": lambda mols: count_morgan_fp(mols, radius=2, n_bits=2048),
    "count_morgan_r3_2048": lambda mols: count_morgan_fp(mols, radius=3, n_bits=2048),
    "count_atompair_2048": lambda mols: count_atompair_fp(mols, n_bits=2048),
    "count_rdkit_fp_2048": lambda mols: count_rdkit_fp(mols, n_bits=2048),
}

"""Importance ratio (density ratio) computation for covariate shift correction.

Shared between:
  - run_ensemble_calibrate_importance.py (post-hoc affine calibration)
  - run_ensemble.py (IW-caruana selection, --importance-weighted-mae)

Pipeline:
  1. Train Morgan-FP + LogisticRegression classifier (train=0, test=1).
  2. Compute density ratio w(x) = p(test|x) / (1 - p(test|x)) * (n_train / n_test).
  3. Clip to [1/3, 3] to prevent extreme weights destabilising downstream fits.
  4. Normalize so sum(w) == n_train (readability only, doesn't change fit point).

The clip + normalize convention matches the 2026-04-21 production recipe
(lb_submissions notes for id=19) and was the rank-1 LB winner of id=31
(2026-04-25, post seed5ens double-swap).
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.linear_model import LogisticRegression


def morgan_matrix(
    smiles_list: list[str], radius: int = 2, n_bits: int = 2048
) -> np.ndarray:
    gen = AllChem.GetMorganGenerator(radius=radius, fpSize=n_bits)
    out = np.zeros((len(smiles_list), n_bits), dtype=np.uint8)
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = gen.GetFingerprint(mol)
        out[i] = np.asarray(fp, dtype=np.uint8)
    return out


def compute_importance_weights(
    train_smiles: list[str],
    test_smiles: list[str],
    *,
    radius: int = 2,
    n_bits: int = 2048,
    clip_lo: float = 1.0 / 3.0,
    clip_hi: float = 3.0,
    seed: int = 42,
    eps: float = 1e-6,
) -> np.ndarray:
    """Per-train-compound importance weight w(x).

    Returns
    -------
    np.ndarray of shape (len(train_smiles),), values in [clip_lo, clip_hi],
    normalized so sum(w) == len(train_smiles).
    """
    X_train = morgan_matrix(train_smiles, radius=radius, n_bits=n_bits)
    X_test = morgan_matrix(test_smiles, radius=radius, n_bits=n_bits)

    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate(
        [
            np.zeros(len(X_train), dtype=np.int32),
            np.ones(len(X_test), dtype=np.int32),
        ]
    )
    clf = LogisticRegression(
        max_iter=1000, solver="liblinear", C=1.0, random_state=seed
    )
    clf.fit(X_all, y_all)
    p_test_given_x = clf.predict_proba(X_train)[:, 1]
    w = (p_test_given_x + eps) / (1.0 - p_test_given_x + eps)
    w = w * (len(X_train) / len(X_test))
    w = np.clip(w, clip_lo, clip_hi)
    w = w * (len(w) / w.sum())
    return w

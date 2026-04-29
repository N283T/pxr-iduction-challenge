"""Proximity-gated affine calibrator for the caruana_bag20 ensemble.

Local version of run_ensemble_calibrate_importance.py:
  1. Compute NN-Tanimoto from each train/test compound to the potent-46 set
     (train pec50 >= 6 AND selectivity = pec50 - counter_pec50 >= 1.5).
  2. Split train and test into "near" (NN >= T) and "far" (NN < T) strata,
     where T = median(NN-Tanimoto over test).
  3. Per stratum, fit a LogReg train-vs-test domain classifier (Morgan FP)
     and a weighted LinearRegression on (oof_pred, y_train) using the
     density-ratio sample weights (clipped [1/3, 3]).
  4. Apply each stratum's affine to its test slice, write submission.

Output: track1_activity/submissions/ens_caruana_bag20_calibrated_proximity.csv

LB A/B mandatory regardless of OOF gate (calibrator family changes have a
documented OOF/LB sign-flip history).

Spec: docs/superpowers/specs/2026-04-29-potent46-proximity-gated-calibrator-design.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.stats import spearmanr  # noqa: F401  (used in main() — Task 4)
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
)

REPO_ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
from data import (  # noqa: E402, F401  (load_* used in main() — Task 4)
    DB_PARAMS,
    load_test_smiles,
    load_train_smiles_target,
    load_train_smiles_with_counter,
)

POTENT_PEC50_THRESHOLD = 6.0
POTENT_SEL_THRESHOLD = 1.5
WEIGHT_CLIP_LO = 1.0 / 3.0
WEIGHT_CLIP_HI = 3.0
MIN_STRATUM_TRAIN = 200


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


def load_potent46_indices() -> np.ndarray:
    """Indices into the train_df row order for potent-46 members.

    potent-46 = train pec50 >= 6 AND selectivity >= 1.5
    selectivity = train_pec50 - counter_pec50 (NaN where counter row missing,
    those rows are excluded).
    """
    df = load_train_smiles_with_counter()
    sel = df["pec50"] - df["counter_pec50"]
    mask = (df["pec50"] >= POTENT_PEC50_THRESHOLD) & (sel >= POTENT_SEL_THRESHOLD)
    indices = np.flatnonzero(mask.to_numpy())
    return indices


def load_caruana_oof_and_test() -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    import json

    with psycopg2.connect(**DB_PARAMS) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, hyperparameters FROM experiments
             WHERE name = 'ens_caruana_bag20'
             ORDER BY id DESC LIMIT 1
            """
        )
        exp_id, hp = cur.fetchone()
        weights_map = hp["weights"]
        print(f"  using ens_caruana_bag20 experiment id={exp_id}")
        print(f"  {len(weights_map)} member weights in hyperparameters")

        oof_stack = None
        for name, weight in weights_map.items():
            cur.execute(
                "SELECT id FROM experiments WHERE name = %s ORDER BY id DESC LIMIT 1",
                (name,),
            )
            row = cur.fetchone()
            if row is None:
                raise RuntimeError(f"member experiment missing: {name}")
            mid = row[0]
            cur.execute(
                """
                SELECT train_idx, oof_prediction
                  FROM experiment_oof_predictions
                 WHERE experiment_id = %s
                 ORDER BY train_idx
                """,
                (mid,),
            )
            rows = cur.fetchall()
            if not rows:
                raise RuntimeError(f"OOF missing for member {name} (id={mid})")
            member = np.asarray([r[1] for r in rows], dtype=np.float64)
            if oof_stack is None:
                oof_stack = np.zeros_like(member)
            oof_stack = oof_stack + weight * member

    oof_preds = oof_stack / sum(weights_map.values())

    sub_path = REPO_ROOT.joinpath(
        "track1_activity", "submissions", "ens_caruana_bag20.csv"
    )
    test_sub = pd.read_csv(sub_path)
    test_col = [c for c in test_sub.columns if c.lower() == "pec50"][0]
    test_preds = test_sub[test_col].to_numpy(dtype=np.float64)
    print(
        f"  reconstructed OOF: n={len(oof_preds)}, "
        f"mean={oof_preds.mean():.4f}, std={oof_preds.std():.4f}"
    )
    _ = json
    return oof_preds, test_preds, test_sub


def compute_nn_tanimoto(
    query_fps: np.ndarray,
    anchor_fps: np.ndarray,
    query_global_idx: np.ndarray | None = None,
    anchor_global_idx: np.ndarray | None = None,
) -> np.ndarray:
    """Max Tanimoto similarity from each query row to any anchor row.

    Both inputs are uint8 (N, n_bits) bit matrices (bits in 0/1).
    Tanimoto = popcount(A AND B) / popcount(A OR B).

    If `query_global_idx` and `anchor_global_idx` are provided, any anchor
    whose global index matches the query's global index is excluded from
    that query's max (self-exclude). Pass None on either side to disable.
    """
    if anchor_fps.shape[0] == 0:
        raise ValueError("anchor_fps is empty")

    # popcount per row, precomputed
    q_pop = query_fps.sum(axis=1).astype(np.int32)  # (Nq,)
    a_pop = anchor_fps.sum(axis=1).astype(np.int32)  # (Na,)

    # Intersection: matrix product of bit matrices (each cell = popcount(q AND a))
    inter = query_fps.astype(np.int32) @ anchor_fps.T.astype(np.int32)  # (Nq, Na)
    union = q_pop[:, None] + a_pop[None, :] - inter  # (Nq, Na)
    # Avoid div-by-zero (both all-zero is impossible for valid mols, but guard)
    tanimoto = np.where(union > 0, inter / np.maximum(union, 1), 0.0)

    # Self-exclude: set tanimoto[i, j] = -inf where query_global_idx[i] == anchor_global_idx[j]
    if query_global_idx is not None and anchor_global_idx is not None:
        # broadcast comparison (Nq, Na)
        mask = query_global_idx[:, None] == anchor_global_idx[None, :]
        tanimoto = np.where(mask, -np.inf, tanimoto)

    nn = tanimoto.max(axis=1)
    # If self-exclude removed every anchor for some query (shouldn't happen
    # unless anchors are a subset of size 1 == query), guard the -inf:
    if np.any(np.isneginf(nn)):
        raise RuntimeError(
            "compute_nn_tanimoto: query has no anchor after self-exclude"
        )
    return nn.astype(np.float64)


def fit_stratum_calibrator(
    train_fps_stratum: np.ndarray,
    test_fps_stratum: np.ndarray,
    oof_stratum: np.ndarray,
    y_stratum: np.ndarray,
    label: str,
) -> tuple[float, float, np.ndarray]:
    """Fit per-stratum domain classifier + weighted affine.

    Returns: (slope, intercept, sample_weights). Prints a one-line summary.
    """
    n_train = len(train_fps_stratum)
    n_test = len(test_fps_stratum)
    if n_train < 2:
        raise RuntimeError(f"stratum '{label}': only {n_train} train rows, cannot fit")
    if n_test < 1:
        raise RuntimeError(f"stratum '{label}': zero test rows")

    # Domain classifier (within stratum only)
    X_all = np.vstack([train_fps_stratum, test_fps_stratum])
    y_all = np.concatenate(
        [np.zeros(n_train, dtype=np.int32), np.ones(n_test, dtype=np.int32)]
    )
    clf = LogisticRegression(max_iter=1000, solver="liblinear", C=1.0, random_state=42)
    clf.fit(X_all, y_all)
    p_test = clf.predict_proba(train_fps_stratum)[:, 1]
    eps = 1e-6
    w = (p_test + eps) / (1.0 - p_test + eps)
    w = w * (n_train / n_test)
    w = np.clip(w, WEIGHT_CLIP_LO, WEIGHT_CLIP_HI)
    w = w * (n_train / w.sum())

    reg = LinearRegression()
    reg.fit(oof_stratum.reshape(-1, 1), y_stratum, sample_weight=w)
    slope = float(reg.coef_[0])
    intercept = float(reg.intercept_)
    print(
        f"  stratum '{label}': n_train={n_train} n_test={n_test} "
        f"w[mean={w.mean():.3f} q25={np.quantile(w, 0.25):.3f} "
        f"q75={np.quantile(w, 0.75):.3f} max={w.max():.3f}]"
        f" -> y = {slope:.4f} * pred + {intercept:.4f}"
    )
    return slope, intercept, w

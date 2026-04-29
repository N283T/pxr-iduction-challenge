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
from sklearn.linear_model import (  # noqa: F401  (used in main() — Task 4)
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

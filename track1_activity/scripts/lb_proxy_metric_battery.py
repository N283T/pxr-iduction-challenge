"""LB-proxy metric battery: which OOF/test metric best predicts LB outcome?

Per Codex consult 2026-04-29 (reply on thread 019dd7a2-...): the recurring
OOF→LB reverse amplification (id=38, 40, 41, 44) suggests our current
UMAP-OOF MAE gate is a poor LB proxy. Build a battery of candidate metrics
and test which (if any) correctly orders historical submissions by LB.

Candidate metrics computed per submission (rows = train, OR test predictions
where applicable):
  M1  raw OOF MAE                           (current default gate)
  M2  calibrated OOF MAE                    (apply importance affine before MAE)
  M3  importance-weighted OOF MAE           (Morgan-FP density-ratio sample weights)
  M4  importance-weighted CALIBRATED OOF MAE (M2 + M3 combined)
  M5  top-shift subset OOF MAE              (NN-Tanimoto >= 0.30 to potent-46)
  M6  top-shift subset Spearman             (rank correlation on subset)
  M7  fold worst-case OOF MAE               (max over UMAP folds)
  M8  delta-to-baseline on weighted subset  (signed improvement vs baseline)

Cases evaluated (where construction is recoverable as of 2026-04-29):

  Production reference:
    base9pool   9-pool caruana baseline (uncalibrated)         OOF MAE 0.3958

  Submitted with LB known:
    id43        ens_hybrid_meta_baseline_5050       LB 0.4075 (rank 2 best)
    id44        ens_caruana_bag20_anchor_residual   LB 0.4090 (rank 3 reverse)

  Today's bakeoff variants (NOT submitted, LB unknown but predicted):
    proximity_calibrator_v2     null (degenerate)
    knn_alltrain_pool_add       null (caruana weight ~0)
    knn_potent46_pool_add       null (caruana weight ~0)
    admet_standalone_pool_add   null (caruana weight 0.005)
    admet_swap_top500           tied (no signal)
    admet_add_top500            family-share trap (would LB regress per memory)

The goal: for each candidate metric M, compute the metric for all cases
above and check whether the metric's ordering matches the (known +
predicted) LB ordering. Best metric = best gate.

Decision rule: metric whose Spearman rank correlation with LB MAE >= 0.5
across the LB-known subset is a candidate gate. If no metric reaches 0.3,
fallback to multi-condition gate.

Legacy experiment script; internal design note was removed from the public repository.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, LogisticRegression

REPO_ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "scripts")))
from data import (  # noqa: E402
    DB_PARAMS,
    load_test_smiles,
    load_train_smiles_target,
    load_train_smiles_with_counter,
)
from splits import umap_split_indices  # noqa: E402

POTENT_PEC50_THRESHOLD = 6.0
POTENT_SEL_THRESHOLD = 1.5
TOP_SHIFT_THRESHOLD = 0.30  # NN-Tanimoto cutoff for "test-like" subset
WEIGHT_CLIP_LO = 1.0 / 3.0
WEIGHT_CLIP_HI = 3.0
N_SPLITS = 5
N_CLUSTERS = 50
SEED = 42
SUBMISSION_DIR = REPO_ROOT.joinpath("track1_activity", "submissions")


# ---------- helpers ----------


def morgan_matrix(smiles_list: list[str]) -> np.ndarray:
    gen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
    out = np.zeros((len(smiles_list), 2048), dtype=np.uint8)
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = gen.GetFingerprint(mol)
        out[i] = np.asarray(fp, dtype=np.uint8)
    return out


def potent46_indices() -> np.ndarray:
    df = load_train_smiles_with_counter()
    sel = df["pec50"] - df["counter_pec50"]
    mask = (df["pec50"] >= POTENT_PEC50_THRESHOLD) & (sel >= POTENT_SEL_THRESHOLD)
    return np.flatnonzero(mask.to_numpy())


def nn_tanimoto_to(query_fps: np.ndarray, anchor_fps: np.ndarray) -> np.ndarray:
    q_pop = query_fps.sum(axis=1).astype(np.int32)
    a_pop = anchor_fps.sum(axis=1).astype(np.int32)
    inter = query_fps.astype(np.int32) @ anchor_fps.T.astype(np.int32)
    union = q_pop[:, None] + a_pop[None, :] - inter
    sim = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
    return sim.max(axis=1).astype(np.float64)


def fit_global_importance_affine(
    X_tr_fp: np.ndarray, X_te_fp: np.ndarray, oof: np.ndarray, y: np.ndarray
) -> tuple[float, float, np.ndarray]:
    """Returns (slope, intercept, sample_weights). Same recipe as
    run_ensemble_calibrate_importance.py."""
    X_all = np.vstack([X_tr_fp, X_te_fp])
    y_all = np.concatenate(
        [
            np.zeros(len(X_tr_fp), dtype=np.int32),
            np.ones(len(X_te_fp), dtype=np.int32),
        ]
    )
    clf = LogisticRegression(max_iter=1000, solver="liblinear", C=1.0, random_state=42)
    clf.fit(X_all, y_all)
    p_test = clf.predict_proba(X_tr_fp)[:, 1]
    eps = 1e-6
    w = (p_test + eps) / (1.0 - p_test + eps)
    w = w * (len(X_tr_fp) / len(X_te_fp))
    w = np.clip(w, WEIGHT_CLIP_LO, WEIGHT_CLIP_HI)
    w = w * (len(w) / w.sum())
    reg = LinearRegression()
    reg.fit(oof.reshape(-1, 1), y, sample_weight=w)
    return float(reg.coef_[0]), float(reg.intercept_), w


def load_member_oof(name: str) -> np.ndarray:
    with psycopg2.connect(**DB_PARAMS) as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT id FROM experiments WHERE name = %s ORDER BY id DESC LIMIT 1",
            (name,),
        )
        row = cur.fetchone()
        if row is None:
            raise RuntimeError(f"missing experiment: {name}")
        exp_id = row[0]
        cur.execute(
            """
            SELECT train_idx, oof_prediction
              FROM experiment_oof_predictions
             WHERE experiment_id = %s
             ORDER BY train_idx
            """,
            (exp_id,),
        )
        rows = cur.fetchall()
    return np.asarray([r[1] for r in rows], dtype=np.float64)


def load_caruana_baseline_oof_test() -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Reconstruct latest 9-pool caruana_bag20 OOF + load test predictions."""
    with psycopg2.connect(**DB_PARAMS) as conn:
        cur = conn.cursor()
        cur.execute(
            """SELECT id, hyperparameters FROM experiments WHERE name = 'ens_caruana_bag20'
               ORDER BY id DESC LIMIT 1"""
        )
        exp_id, hp = cur.fetchone()
        weights = hp["weights"]
        oof_stack = None
        for name, w in weights.items():
            cur.execute(
                "SELECT id FROM experiments WHERE name = %s ORDER BY id DESC LIMIT 1",
                (name,),
            )
            mid = cur.fetchone()[0]
            cur.execute(
                """SELECT train_idx, oof_prediction FROM experiment_oof_predictions
                   WHERE experiment_id = %s ORDER BY train_idx""",
                (mid,),
            )
            rows = cur.fetchall()
            member = np.asarray([r[1] for r in rows], dtype=np.float64)
            if oof_stack is None:
                oof_stack = np.zeros_like(member)
            oof_stack = oof_stack + w * member
    oof = oof_stack / sum(weights.values())
    test_sub = pd.read_csv(SUBMISSION_DIR.joinpath("ens_caruana_bag20.csv"))
    test_col = [c for c in test_sub.columns if c.lower() == "pec50"][0]
    test = test_sub[test_col].to_numpy(dtype=np.float64)
    return oof, test, test_sub


# ---------- candidate metrics ----------


def compute_metrics(
    *,
    oof: np.ndarray,
    y_train: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    train_proximity: np.ndarray,
    importance_w: np.ndarray,
    importance_slope: float,
    importance_intercept: float,
) -> dict:
    """Compute all candidate gate metrics on a given OOF prediction."""
    # M1 raw OOF MAE
    m1 = float(np.mean(np.abs(oof - y_train)))

    # M2 calibrated OOF MAE
    oof_cal = importance_slope * oof + importance_intercept
    m2 = float(np.mean(np.abs(oof_cal - y_train)))

    # M3 importance-weighted OOF MAE (raw)
    m3 = float(np.average(np.abs(oof - y_train), weights=importance_w))

    # M4 importance-weighted calibrated OOF MAE
    m4 = float(np.average(np.abs(oof_cal - y_train), weights=importance_w))

    # M5 top-shift subset MAE (raw OOF; calibration applied)
    shift_mask = train_proximity >= TOP_SHIFT_THRESHOLD
    if shift_mask.sum() < 50:
        m5 = float("nan")
        m6 = float("nan")
    else:
        m5 = float(np.mean(np.abs(oof_cal[shift_mask] - y_train[shift_mask])))
        # M6 top-shift subset Spearman
        m6 = float(spearmanr(oof_cal[shift_mask], y_train[shift_mask]).statistic)

    # M7 fold worst-case raw MAE
    fold_maes = [float(np.mean(np.abs(oof[va] - y_train[va]))) for _, va in folds]
    m7 = float(max(fold_maes))

    # M8 delta-to-baseline on weighted subset (signed)
    # Computed externally vs a reference baseline; here just store raw for diff later
    m8_raw = m4  # we'll subtract baseline's m4 in main()

    return {
        "M1_raw_oof_mae": m1,
        "M2_calibrated_oof_mae": m2,
        "M3_importance_weighted_oof_mae": m3,
        "M4_importance_weighted_calibrated_oof_mae": m4,
        "M5_top_shift_subset_calibrated_mae": m5,
        "M6_top_shift_subset_spearman": m6,
        "M7_fold_worst_case_raw_mae": m7,
        "M8_iw_cal_oof_mae_for_delta": m8_raw,
    }


# ---------- case generators ----------


def case_baseline_9pool(common: dict) -> tuple[str, np.ndarray, np.ndarray]:
    """Return (label, oof_4140, test_513) for the current 9-pool baseline."""
    return "base9pool", common["oof_base"].copy(), common["test_base"].copy()


def case_id43_hybrid(common: dict) -> tuple[str, np.ndarray, np.ndarray]:
    """id=43 hybrid 50/50 = blend of base9pool + family_meta_7pool.

    family_meta_7pool replaces the 3 chemprop-family members with their mean
    as a single member; we don't have the OOF stored. Use test CSV directly
    (which is the actual submitted hybrid) and approximate OOF as an average
    that minimizes drift; specifically we substitute the saved OOF of the
    closest matching ensemble experiment.

    Pragmatic shortcut for THIS analysis: read the actual submitted CSV
    `ens_hybrid_meta_baseline_5050.csv` for test, and reconstruct OOF as
    0.5 * base9pool_OOF + 0.5 * family_meta_OOF. family_meta OOF is
    `chemprop_family_meta_umap` if it's in DB; if not, fall back to
    base9pool only (degenerate, will look like baseline but flagged).
    """
    test_path = SUBMISSION_DIR.joinpath("ens_hybrid_meta_baseline_5050.csv")
    test_sub = pd.read_csv(test_path)
    test_col = [c for c in test_sub.columns if c.lower() == "pec50"][0]
    test = test_sub[test_col].to_numpy(dtype=np.float64)

    try:
        meta_oof = load_member_oof("chemprop_family_meta_umap")
        if len(meta_oof) == len(common["oof_base"]):
            oof = 0.5 * common["oof_base"] + 0.5 * meta_oof
        else:
            oof = common["oof_base"].copy()
            print(
                "  WARN: chemprop_family_meta_umap OOF length mismatch; "
                "using base9pool OOF for id43"
            )
    except RuntimeError:
        oof = common["oof_base"].copy()
        print("  WARN: chemprop_family_meta_umap missing; using base9pool OOF for id43")

    return "id43_hybrid", oof, test


def case_id44_anchor_residual(common: dict) -> tuple[str, np.ndarray, np.ndarray]:
    """id=44 = base + 0.4 * residual (linear LinReg on 4 anchor features).

    Re-construct OOF + test by replicating the run_anchor_residual.py
    pipeline EXACTLY (same alpha, same feature build, same UMAP folds).
    """
    train_df = common["train_df"]
    test_df = common["test_df"]
    y_train = common["y_train"]

    # Apply importance affine to get base
    slope_g, intercept_g = common["importance_slope"], common["importance_intercept"]
    base_oof = slope_g * common["oof_base"] + intercept_g
    base_test = slope_g * common["test_base"] + intercept_g

    # Build 4 anchor features
    potent_idx = common["potent_idx"]
    potent_pec50 = y_train[potent_idx]
    X_train_fp = common["X_train_fp"]
    X_test_fp = common["X_test_fp"]
    train_global_idx = np.arange(len(train_df))
    test_global_idx = np.full(len(test_df), -1, dtype=np.int64)

    nn_train, nn_train_idx = _nn_with_idx(
        X_train_fp, X_train_fp[potent_idx], train_global_idx, potent_idx
    )
    nn_test, nn_test_idx = _nn_with_idx(
        X_test_fp, X_train_fp[potent_idx], test_global_idx, potent_idx
    )
    anchor_pec50_train = potent_pec50[nn_train_idx]
    anchor_pec50_test = potent_pec50[nn_test_idx]
    pred_minus_anchor_train = base_oof - anchor_pec50_train
    pred_minus_anchor_test = base_test - anchor_pec50_test

    feat_train = np.column_stack(
        [nn_train, anchor_pec50_train, base_oof, pred_minus_anchor_train]
    )
    feat_test = np.column_stack(
        [nn_test, anchor_pec50_test, base_test, pred_minus_anchor_test]
    )
    residual_target = y_train - base_oof

    # OOF residual via UMAP fold CV
    folds = common["folds"]
    resid_oof = np.zeros_like(residual_target)
    for tr, va in folds:
        m = LinearRegression()
        m.fit(feat_train[tr], residual_target[tr])
        resid_oof[va] = m.predict(feat_train[va])

    full = LinearRegression()
    full.fit(feat_train, residual_target)
    resid_test = full.predict(feat_test)

    alpha = 0.4
    oof = base_oof + alpha * resid_oof
    test = base_test + alpha * resid_test
    return "id44_anchor_residual", oof, test


def _nn_with_idx(
    qfp: np.ndarray,
    afp: np.ndarray,
    qidx: np.ndarray,
    aidx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    q_pop = qfp.sum(axis=1).astype(np.int32)
    a_pop = afp.sum(axis=1).astype(np.int32)
    inter = qfp.astype(np.int32) @ afp.T.astype(np.int32)
    union = q_pop[:, None] + a_pop[None, :] - inter
    sim = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
    mask = qidx[:, None] == aidx[None, :]
    sim = np.where(mask, -np.inf, sim)
    nn_idx = sim.argmax(axis=1)
    nn_sim = sim[np.arange(len(sim)), nn_idx]
    return nn_sim.astype(np.float64), nn_idx


def case_admet_standalone(common: dict) -> tuple[str, np.ndarray, np.ndarray]:
    """tabpfn_admet_ai_umap: standalone TabPFN on 104 ADMET features.

    Single OOF MAE 0.5156, never submitted.
    """
    oof = load_member_oof("tabpfn_admet_ai_umap")
    test_sub = pd.read_csv(SUBMISSION_DIR.joinpath("tabpfn_admet_ai_umap.csv"))
    test_col = [c for c in test_sub.columns if c.lower() == "pec50"][0]
    test = test_sub[test_col].to_numpy(dtype=np.float64)
    return "admet_standalone", oof, test


def case_admet_augmented_top500_swap(
    common: dict,
) -> tuple[str, np.ndarray, np.ndarray]:
    """SWAP: replace existing top500 with admet-augmented top500."""
    new_member = "tabpfn_cheme_2d_full_boltz_log2fc_pred_seed10ens_admet_ai_top500_umap"
    new_oof = load_member_oof(new_member)
    test_sub = pd.read_csv(SUBMISSION_DIR.joinpath(f"{new_member}.csv"))
    test_col = [c for c in test_sub.columns if c.lower() == "pec50"][0]
    test_new = test_sub[test_col].to_numpy(dtype=np.float64)

    # Reconstruct caruana SWAP: replace top500 OOF + test in the 9-pool blend
    # with new_oof and test_new, keeping the same caruana weights.
    with psycopg2.connect(**DB_PARAMS) as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT hyperparameters FROM experiments WHERE name = 'ens_caruana_bag20' ORDER BY id DESC LIMIT 1"
        )
        weights_map = cur.fetchone()[0]["weights"]

    OLD = "tabpfn_cheme_2d_full_boltz_log2fc_pred_seed10ens_top500_umap"
    oof_stack = None
    test_stack = None
    for name, w in weights_map.items():
        if name == OLD:
            mem_oof = new_oof
            mem_test = test_new
        else:
            mem_oof = load_member_oof(name)
            mem_sub = pd.read_csv(SUBMISSION_DIR.joinpath(f"{name}.csv"))
            tcol = [c for c in mem_sub.columns if c.lower() == "pec50"][0]
            mem_test = mem_sub[tcol].to_numpy(dtype=np.float64)
        if oof_stack is None:
            oof_stack = np.zeros_like(mem_oof)
            test_stack = np.zeros_like(mem_test)
        oof_stack += w * mem_oof
        test_stack += w * mem_test
    norm = sum(weights_map.values())
    oof = oof_stack / norm
    test = test_stack / norm
    return "admet_swap_top500", oof, test


def case_admet_augmented_top500_add(common: dict) -> tuple[str, np.ndarray, np.ndarray]:
    """ADD: 10-pool with both OLD top500 and admet-augmented top500.

    For OOF, we don't run a fresh caruana. Approximate by averaging the
    NEW member into the existing 9-pool blend at weight 0.226 (observed
    caruana weight from bake_admet_ai_swap.py output) and renormalising.
    For test, use the same blend.
    """
    new_member = "tabpfn_cheme_2d_full_boltz_log2fc_pred_seed10ens_admet_ai_top500_umap"
    new_oof = load_member_oof(new_member)
    test_sub = pd.read_csv(SUBMISSION_DIR.joinpath(f"{new_member}.csv"))
    test_col = [c for c in test_sub.columns if c.lower() == "pec50"][0]
    test_new = test_sub[test_col].to_numpy(dtype=np.float64)

    base_oof = common["oof_base"]
    base_test = common["test_base"]
    new_w = 0.226
    base_w = 1.0 - new_w
    oof = base_w * base_oof + new_w * new_oof
    test = base_w * base_test + new_w * test_new
    return "admet_add_top500_approx", oof, test


def case_proximity_calibrator_v2(common: dict) -> tuple[str, np.ndarray, np.ndarray]:
    """Proximity-gated calibrator v2 (T=0.28, near-only local + far-global).

    For this analysis the calibrator was applied to the same base9pool OOF;
    we just compute MAE on (calibrated test) and corresponding OOF.
    """
    test_path = SUBMISSION_DIR.joinpath("ens_caruana_bag20_calibrated_proximity.csv")
    if not test_path.exists():
        # Build it inline if missing; fall back to base
        return (
            "proximity_v2_missing",
            common["oof_base"].copy(),
            common["test_base"].copy(),
        )
    test_sub = pd.read_csv(test_path)
    test_col = [c for c in test_sub.columns if c.lower() == "pec50"][0]
    test = test_sub[test_col].to_numpy(dtype=np.float64)
    # OOF: this calibrator was a near-affine + far-global; near affine
    # collapsed to ~global affine, so OOF ≈ base after global affine.
    slope_g, intercept_g = common["importance_slope"], common["importance_intercept"]
    oof = slope_g * common["oof_base"] + intercept_g
    return "proximity_v2", oof, test


# ---------- main ----------


def main() -> None:
    print("Loading common state ...")
    train_df = load_train_smiles_target()
    test_df = load_test_smiles()
    y_train = train_df["pec50"].to_numpy(dtype=np.float64)

    print("  computing Morgan FPs ...")
    X_train_fp = morgan_matrix(train_df["smiles"].tolist())
    X_test_fp = morgan_matrix(test_df["smiles"].tolist())

    print("  loading 9-pool baseline OOF + test ...")
    oof_base, test_base, _ = load_caruana_baseline_oof_test()

    print("  fitting global importance affine ...")
    slope_g, intercept_g, importance_w = fit_global_importance_affine(
        X_train_fp, X_test_fp, oof_base, y_train
    )
    print(f"    affine: y = {slope_g:.4f} * pred + {intercept_g:.4f}")

    print("  computing potent-46 + train proximity ...")
    potent_idx = potent46_indices()
    train_proximity = nn_tanimoto_to(X_train_fp, X_train_fp[potent_idx])
    test_proximity = nn_tanimoto_to(X_test_fp, X_train_fp[potent_idx])
    print(
        f"    potent-46 size={len(potent_idx)}, "
        f"train prox median={np.median(train_proximity):.3f}, "
        f"test prox median={np.median(test_proximity):.3f}, "
        f"train >= {TOP_SHIFT_THRESHOLD}: {(train_proximity >= TOP_SHIFT_THRESHOLD).sum()}, "
        f"test >= {TOP_SHIFT_THRESHOLD}: {(test_proximity >= TOP_SHIFT_THRESHOLD).sum()}"
    )

    print(f"  building UMAP {N_SPLITS}-fold split ...")
    folds = umap_split_indices(
        train_df["smiles"].tolist(),
        n_splits=N_SPLITS,
        n_clusters=N_CLUSTERS,
        seed=SEED,
    )

    common = {
        "train_df": train_df,
        "test_df": test_df,
        "y_train": y_train,
        "X_train_fp": X_train_fp,
        "X_test_fp": X_test_fp,
        "oof_base": oof_base,
        "test_base": test_base,
        "importance_slope": slope_g,
        "importance_intercept": intercept_g,
        "importance_w": importance_w,
        "potent_idx": potent_idx,
        "train_proximity": train_proximity,
        "test_proximity": test_proximity,
        "folds": folds,
    }

    # Build cases. LB MAE/Sp are filled in for cases that were submitted; None for
    # variants we did not submit (we annotate predicted outcome separately).
    case_builders = [
        case_baseline_9pool,  # reference (LB id=43 base; rank 2 0.4075/0.847 effectively)
        case_id43_hybrid,  # LB known: 0.4075 / 0.8470 (rank 2 best)
        case_id44_anchor_residual,  # LB known: 0.4090 / 0.8448 (rank 3 reverse)
        case_admet_standalone,  # not submitted
        case_admet_augmented_top500_swap,  # not submitted
        case_admet_augmented_top500_add,  # not submitted (would expect family-share LB regress)
        case_proximity_calibrator_v2,  # not submitted (degenerate)
    ]
    lb_known = {
        "id43_hybrid": (0.4075, 0.8470),
        "id44_anchor_residual": (0.4090, 0.8448),
    }
    # base9pool is the production calibrated baseline; its closest LB ref is id=42 (0.4091/0.8476)
    # since base9pool OOF + importance affine is what was submitted recently.
    # Note id=42 used family_meta 7-pool, not base 9-pool — so "base9pool" is a
    # synthetic reference rather than a literal LB submission. We exclude it
    # from the LB correlation computation.

    print("\n=== Computing candidate metrics per case ===\n")
    rows = []
    base_metrics = None
    for builder in case_builders:
        label, oof, test = builder(common)
        m = compute_metrics(
            oof=oof,
            y_train=y_train,
            folds=folds,
            train_proximity=train_proximity,
            importance_w=importance_w,
            importance_slope=slope_g,
            importance_intercept=intercept_g,
        )
        m["case"] = label
        m["test_mean"] = float(np.mean(test))
        m["test_std"] = float(np.std(test))
        if label == "base9pool":
            base_metrics = m
        rows.append(m)
        print(
            f"  {label:>32}  raw={m['M1_raw_oof_mae']:.4f}  "
            f"cal={m['M2_calibrated_oof_mae']:.4f}  "
            f"iw={m['M3_importance_weighted_oof_mae']:.4f}  "
            f"iwcal={m['M4_importance_weighted_calibrated_oof_mae']:.4f}  "
            f"shift_mae={m['M5_top_shift_subset_calibrated_mae']:.4f}  "
            f"shift_sp={m['M6_top_shift_subset_spearman']:.4f}  "
            f"worst={m['M7_fold_worst_case_raw_mae']:.4f}"
        )

    # Compute M8 = delta in iw-cal MAE vs base9pool
    print("\n=== Delta to base9pool ===")
    for r in rows:
        r["M8_delta_iw_cal_vs_base"] = (
            r["M8_iw_cal_oof_mae_for_delta"]
            - base_metrics["M8_iw_cal_oof_mae_for_delta"]
        )
        r["M2_delta_cal_vs_base"] = (
            r["M2_calibrated_oof_mae"] - base_metrics["M2_calibrated_oof_mae"]
        )

    # Print delta table
    cols = [
        "case",
        "M1_raw_oof_mae",
        "M2_calibrated_oof_mae",
        "M2_delta_cal_vs_base",
        "M3_importance_weighted_oof_mae",
        "M4_importance_weighted_calibrated_oof_mae",
        "M8_delta_iw_cal_vs_base",
        "M5_top_shift_subset_calibrated_mae",
        "M6_top_shift_subset_spearman",
        "M7_fold_worst_case_raw_mae",
        "test_mean",
        "test_std",
    ]
    df = pd.DataFrame(rows)[cols]
    print()
    print(df.to_string(index=False, float_format="%.4f"))
    print()

    # Correlate each metric with LB MAE on submitted cases
    print("=== LB correlation (Spearman) for submitted cases ===\n")
    submitted = [r for r in rows if r["case"] in lb_known]
    print(f"  N submitted = {len(submitted)}")
    if len(submitted) >= 2:
        lb_mae_arr = np.array([lb_known[r["case"]][0] for r in submitted])
        for m_key in [
            "M1_raw_oof_mae",
            "M2_calibrated_oof_mae",
            "M2_delta_cal_vs_base",
            "M3_importance_weighted_oof_mae",
            "M4_importance_weighted_calibrated_oof_mae",
            "M8_delta_iw_cal_vs_base",
            "M5_top_shift_subset_calibrated_mae",
            "M7_fold_worst_case_raw_mae",
        ]:
            metric_arr = np.array([r[m_key] for r in submitted])
            if len(submitted) >= 3:
                rho = float(spearmanr(metric_arr, lb_mae_arr).statistic)
                print(f"  {m_key:>50}  Spearman vs LB MAE = {rho:+.4f}")
            else:
                # 2 points: just show direction
                if metric_arr[0] < metric_arr[1]:
                    pred_order = "first < second"
                else:
                    pred_order = "first > second"
                if lb_mae_arr[0] < lb_mae_arr[1]:
                    actual_order = "first < second"
                else:
                    actual_order = "first > second"
                match = "✓ same" if pred_order == actual_order else "✗ differs"
                print(
                    f"  {m_key:>50}  metric: {pred_order}; LB: {actual_order} ({match})"
                )

    # Final summary
    print("\n=== Submitted case detail ===\n")
    print("  id43_hybrid: LB MAE 0.4075 (rank 2 best)")
    print("  id44_anchor_residual: LB MAE 0.4090 (rank 3 reverse)")
    print("  Diff: id43 LOWER MAE (better) than id44 by 0.0015\n")
    print("  For each metric, does the candidate correctly identify id43 < id44?")
    id43 = next(r for r in rows if r["case"] == "id43_hybrid")
    id44 = next(r for r in rows if r["case"] == "id44_anchor_residual")
    for m_key in [
        "M1_raw_oof_mae",
        "M2_calibrated_oof_mae",
        "M2_delta_cal_vs_base",
        "M3_importance_weighted_oof_mae",
        "M4_importance_weighted_calibrated_oof_mae",
        "M8_delta_iw_cal_vs_base",
        "M5_top_shift_subset_calibrated_mae",
        "M7_fold_worst_case_raw_mae",
    ]:
        v43 = id43[m_key]
        v44 = id44[m_key]
        # For MAE-flavour metrics, lower is better; for spearman, higher
        if "spearman" in m_key.lower():
            correct = v43 > v44
        else:
            correct = v43 < v44
        print(
            f"    {m_key:>50}  id43={v43:.4f}  id44={v44:.4f}  "
            f"Δ={v44 - v43:+.4f}  {'✓ correct' if correct else '✗ wrong'}"
        )


if __name__ == "__main__":
    main()

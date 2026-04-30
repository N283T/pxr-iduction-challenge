"""LB-proxy metric battery v2: stronger adversarial classifier + 4 LB-known cases.

Extends 2026-04-29 lb_proxy_metric_battery.py with:

  - All 4 LB-known submissions: id42 (family_meta_7pool), id43 (hybrid_5050),
    id44 (anchor_residual), id45 (admet_no_log2fc).
  - Stronger adversarial classifier: LightGBM on Morgan FP (vs prior LogReg).
  - Test-likeness top-quartile MAE/Sp: hard top-25% slice without weight clip.
  - Spearman correlation across the 4 LB-known cases (N=4 enables real ordering
    test, not just pairwise direction).

Per Codex consult 2026-04-30: validation proxy redesign is highest-EV. The
existing battery showed M2 (calibrated OOF MAE) correctly distinguishes
id43 vs id44 but only 2 LB-known cases were available. With 4 cases now and
a stronger classifier, we can:

  1. Verify which metric ordering best matches LB ordering.
  2. Quantify improvement from LightGBM vs LogReg classifier.
  3. Test whether top-quartile slicing (hard) beats sample weighting (soft).

Output: docs/superpowers/runs/2026-04-30-lb-proxy-battery-v2.log
"""

from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import psycopg2
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "scripts")))
from data import (  # noqa: E402
    DB_PARAMS,
    load_test_smiles,
    load_train_smiles_target,
)
from splits import umap_split_indices  # noqa: E402

WEIGHT_CLIP_LO = 1.0 / 3.0
WEIGHT_CLIP_HI = 3.0
N_SPLITS = 5
N_CLUSTERS = 50
SEED = 42
SUBMISSION_DIR = REPO_ROOT.joinpath("track1_activity", "submissions")

# Known LB outcomes (lb_submissions.id -> (LB MAE, LB Sp, LB rank))
LB_KNOWN = {
    "id42_family_meta": (0.4091, 0.8476, 3),
    "id43_hybrid": (0.4075, 0.8470, 2),
    "id44_anchor_residual": (0.4090, 0.8448, 3),
    "id45_admet_no_log2fc": (0.4094, 0.8429, 2),
}

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


def fit_classifier_logreg(
    X_tr: np.ndarray, X_te: np.ndarray, *, seed: int = 42
) -> tuple[np.ndarray, float]:
    """Returns (p_test_given_x_train, AUC_on_5fold_CV)."""
    X_all = np.vstack([X_tr, X_te])
    y_all = np.concatenate(
        [np.zeros(len(X_tr), dtype=np.int32), np.ones(len(X_te), dtype=np.int32)]
    )
    clf = LogisticRegression(
        max_iter=1000, solver="liblinear", C=1.0, random_state=seed
    )
    # 5-fold CV AUC (out-of-fold)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y_all))
    folds = np.array_split(perm, 5)
    p_oof = np.zeros(len(y_all), dtype=np.float64)
    for k in range(5):
        va = folds[k]
        tr = np.concatenate([folds[j] for j in range(5) if j != k])
        m = LogisticRegression(
            max_iter=1000, solver="liblinear", C=1.0, random_state=seed
        )
        m.fit(X_all[tr], y_all[tr])
        p_oof[va] = m.predict_proba(X_all[va])[:, 1]
    auc = float(roc_auc_score(y_all, p_oof))
    # Re-train full for prediction on train
    clf.fit(X_all, y_all)
    p_train = clf.predict_proba(X_tr)[:, 1]
    return p_train, auc


def fit_classifier_lgbm(
    X_tr: np.ndarray, X_te: np.ndarray, *, seed: int = 42
) -> tuple[np.ndarray, float]:
    """Returns (p_test_given_x_train, AUC_on_5fold_CV). Stronger classifier."""
    X_all = np.vstack([X_tr, X_te]).astype(np.float32)
    y_all = np.concatenate(
        [np.zeros(len(X_tr), dtype=np.int32), np.ones(len(X_te), dtype=np.int32)]
    )
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y_all))
    folds = np.array_split(perm, 5)
    p_oof = np.zeros(len(y_all), dtype=np.float64)
    params = dict(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        reg_lambda=1.0,
        random_state=seed,
        verbose=-1,
    )
    for k in range(5):
        va = folds[k]
        tr = np.concatenate([folds[j] for j in range(5) if j != k])
        m = lgb.LGBMClassifier(**params)
        m.fit(X_all[tr], y_all[tr])
        p_oof[va] = m.predict_proba(X_all[va])[:, 1]
    auc = float(roc_auc_score(y_all, p_oof))
    full = lgb.LGBMClassifier(**params)
    full.fit(X_all, y_all)
    p_train = full.predict_proba(X_tr)[:, 1]
    return p_train, auc


def importance_weights_from_p(
    p_train: np.ndarray, n_train: int, n_test: int, *, eps: float = 1e-6
) -> np.ndarray:
    """Convert p(test|x) into normalised, clipped importance weights."""
    w = (p_train + eps) / (1.0 - p_train + eps)
    w = w * (n_train / n_test)
    w = np.clip(w, WEIGHT_CLIP_LO, WEIGHT_CLIP_HI)
    w = w * (len(w) / w.sum())
    return w


def fit_global_affine(
    oof: np.ndarray, y: np.ndarray, w: np.ndarray
) -> tuple[float, float]:
    reg = LinearRegression()
    reg.fit(oof.reshape(-1, 1), y, sample_weight=w)
    return float(reg.coef_[0]), float(reg.intercept_)


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


def load_caruana_baseline_oof_test() -> tuple[np.ndarray, np.ndarray, dict]:
    """Reconstruct latest 9-pool caruana_bag20 OOF + load weights map."""
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
    norm = sum(weights.values())
    oof = oof_stack / norm
    test_sub = pd.read_csv(SUBMISSION_DIR.joinpath("ens_caruana_bag20.csv"))
    test_col = [c for c in test_sub.columns if c.lower() == "pec50"][0]
    test = test_sub[test_col].to_numpy(dtype=np.float64)
    return oof, test, weights


def load_test_csv(name: str) -> np.ndarray:
    sub = pd.read_csv(SUBMISSION_DIR.joinpath(name))
    col = [c for c in sub.columns if c.lower() == "pec50"][0]
    return sub[col].to_numpy(dtype=np.float64)


# ---------- case construction ----------


def build_id42_oof(common: dict) -> np.ndarray:
    """family_meta 7-pool OOF (chemprop_family_meta replaces 3 chemprop members)."""
    family_meta_oof = load_member_oof("tabpfn_chemprop_family_meta_umap")
    # 7-pool weights from issue #100 (id=42)
    weights = {
        "tabpfn_chemprop_family_meta_umap": 0.539,
        "tabpfn_kermt_pretrain_embed_umap_default": 0.148,
        "tabpfn_attentivefp_pretrain_embed_umap_default": 0.076,
        "tabpfn_gatedgcn_pretrain_embed_umap_default": 0.067,
        "tabpfn_pooled_boltz_allpairs_umap_default": 0.065,
        "tabpfn_pooled_boltz_umap_default": 0.056,
        "tabpfn_molformer_c3_pretrain_embed_umap": 0.049,
    }
    oof_stack = np.zeros_like(family_meta_oof)
    for name, w in weights.items():
        if name == "tabpfn_chemprop_family_meta_umap":
            mem = family_meta_oof
        else:
            mem = load_member_oof(name)
        oof_stack += w * mem
    return oof_stack / sum(weights.values())


def build_id43_oof(common: dict) -> np.ndarray:
    """id=43 hybrid 50/50 = 0.5 * base9pool + 0.5 * family_meta_7pool (OOF)."""
    base = common["oof_base"]
    meta = build_id42_oof(common)
    return 0.5 * base + 0.5 * meta


def build_id44_oof(common: dict) -> np.ndarray:
    """id=44 = base + 0.4 * residual on 4 anchor features (OOF via UMAP CV)."""
    train_df = common["train_df"]
    y_train = common["y_train"]
    slope_g, intercept_g = common["importance_slope"], common["importance_intercept"]
    base_oof = slope_g * common["oof_base"] + intercept_g

    potent_idx = common["potent_idx"]
    potent_pec50 = y_train[potent_idx]
    X_train_fp = common["X_train_fp"]
    train_global_idx = np.arange(len(train_df))

    nn_train, nn_train_idx = _nn_with_idx(
        X_train_fp, X_train_fp[potent_idx], train_global_idx, potent_idx
    )
    anchor_pec50_train = potent_pec50[nn_train_idx]
    pred_minus_anchor_train = base_oof - anchor_pec50_train

    feat_train = np.column_stack(
        [nn_train, anchor_pec50_train, base_oof, pred_minus_anchor_train]
    )
    residual_target = y_train - base_oof
    folds = common["folds"]
    resid_oof = np.zeros_like(residual_target)
    for tr, va in folds:
        m = LinearRegression()
        m.fit(feat_train[tr], residual_target[tr])
        resid_oof[va] = m.predict(feat_train[va])
    return base_oof + 0.4 * resid_oof


def build_id45_oof(common: dict) -> np.ndarray:
    """id=45 admet_no_log2fc = caruana ADD over base9pool with admet_ai member.

    Approximation: blend base9pool with the admet_ai standalone member at the
    observed caruana weight (~0.005 from feedback memory).
    Acknowledged limitation: this is an approximation; the actual id=45 used a
    re-bagged caruana over 10-pool (9 base + admet). The key effect (small new
    weight on a low-signal member) is preserved.
    """
    base = common["oof_base"]
    admet_oof = load_member_oof("tabpfn_admet_ai_umap")
    if len(admet_oof) != len(base):
        return base.copy()
    # Re-bagged caruana would shift weights slightly; without the exact run,
    # use a near-base blend with admet at small weight.
    new_w = 0.05  # rough approximation; LB result is in family-share trap zone
    return (1.0 - new_w) * base + new_w * admet_oof


def _nn_with_idx(
    qfp: np.ndarray, afp: np.ndarray, qidx: np.ndarray, aidx: np.ndarray
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


# ---------- metrics ----------


def compute_metrics(
    *,
    oof: np.ndarray,
    y_train: np.ndarray,
    importance_w_lr: np.ndarray,
    importance_w_lgb: np.ndarray,
    p_train_lr: np.ndarray,
    p_train_lgb: np.ndarray,
    importance_slope_lr: float,
    importance_intercept_lr: float,
) -> dict:
    """Compute metrics M1-M14 on OOF predictions.

    M1-M4: existing (raw, calibrated, IW raw, IW calibrated).
    M9-M11: LightGBM-classifier variants (stronger discrimination).
    M12-M13: top-quartile (hard 25% slice by classifier prob).
    M14: bottom-quartile (least test-like) MAE for control.
    """
    # M1 raw
    m1 = float(np.mean(np.abs(oof - y_train)))
    # M2 calibrated (LogReg-based affine)
    oof_cal_lr = importance_slope_lr * oof + importance_intercept_lr
    m2 = float(np.mean(np.abs(oof_cal_lr - y_train)))
    # M3 IW raw (LogReg)
    m3 = float(np.average(np.abs(oof - y_train), weights=importance_w_lr))
    # M4 IW calibrated (LogReg)
    m4 = float(np.average(np.abs(oof_cal_lr - y_train), weights=importance_w_lr))

    # LightGBM variants
    m9 = float(np.average(np.abs(oof - y_train), weights=importance_w_lgb))
    m10 = float(np.average(np.abs(oof_cal_lr - y_train), weights=importance_w_lgb))

    # Top-quartile by LightGBM p_train
    q75 = np.quantile(p_train_lgb, 0.75)
    top_mask = p_train_lgb >= q75
    m12_mae = float(np.mean(np.abs(oof_cal_lr[top_mask] - y_train[top_mask])))
    m13_sp = float(spearmanr(oof_cal_lr[top_mask], y_train[top_mask]).statistic)

    # Bottom-quartile (control: should be FAR from LB outcome)
    q25 = np.quantile(p_train_lgb, 0.25)
    bot_mask = p_train_lgb <= q25
    m14_mae = float(np.mean(np.abs(oof_cal_lr[bot_mask] - y_train[bot_mask])))

    return {
        "M1_raw": m1,
        "M2_cal_lr": m2,
        "M3_iw_raw_lr": m3,
        "M4_iw_cal_lr": m4,
        "M9_iw_raw_lgb": m9,
        "M10_iw_cal_lgb": m10,
        "M12_top25_cal_mae": m12_mae,
        "M13_top25_cal_sp": m13_sp,
        "M14_bot25_cal_mae": m14_mae,
    }


# ---------- main ----------


def main() -> None:
    print("Loading common state ...")
    train_df = load_train_smiles_target()
    test_df = load_test_smiles()
    y_train = train_df["pec50"].to_numpy(dtype=np.float64)

    print("  computing Morgan FPs ...")
    X_train_fp = morgan_matrix(train_df["smiles"].tolist())
    X_test_fp = morgan_matrix(test_df["smiles"].tolist())
    n_tr, n_te = len(X_train_fp), len(X_test_fp)

    print("  loading 9-pool baseline OOF + test ...")
    oof_base, test_base, weights_map = load_caruana_baseline_oof_test()

    print("\n=== Adversarial classifier comparison ===")
    p_train_lr, auc_lr = fit_classifier_logreg(X_train_fp, X_test_fp)
    print(f"  LogReg   AUC (5-fold) = {auc_lr:.4f}")
    p_train_lgb, auc_lgb = fit_classifier_lgbm(X_train_fp, X_test_fp)
    print(f"  LightGBM AUC (5-fold) = {auc_lgb:.4f}  (delta = {auc_lgb - auc_lr:+.4f})")
    print(
        f"  p(test|train) median: LR={np.median(p_train_lr):.3f}  LGB={np.median(p_train_lgb):.3f}"
    )
    print(
        f"  p(test|train) max:    LR={np.max(p_train_lr):.3f}  LGB={np.max(p_train_lgb):.3f}"
    )
    rho_lr_lgb = float(spearmanr(p_train_lr, p_train_lgb).statistic)
    print(f"  Spearman(LR scores, LGB scores) = {rho_lr_lgb:.4f}")

    importance_w_lr = importance_weights_from_p(p_train_lr, n_tr, n_te)
    importance_w_lgb = importance_weights_from_p(p_train_lgb, n_tr, n_te)

    print("\n  fitting global importance affine (LogReg-based, prod recipe) ...")
    slope_lr, intercept_lr = fit_global_affine(oof_base, y_train, importance_w_lr)
    print(f"    affine: y = {slope_lr:.4f} * pred + {intercept_lr:.4f}")

    # Build potent-46 + UMAP folds for id=44 reconstruction
    from lb_proxy_metric_battery import potent46_indices  # noqa: E402

    potent_idx = potent46_indices()
    print(f"  potent-46 size: {len(potent_idx)}")
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
        "importance_slope": slope_lr,
        "importance_intercept": intercept_lr,
        "potent_idx": potent_idx,
        "folds": folds,
    }

    # Cases (label, oof builder, test CSV path)
    cases = [
        ("base9pool", lambda c: c["oof_base"].copy(), "ens_caruana_bag20.csv"),
        (
            "id42_family_meta",
            build_id42_oof,
            "ens_caruana_bag20_calibrated_importance_meta_id42.csv",
        ),
        ("id43_hybrid", build_id43_oof, "ens_hybrid_meta_baseline_5050.csv"),
        (
            "id44_anchor_residual",
            build_id44_oof,
            "ens_caruana_bag20_anchor_residual.csv",
        ),
        (
            "id45_admet_no_log2fc",
            build_id45_oof,
            "ens_caruana_bag20_admet_ai_no_log2fc_calibrated_importance.csv",
        ),
    ]

    print("\n=== Computing metrics per case ===\n")
    rows = []
    for label, oof_builder, test_csv in cases:
        oof = oof_builder(common)
        test = load_test_csv(test_csv)
        m = compute_metrics(
            oof=oof,
            y_train=y_train,
            importance_w_lr=importance_w_lr,
            importance_w_lgb=importance_w_lgb,
            p_train_lr=p_train_lr,
            p_train_lgb=p_train_lgb,
            importance_slope_lr=slope_lr,
            importance_intercept_lr=intercept_lr,
        )
        m["case"] = label
        m["test_mean"] = float(np.mean(test))
        m["test_std"] = float(np.std(test))
        rows.append(m)
        print(
            f"  {label:>26}  raw={m['M1_raw']:.4f}  cal={m['M2_cal_lr']:.4f}  "
            f"iw_lr={m['M4_iw_cal_lr']:.4f}  iw_lgb={m['M10_iw_cal_lgb']:.4f}  "
            f"top25={m['M12_top25_cal_mae']:.4f}  top25_sp={m['M13_top25_cal_sp']:.4f}  "
            f"bot25={m['M14_bot25_cal_mae']:.4f}"
        )

    df = pd.DataFrame(rows)
    base_row = df[df["case"] == "base9pool"].iloc[0]
    for m_key in [
        "M1_raw",
        "M2_cal_lr",
        "M3_iw_raw_lr",
        "M4_iw_cal_lr",
        "M9_iw_raw_lgb",
        "M10_iw_cal_lgb",
        "M12_top25_cal_mae",
        "M13_top25_cal_sp",
        "M14_bot25_cal_mae",
    ]:
        df[f"d_{m_key}"] = df[m_key] - base_row[m_key]

    print("\n=== Delta to base9pool ===")
    cols_show = [
        "case",
        "d_M2_cal_lr",
        "d_M4_iw_cal_lr",
        "d_M10_iw_cal_lgb",
        "d_M12_top25_cal_mae",
        "d_M13_top25_cal_sp",
        "d_M14_bot25_cal_mae",
    ]
    print(df[cols_show].to_string(index=False, float_format="%+.4f"))

    # === LB correlation across N=4 LB-known cases ===
    print("\n=== LB correlation (Spearman) for 4 LB-known cases ===\n")
    submitted = df[df["case"].isin(LB_KNOWN.keys())].copy()
    submitted["LB_MAE"] = submitted["case"].map(lambda c: LB_KNOWN[c][0])
    submitted["LB_Sp"] = submitted["case"].map(lambda c: LB_KNOWN[c][1])

    print(f"  N submitted = {len(submitted)}")
    print()
    print("  LB MAE ordering (lower = better):")
    print(
        submitted[["case", "LB_MAE", "LB_Sp"]]
        .sort_values("LB_MAE")
        .to_string(index=False)
    )
    print()

    print("  Correlation with LB MAE:")
    for m_key in [
        "M1_raw",
        "M2_cal_lr",
        "M3_iw_raw_lr",
        "M4_iw_cal_lr",
        "M9_iw_raw_lgb",
        "M10_iw_cal_lgb",
        "M12_top25_cal_mae",
        "M14_bot25_cal_mae",
        "d_M2_cal_lr",
        "d_M4_iw_cal_lr",
        "d_M10_iw_cal_lgb",
        "d_M12_top25_cal_mae",
    ]:
        rho = float(spearmanr(submitted[m_key], submitted["LB_MAE"]).statistic)
        marker = " *" if abs(rho) >= 0.5 else "  "
        print(f"   {marker} {m_key:>28}  Spearman = {rho:+.4f}")

    print("\n  Correlation with LB Sp (higher = better):")
    for m_key in ["M13_top25_cal_sp"]:
        rho = float(spearmanr(submitted[m_key], submitted["LB_Sp"]).statistic)
        print(f"     {m_key:>28}  Spearman = {rho:+.4f}")

    out_path = REPO_ROOT.joinpath(
        "docs", "superpowers", "runs", "2026-04-30-lb-proxy-battery-v2-table.csv"
    )
    df.to_csv(out_path, index=False)
    print(f"\nFull table written to {out_path}")


if __name__ == "__main__":
    main()

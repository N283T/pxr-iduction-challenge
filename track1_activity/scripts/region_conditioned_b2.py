"""Region-conditioned caruana, Phase B-2.

Building on Phase B-1 diagnostic which found:
  - Test = 100% in Q4 (highest test-likeness quartile from LightGBM AUC=0.96).
  - Per-regime caruana yields OOF Δ = -0.0069 (3x noise floor).
  - BUT Q4-localised chemprop family share ~0.89 vs global 0.76 sweet spot,
    Codex-flagged as LB-regress zone.

This script builds two variants and compares:
  V1 unconstrained: per-regime caruana, accepts whatever family share results.
  V2 family-capped: per-regime caruana with chemprop family share <= 0.70
                    (matches the production HARD constraint applied globally).

Test prediction effectively uses Q4 weights (all 513 test compounds map there).
For OOF evaluation each train compound uses its own regime's weights.

Outputs:
  - Per-regime weight tables (V1, V2) with family share annotation.
  - Overall + per-regime OOF MAE and Sp.
  - Importance-calibrated test predictions for both variants.
  - CSV submissions for V1 + V2 (saved but not auto-submitted).
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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "scripts")))
from data import (  # noqa: E402
    DB_PARAMS,
    load_test_smiles,
    load_train_smiles_target,
)

OUT_DIR = REPO_ROOT.joinpath("docs", "superpowers", "runs")
SUBMISSION_DIR = REPO_ROOT.joinpath("track1_activity", "submissions")
SEED = 42
N_QUARTILES = 4
WEIGHT_CLIP_LO = 1.0 / 3.0
WEIGHT_CLIP_HI = 3.0
FAMILY_CAP = 0.70

# Chemprop family members (per memory feedback_drop_lowweight_full_reverse_amp)
CHEMPROP_FAMILY = {
    "tabpfn_cheme_2d_full_boltz_log2fc_pred_optuna_trial10_seed5ens_umap_default",
    "tabpfn_cheme_2d_full_boltz_log2fc_pred_seed10ens_top500_umap",
    "tabpfn_chemprop_pretrain_embed_umap_default",
}


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


def fit_classifier_lgbm(
    X_tr: np.ndarray, X_te: np.ndarray, *, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, float]:
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
    p_train = full.predict_proba(X_tr.astype(np.float32))[:, 1]
    p_test = full.predict_proba(X_te.astype(np.float32))[:, 1]
    return p_train, p_test, auc


def load_pool() -> tuple[
    dict[str, np.ndarray], dict[str, np.ndarray], dict[str, float]
]:
    """Returns (oofs, test_preds, global_weights) for the 9-pool baseline."""
    with psycopg2.connect(**DB_PARAMS) as conn:
        cur = conn.cursor()
        cur.execute(
            """SELECT id, hyperparameters FROM experiments WHERE name = 'ens_caruana_bag20'
               ORDER BY id DESC LIMIT 1"""
        )
        _exp_id, hp = cur.fetchone()
        weights = hp["weights"]
        oofs: dict[str, np.ndarray] = {}
        for name in weights:
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
            oofs[name] = np.asarray([r[1] for r in rows], dtype=np.float64)

    test_preds: dict[str, np.ndarray] = {}
    for name in weights:
        sub = pd.read_csv(SUBMISSION_DIR.joinpath(f"{name}.csv"))
        col = [c for c in sub.columns if c.lower() == "pec50"][0]
        test_preds[name] = sub[col].to_numpy(dtype=np.float64)
    return oofs, test_preds, weights


def caruana_select(
    oofs: dict[str, np.ndarray],
    y: np.ndarray,
    *,
    sample_mask: np.ndarray | None = None,
    n_steps: int = 100,
    n_bags: int = 20,
    seed: int = 42,
    family_cap: float | None = None,
    family_members: set[str] | None = None,
) -> dict[str, float]:
    """Caruana with optional family-share cap.

    family_cap: if set, reject any greedy move that would push the running
    family-share count fraction above this value. Family-share is computed as
    (count of picks from family_members) / (total picks so far + 1).
    """
    members = list(oofs.keys())
    M = len(members)
    family_idx = (
        {i for i, m in enumerate(members) if m in family_members}
        if family_members is not None
        else set()
    )
    if sample_mask is None:
        sample_mask = np.ones(len(y), dtype=bool)
    y_sel = y[sample_mask]
    oof_arr = np.stack([oofs[m][sample_mask] for m in members], axis=1)

    rng = np.random.default_rng(seed)
    counts = np.zeros(M, dtype=np.float64)
    for _bag in range(n_bags):
        bag_idx = rng.choice(np.arange(len(y_sel)), size=len(y_sel), replace=True)
        y_bag = y_sel[bag_idx]
        oof_bag = oof_arr[bag_idx]
        bag_counts = np.zeros(M, dtype=np.float64)
        ens_sum = np.zeros_like(y_bag)
        n = 0
        family_picks = 0
        for _ in range(n_steps):
            best_m, best_mae = None, float("inf")
            for j in range(M):
                # Family cap check
                if family_cap is not None and j in family_idx:
                    proposed = (family_picks + 1) / (n + 1)
                    if proposed > family_cap:
                        continue
                cand_sum = ens_sum + oof_bag[:, j]
                cand_pred = cand_sum / (n + 1)
                mae = float(np.mean(np.abs(cand_pred - y_bag)))
                if mae < best_mae:
                    best_mae = mae
                    best_m = j
            if best_m is None:
                # All family-cap blocked AND no non-family available; pick best
                # ignoring cap as fallback to keep ensemble valid.
                for j in range(M):
                    cand_sum = ens_sum + oof_bag[:, j]
                    cand_pred = cand_sum / (n + 1)
                    mae = float(np.mean(np.abs(cand_pred - y_bag)))
                    if mae < best_mae:
                        best_mae = mae
                        best_m = j
            ens_sum = ens_sum + oof_bag[:, best_m]
            n += 1
            bag_counts[best_m] += 1.0
            if best_m in family_idx:
                family_picks += 1
        counts += bag_counts / bag_counts.sum()
    counts /= n_bags
    return {members[i]: float(counts[i]) for i in range(M)}


def fmt_weights(weights: dict[str, float], family: set[str]) -> str:
    lines = []
    fam_share = 0.0
    for m, w in sorted(weights.items(), key=lambda kv: -kv[1]):
        marker = "F" if m in family else " "
        lines.append(f"      {marker} {w:.3f}  {m}")
        if m in family:
            fam_share += w
    lines.append(f"    family_share = {fam_share:.4f}")
    return "\n".join(lines)


def family_share(weights: dict[str, float], family: set[str]) -> float:
    return sum(w for m, w in weights.items() if m in family)


def importance_weights_from_p(
    p_train: np.ndarray, n_train: int, n_test: int, *, eps: float = 1e-6
) -> np.ndarray:
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


def blend_oof(
    oofs: dict[str, np.ndarray],
    weights: dict[str, float],
    mask: np.ndarray | None = None,
) -> np.ndarray:
    norm = sum(weights.values())
    out = np.zeros_like(next(iter(oofs.values())))
    for name, w in weights.items():
        out += (w / norm) * oofs[name]
    if mask is not None:
        return out[mask]
    return out


def blend_test(
    test_preds: dict[str, np.ndarray], weights: dict[str, float]
) -> np.ndarray:
    norm = sum(weights.values())
    out = np.zeros_like(next(iter(test_preds.values())))
    for name, w in weights.items():
        out += (w / norm) * test_preds[name]
    return out


def main() -> None:
    print("Loading data ...")
    train_df = load_train_smiles_target()
    test_df = load_test_smiles()
    y_train = train_df["pec50"].to_numpy(dtype=np.float64)
    n_tr = len(train_df)

    X_tr = morgan_matrix(train_df["smiles"].tolist())
    X_te = morgan_matrix(test_df["smiles"].tolist())

    print("  fit LightGBM adversarial classifier ...")
    p_train, p_test, auc = fit_classifier_lgbm(X_tr, X_te)
    print(f"    AUC = {auc:.4f}")

    q_edges = np.quantile(p_train, np.linspace(0, 1, N_QUARTILES + 1))
    q_edges[0] = -np.inf
    q_edges[-1] = np.inf
    quartile = np.digitize(p_train, q_edges[1:-1], right=False)

    test_quartile = np.digitize(p_test, q_edges[1:-1], right=False)
    print("\n  Test quartile distribution:")
    for q in range(N_QUARTILES):
        n = (test_quartile == q).sum()
        print(f"    Q{q + 1}: n={n} ({100 * n / len(p_test):.1f}%)")

    print("\nLoading 9-pool ...")
    oofs, test_preds, global_w = load_pool()
    members = list(oofs.keys())
    print(f"  pool size = {len(members)}")
    print(
        f"  global weights family_share = {family_share(global_w, CHEMPROP_FAMILY):.4f}"
    )

    # Importance weights for calibration (LogReg-based, prod recipe)
    from importance_weights import compute_importance_weights  # noqa: E402

    iw = compute_importance_weights(
        train_df["smiles"].tolist(), test_df["smiles"].tolist()
    )

    # Global ensemble for reference
    global_oof = blend_oof(oofs, global_w)
    slope_g, intercept_g = fit_global_affine(global_oof, y_train, iw)
    print(f"  global affine: y = {slope_g:.4f} * pred + {intercept_g:.4f}")
    global_oof_cal = slope_g * global_oof + intercept_g
    global_oof_mae = float(np.mean(np.abs(global_oof - y_train)))
    global_oof_cal_mae = float(np.mean(np.abs(global_oof_cal - y_train)))
    print(f"  global OOF MAE raw={global_oof_mae:.4f} cal={global_oof_cal_mae:.4f}")

    # === Per-regime caruana: V1 unconstrained, V2 family-capped ===
    print("\n=== V1 unconstrained per-regime ===")
    v1_weights = {}
    for q in range(N_QUARTILES):
        mask = quartile == q
        if mask.sum() < 50:
            v1_weights[q] = global_w
            continue
        wq = caruana_select(oofs, y_train, sample_mask=mask, seed=SEED)
        v1_weights[q] = wq
        fs = family_share(wq, CHEMPROP_FAMILY)
        print(f"\n  Q{q + 1} (n={mask.sum()}) family_share={fs:.4f}")
        print(fmt_weights(wq, CHEMPROP_FAMILY))

    print(f"\n=== V2 family-capped (chemprop family <= {FAMILY_CAP}) per-regime ===")
    v2_weights = {}
    for q in range(N_QUARTILES):
        mask = quartile == q
        if mask.sum() < 50:
            v2_weights[q] = global_w
            continue
        wq = caruana_select(
            oofs,
            y_train,
            sample_mask=mask,
            seed=SEED,
            family_cap=FAMILY_CAP,
            family_members=CHEMPROP_FAMILY,
        )
        v2_weights[q] = wq
        fs = family_share(wq, CHEMPROP_FAMILY)
        print(f"\n  Q{q + 1} (n={mask.sum()}) family_share={fs:.4f}")
        print(fmt_weights(wq, CHEMPROP_FAMILY))

    # === OOF reconstruction per variant ===
    def region_oof(weights_per_q: dict[int, dict[str, float]]) -> np.ndarray:
        out = np.zeros_like(y_train)
        for q in range(N_QUARTILES):
            mask = quartile == q
            wmap = weights_per_q.get(q, global_w)
            norm = sum(wmap.values())
            for name, w in wmap.items():
                out[mask] += (w / norm) * oofs[name][mask]
        return out

    v1_oof = region_oof(v1_weights)
    v2_oof = region_oof(v2_weights)

    print("\n=== OOF MAE comparison ===")
    for label, oof in [
        ("global", global_oof),
        ("v1_unconstrained", v1_oof),
        ("v2_capped", v2_oof),
    ]:
        mae_raw = float(np.mean(np.abs(oof - y_train)))
        slope_l, intercept_l = fit_global_affine(oof, y_train, iw)
        oof_cal = slope_l * oof + intercept_l
        mae_cal = float(np.mean(np.abs(oof_cal - y_train)))
        sp_full = float(spearmanr(oof, y_train).statistic)
        # Q4-restricted (test-relevant region)
        q4_mask = quartile == 3
        mae_q4 = float(np.mean(np.abs(oof[q4_mask] - y_train[q4_mask])))
        sp_q4 = float(spearmanr(oof[q4_mask], y_train[q4_mask]).statistic)
        print(
            f"  {label:>20}  raw={mae_raw:.4f}  cal={mae_cal:.4f}  "
            f"sp_full={sp_full:.4f}  Q4_mae={mae_q4:.4f}  Q4_sp={sp_q4:.4f}"
        )

    print("\n=== Strict gate evaluation (M2 calibrated OOF MAE delta) ===")
    print(f"  global cal OOF MAE = {global_oof_cal_mae:.4f}")
    for label, oof in [("v1_unconstrained", v1_oof), ("v2_capped", v2_oof)]:
        slope_l, intercept_l = fit_global_affine(oof, y_train, iw)
        oof_cal = slope_l * oof + intercept_l
        mae_cal = float(np.mean(np.abs(oof_cal - y_train)))
        sp = float(spearmanr(oof, y_train).statistic)
        global_sp = float(spearmanr(global_oof, y_train).statistic)
        d_mae = mae_cal - global_oof_cal_mae
        d_sp = sp - global_sp
        m2_pass = d_mae <= -0.003
        sp_pass = d_sp >= -0.002
        # Compute family share IN Q4 (test-relevant region)
        q_for_test = 3  # Q4
        wq4 = (v1_weights if label == "v1_unconstrained" else v2_weights)[q_for_test]
        fs_q4 = family_share(wq4, CHEMPROP_FAMILY) / sum(wq4.values())
        family_pass = 0.65 <= fs_q4 <= 0.80
        all_pass = m2_pass and sp_pass and family_pass
        print(
            f"  {label}: ΔM2={d_mae:+.4f} ({'PASS' if m2_pass else 'FAIL'})  "
            f"ΔSp={d_sp:+.4f} ({'PASS' if sp_pass else 'FAIL'})  "
            f"Q4_family_share={fs_q4:.4f} ({'PASS' if family_pass else 'FAIL'})  "
            f"=> {'ALL PASS' if all_pass else 'BLOCKED'}"
        )

    # === Test predictions: route by test_quartile ===
    print("\n=== Test predictions (per-regime routing) ===")
    test_global = blend_test(test_preds, global_w)

    def region_test_pred(weights_per_q: dict[int, dict[str, float]]) -> np.ndarray:
        out = np.zeros_like(test_global)
        for q in range(N_QUARTILES):
            mask = test_quartile == q
            if mask.sum() == 0:
                continue
            wmap = weights_per_q.get(q, global_w)
            norm = sum(wmap.values())
            for name, w in wmap.items():
                out[mask] += (w / norm) * test_preds[name][mask]
        return out

    v1_test = region_test_pred(v1_weights)
    v2_test = region_test_pred(v2_weights)

    # Apply importance affine (computed from each variant's OOF)
    for label, oof, test in [
        ("global", global_oof, test_global),
        ("v1_unconstrained", v1_oof, v1_test),
        ("v2_capped", v2_oof, v2_test),
    ]:
        slope_l, intercept_l = fit_global_affine(oof, y_train, iw)
        test_cal = slope_l * test + intercept_l
        print(
            f"  {label:>20}  test mean={np.mean(test_cal):.3f} "
            f"std={np.std(test_cal):.3f}"
        )

    # Save submissions for V1 + V2 (with importance affine)
    test_smiles = test_df["smiles"].tolist()
    test_names = test_df["molecule_name"].astype(str).tolist()

    for label, oof, test in [
        ("v1_unconstrained", v1_oof, v1_test),
        ("v2_capped", v2_oof, v2_test),
    ]:
        slope_l, intercept_l = fit_global_affine(oof, y_train, iw)
        test_cal = slope_l * test + intercept_l
        out_path = SUBMISSION_DIR.joinpath(f"ens_region_conditioned_{label}.csv")
        df = pd.DataFrame(
            {"SMILES": test_smiles, "Molecule Name": test_names, "pEC50": test_cal}
        )
        df.to_csv(out_path, index=False)
        print(f"  saved {out_path.name}")

    print(f"\nUsed train n={n_tr}, test n={len(test_df)}")


if __name__ == "__main__":
    main()

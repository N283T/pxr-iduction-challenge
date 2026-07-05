"""MLPlatt regression calibrator (Bajger et al. arXiv:2601.08345v1, 2026-01).

Adapts MLPlatt (post-hoc calibration of ranker outputs to CTR probabilities)
to our PXR pEC50 regression task. Replaces the production importance-affine
calibrator with a small context-aware MLP that:

  - takes (base ensemble OOF pred, context features) -> calibrated pEC50
  - uses MAE loss (instead of BCE for CTR)
  - linear output (instead of sigmoid for probability)
  - gradient-penalty monotonicity w.r.t. the base prediction
    (preserves item ordering -> protects Sp)

Tests 4 context-feature variants:
  V1 scaffold_cluster: UMAP+KMeans cluster ID (50 clusters), one-hot
  V2 potency_bin: base prediction quartile (4 bins), one-hot
  V3 nn_potent46: Tanimoto NN distance to 46 potent train compounds (1-D)
  V4 combo: V1 + V2 + V3 concat

Per Codex Q3 priority (Sp-driven monotone calibrator) and DR Q3 ranking-
calibration recommendation. Today's id=46 LB result showed Sp -0.0041
regression -- exactly the failure mode this calibrator is designed to fix.

Strict gate: ΔM2 ≤ -0.003, ΔSp ≥ -0.002, family share unchanged (calibrator
doesn't touch ensemble weights), wt rule N/A.

Output: track1_activity/submissions/ens_caruana_bag20_mlplatt_<variant>.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
import torch
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "scripts")))
from data import (  # noqa: E402
    DB_PARAMS,
    load_test_smiles,
    load_train_smiles_target,
    load_train_smiles_with_counter,
)
from splits import _morgan_fp_matrix, umap_split_indices  # noqa: E402

OUT_DIR = REPO_ROOT.joinpath("track1_activity", "reports")
SUBMISSION_DIR = REPO_ROOT.joinpath("track1_activity", "submissions")
SEED = 42
N_CLUSTERS = 50
N_SPLITS = 5
WEIGHT_CLIP_LO = 1.0 / 3.0
WEIGHT_CLIP_HI = 3.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
POTENT_PEC50 = 6.0
POTENT_SEL = 1.5

# MLPlatt training hyperparams
N_EPOCHS = 200
LR = 5e-3
WD = 1e-4
THETA_MONO = 1.0  # monotonicity penalty weight
EMBED_DIM = 8


# ---------- data loading ----------


def load_pool() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, float]]:
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


def potent46_indices_with_pec50(
    train_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    df = load_train_smiles_with_counter()
    sel = df["pec50"] - df["counter_pec50"]
    mask = (df["pec50"] >= POTENT_PEC50) & (sel >= POTENT_SEL)
    idx = np.flatnonzero(mask.to_numpy())
    return idx, df["pec50"].to_numpy()[idx]


def nn_tanimoto_to_anchors(
    fps: np.ndarray, anchor_fps: np.ndarray, exclude_self: bool = True
) -> np.ndarray:
    q_pop = fps.sum(axis=1).astype(np.int32)
    a_pop = anchor_fps.sum(axis=1).astype(np.int32)
    inter = fps.astype(np.int32) @ anchor_fps.T.astype(np.int32)
    union = q_pop[:, None] + a_pop[None, :] - inter
    sim = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
    if exclude_self:
        # Exclude compounds whose Tanimoto = 1.0 (likely self) — only for
        # computing NN distance; this is a guard against potent-46 ⊂ train.
        sim = np.where(sim >= 0.999, -np.inf, sim)
    return sim.max(axis=1).astype(np.float64)


# ---------- importance weights ----------


def importance_weights(
    train_smiles: list[str], test_smiles: list[str]
) -> np.ndarray:
    from importance_weights import compute_importance_weights  # noqa: E402

    return compute_importance_weights(train_smiles, test_smiles)


def fit_global_affine(
    oof: np.ndarray, y: np.ndarray, w: np.ndarray
) -> tuple[float, float]:
    reg = LinearRegression()
    reg.fit(oof.reshape(-1, 1), y, sample_weight=w)
    return float(reg.coef_[0]), float(reg.intercept_)


# ---------- MLPlatt model ----------


class MLPlattRegressor(nn.Module):
    """Small context-aware monotonic regressor.

    Architecture mirrors the Allegro paper but adapted for regression:
      - Context Model: ctx_dim -> 16 -> 8 (ReLU)
      - Concat (8-dim ctx embed, 1-dim base score)
      - MonoMLP: 9 -> 8 -> 8 -> 8 -> 1 (ReLU intermediate, linear output)
      - No sigmoid (regression).
    """

    def __init__(self, ctx_dim: int, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.ctx_dim = ctx_dim
        if ctx_dim > 0:
            self.context_model = nn.Sequential(
                nn.Linear(ctx_dim, 16),
                nn.ReLU(),
                nn.Linear(16, embed_dim),
                nn.ReLU(),
            )
        else:
            self.context_model = None
            embed_dim = 0
        self.mono_mlp = nn.Sequential(
            nn.Linear(embed_dim + 1, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(
        self, ctx: torch.Tensor | None, score: torch.Tensor
    ) -> torch.Tensor:
        # score: (B,) — must require grad if computing monotone penalty
        if self.context_model is not None and ctx is not None:
            emb = self.context_model(ctx)
            x = torch.cat([emb, score.unsqueeze(-1)], dim=-1)
        else:
            x = score.unsqueeze(-1)
        return self.mono_mlp(x).squeeze(-1)


def mlplatt_loss(
    pred: torch.Tensor,
    y: torch.Tensor,
    score: torch.Tensor,
    sample_w: torch.Tensor,
    theta: float = THETA_MONO,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Weighted MAE
    mae = (sample_w * (pred - y).abs()).sum() / sample_w.sum()
    # Monotonicity: ∂pred/∂score must be >= 0
    grad_score = torch.autograd.grad(
        pred.sum(), score, create_graph=True, retain_graph=True
    )[0]
    mono = torch.relu(-grad_score).mean()
    return mae + theta * mono, mae.detach(), mono.detach()


def train_mlplatt(
    ctx_train: np.ndarray | None,
    score_train: np.ndarray,
    y_train: np.ndarray,
    sample_w: np.ndarray,
    ctx_dim: int,
    *,
    n_epochs: int = N_EPOCHS,
    lr: float = LR,
    wd: float = WD,
    seed: int = SEED,
    verbose: bool = False,
) -> MLPlattRegressor:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MLPlattRegressor(ctx_dim=ctx_dim).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    score_t = torch.tensor(score_train, dtype=torch.float32, device=DEVICE)
    y_t = torch.tensor(y_train, dtype=torch.float32, device=DEVICE)
    w_t = torch.tensor(sample_w, dtype=torch.float32, device=DEVICE)
    if ctx_dim > 0 and ctx_train is not None:
        ctx_t = torch.tensor(ctx_train, dtype=torch.float32, device=DEVICE)
    else:
        ctx_t = None

    for ep in range(n_epochs):
        model.train()
        opt.zero_grad()
        score_in = score_t.detach().clone().requires_grad_(True)
        pred = model(ctx_t, score_in)
        loss, mae, mono = mlplatt_loss(pred, y_t, score_in, w_t, theta=THETA_MONO)
        loss.backward()
        opt.step()
        if verbose and (ep % 20 == 0 or ep == n_epochs - 1):
            print(
                f"      ep {ep}: loss={loss.item():.4f}  mae={mae.item():.4f}  "
                f"mono={mono.item():.6f}"
            )
    return model


def predict_mlplatt(
    model: MLPlattRegressor, ctx: np.ndarray | None, score: np.ndarray
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        score_t = torch.tensor(score, dtype=torch.float32, device=DEVICE)
        if ctx is not None and model.ctx_dim > 0:
            ctx_t = torch.tensor(ctx, dtype=torch.float32, device=DEVICE)
        else:
            ctx_t = None
        pred = model(ctx_t, score_t)
    return pred.cpu().numpy().astype(np.float64)


# ---------- main ----------


def main() -> None:
    print("Loading data ...")
    train_df = load_train_smiles_target()
    test_df = load_test_smiles()
    y_train = train_df["pec50"].to_numpy(dtype=np.float64)
    train_smiles = train_df["smiles"].tolist()
    test_smiles = test_df["smiles"].tolist()

    print("  Morgan FPs ...")
    X_tr_fp = _morgan_fp_matrix(train_smiles)
    X_te_fp = _morgan_fp_matrix(test_smiles)

    print("Loading 9-pool ...")
    oofs, test_preds, global_w = load_pool()

    # base ensemble
    norm = sum(global_w.values())
    base_oof = np.zeros_like(y_train)
    base_test = np.zeros(len(test_df), dtype=np.float64)
    for name, w in global_w.items():
        base_oof += (w / norm) * oofs[name]
        base_test += (w / norm) * test_preds[name]

    # Importance affine production calibrator (reference)
    iw = importance_weights(train_smiles, test_smiles)
    slope_g, intercept_g = fit_global_affine(base_oof, y_train, iw)
    print(f"  importance affine: y = {slope_g:.4f} * pred + {intercept_g:.4f}")
    base_oof_cal = slope_g * base_oof + intercept_g
    base_test_cal = slope_g * base_test + intercept_g
    base_cal_mae = float(np.mean(np.abs(base_oof_cal - y_train)))
    base_sp = float(spearmanr(base_oof_cal, y_train).statistic)
    print(f"  base (importance affine) cal OOF MAE={base_cal_mae:.4f}  Sp={base_sp:.4f}")

    # === Context features ===
    print("\nBuilding context features ...")
    # Cluster IDs (UMAP + KMeans)
    print("  UMAP + KMeans 50 clusters ...")
    import umap

    reducer = umap.UMAP(
        n_components=10, metric="jaccard", random_state=SEED, n_neighbors=30
    )
    embedding_train = reducer.fit_transform(X_tr_fp)
    embedding_test = reducer.transform(X_te_fp)
    km = KMeans(n_clusters=N_CLUSTERS, random_state=SEED, n_init=10)
    cluster_train = km.fit_predict(embedding_train)
    cluster_test = km.predict(embedding_test)
    print(f"    train cluster distribution: {np.bincount(cluster_train)[:10]} ...")

    # Potency bin (base_oof_cal quartiles)
    pec_edges = np.quantile(base_oof_cal, [0.25, 0.5, 0.75])
    potency_train = np.digitize(base_oof_cal, pec_edges)
    potency_test = np.digitize(base_test_cal, pec_edges)

    # NN-Tanimoto to potent-46
    potent_idx, _ = potent46_indices_with_pec50(train_df)
    nn_train = nn_tanimoto_to_anchors(X_tr_fp, X_tr_fp[potent_idx], exclude_self=True)
    nn_test = nn_tanimoto_to_anchors(X_te_fp, X_tr_fp[potent_idx], exclude_self=False)
    print(
        f"    potent-46 size={len(potent_idx)}  nn_train mean={nn_train.mean():.3f}  "
        f"nn_test mean={nn_test.mean():.3f}"
    )

    def _onehot(labels: np.ndarray, n_classes: int) -> np.ndarray:
        oh = np.zeros((len(labels), n_classes), dtype=np.float32)
        oh[np.arange(len(labels)), labels] = 1.0
        return oh

    def make_ctx(variant: str, train_or_test: str) -> np.ndarray | None:
        if train_or_test == "train":
            cl, po, nn = cluster_train, potency_train, nn_train
        else:
            cl, po, nn = cluster_test, potency_test, nn_test
        if variant == "scaffold_cluster":
            return _onehot(cl, N_CLUSTERS)
        if variant == "potency_bin":
            return _onehot(po, 4)
        if variant == "nn_potent46":
            return nn.reshape(-1, 1).astype(np.float32)
        if variant == "combo":
            return np.concatenate(
                [_onehot(cl, N_CLUSTERS), _onehot(po, 4), nn.reshape(-1, 1).astype(np.float32)],
                axis=1,
            )
        if variant == "none":
            return None
        raise ValueError(variant)

    folds = umap_split_indices(train_smiles, n_splits=N_SPLITS, n_clusters=N_CLUSTERS, seed=SEED)

    variants = ["none", "scaffold_cluster", "potency_bin", "nn_potent46", "combo"]
    print("\n=== MLPlatt OOF bake-off ===\n")
    print(f"  base (importance affine): cal MAE={base_cal_mae:.4f}  Sp={base_sp:.4f}\n")

    rows = []
    test_predictions: dict[str, np.ndarray] = {}
    for variant in variants:
        print(f"  --- variant: {variant} ---")
        ctx_train_full = make_ctx(variant, "train")
        ctx_test_full = make_ctx(variant, "test")
        ctx_dim = ctx_train_full.shape[1] if ctx_train_full is not None else 0
        print(f"    ctx_dim = {ctx_dim}")

        # 5-fold OOF
        mlplatt_oof = np.zeros_like(y_train)
        for fold_idx, (tr, va) in enumerate(folds):
            score_tr = base_oof[tr]
            score_va = base_oof[va]
            y_tr = y_train[tr]
            sw_tr = iw[tr]
            ctx_tr = ctx_train_full[tr] if ctx_train_full is not None else None
            ctx_va = ctx_train_full[va] if ctx_train_full is not None else None
            model = train_mlplatt(
                ctx_tr,
                score_tr,
                y_tr,
                sw_tr,
                ctx_dim=ctx_dim,
                seed=SEED + fold_idx,
                verbose=(fold_idx == 0 and variant == "none"),
            )
            mlplatt_oof[va] = predict_mlplatt(model, ctx_va, score_va)

        cal_mae = float(np.mean(np.abs(mlplatt_oof - y_train)))
        sp = float(spearmanr(mlplatt_oof, y_train).statistic)
        d_m2 = cal_mae - base_cal_mae
        d_sp = sp - base_sp

        # Sanity: check OOF monotonicity (Spearman vs base_oof_cal)
        rho_with_base = float(spearmanr(mlplatt_oof, base_oof_cal).statistic)
        n_inversions = int(((mlplatt_oof[:, None] - mlplatt_oof[None, :]) * (base_oof_cal[:, None] - base_oof_cal[None, :]) < 0).sum() / 2)

        # Strict gate
        m2_pass = d_m2 <= -0.003
        sp_pass = d_sp >= -0.002
        all_pass = m2_pass and sp_pass

        print(
            f"    cal MAE={cal_mae:.4f}  ΔM2={d_m2:+.4f}  Sp={sp:.4f}  ΔSp={d_sp:+.4f}  "
            f"rho_with_base={rho_with_base:.4f}  inversions={n_inversions:,}"
        )
        gate = "ALL PASS" if all_pass else (
            "M2 fail" if not m2_pass else "Sp fail"
        )
        print(f"    gate: {gate}")

        # Train on full + predict test
        model_full = train_mlplatt(
            ctx_train_full,
            base_oof,
            y_train,
            iw,
            ctx_dim=ctx_dim,
            seed=SEED,
            verbose=False,
        )
        test_pred = predict_mlplatt(model_full, ctx_test_full, base_test)

        rows.append({
            "variant": variant,
            "ctx_dim": ctx_dim,
            "oof_mae": cal_mae,
            "d_m2": d_m2,
            "oof_sp": sp,
            "d_sp": d_sp,
            "rho_with_base": rho_with_base,
            "inversions": n_inversions,
            "all_pass": all_pass,
        })
        test_predictions[variant] = test_pred

    df = pd.DataFrame(rows)
    print("\n=== Summary ===\n")
    print(df.to_string(index=False, float_format="%.4f"))

    df.to_csv(OUT_DIR.joinpath("2026-04-30-mlplatt-bakeoff.csv"), index=False)

    # Save submission CSVs for variants that passed
    print("\n=== Submission CSV generation ===\n")
    test_smiles_list = test_df["smiles"].tolist()
    test_names = test_df["molecule_name"].astype(str).tolist()
    for row in rows:
        variant = row["variant"]
        if not row["all_pass"] and variant != "combo":
            continue
        test_pred = test_predictions[variant]
        out_path = SUBMISSION_DIR.joinpath(f"ens_caruana_bag20_mlplatt_{variant}.csv")
        out_df = pd.DataFrame(
            {"SMILES": test_smiles_list, "Molecule Name": test_names, "pEC50": test_pred}
        )
        out_df.to_csv(out_path, index=False)
        print(
            f"  saved {out_path.name}  test mean={np.mean(test_pred):.3f} "
            f"std={np.std(test_pred):.3f}  (gate={row['all_pass']})"
        )

    # === Recommendation ===
    candidates = [r for r in rows if r["all_pass"]]
    if candidates:
        best = min(candidates, key=lambda r: r["d_m2"])
        print(
            f"\n  RECOMMEND: {best['variant']} (ΔM2={best['d_m2']:+.4f}  ΔSp={best['d_sp']:+.4f})"
        )
    else:
        print("\n  No variants pass the strict gate. Defer.")


if __name__ == "__main__":
    main()

"""Pairwise ranking calibrator (Codex Q3 + DR Q3 ranking calibration path).

After MLPlatt MLP and per-bin affine both null on PXR (4140 train scale
insufficient for context-conditioning), this script tests context-FREE
calibrators that directly optimize ranking quality:

  V1 affine_pairwise: 2-param y = a*x + b optimized with MAE + λ * pairwise hinge
  V2 mono_mlp_pairwise: monotone MLP (1 -> 8 -> 8 -> 1) with grad penalty + pairwise

This is "global ranking optimization" -- no per-compound conditioning, so the
data scale problem from Phase B and MLPlatt is sidestepped. The hypothesis:
the global importance affine is optimal for MAE alone; adding pairwise hinge
loss might trade tiny MAE for meaningful Sp improvement.

The id=46 LB result showed Sp -0.0041 regression. If pairwise-trained
calibrator can recover Sp without trashing MAE, we have an LB-targeted lever.

Strict gate: ΔM2 ≤ -0.003 OR (ΔM2 close to 0 AND ΔSp >= +0.005).
The second is a relaxed gate for Sp-targeted methods that may be MAE-flat.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "scripts")))
from data import (  # noqa: E402
    DB_PARAMS,
    load_test_smiles,
    load_train_smiles_target,
)
from splits import umap_split_indices  # noqa: E402

OUT_DIR = REPO_ROOT.joinpath("docs", "superpowers", "runs")
SUBMISSION_DIR = REPO_ROOT.joinpath("track1_activity", "submissions")
SEED = 42
N_SPLITS = 5
N_CLUSTERS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparams
N_EPOCHS = 1500
LR = 1e-2
WD = 0.0
LAMBDA_PAIRWISE = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]  # lambda sweep
MARGIN = 0.05  # pairwise hinge margin in pEC50 units
THETA_MONO = 1.0  # monotonicity penalty for MLP variant


def load_pool() -> tuple[
    dict[str, np.ndarray], dict[str, np.ndarray], dict[str, float]
]:
    with psycopg2.connect(**DB_PARAMS) as conn:
        cur = conn.cursor()
        cur.execute(
            """SELECT id, hyperparameters FROM experiments WHERE name = 'ens_caruana_bag20'
               ORDER BY id DESC LIMIT 1"""
        )
        _, hp = cur.fetchone()
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


# ---------- Calibrators ----------


class AffineCalibrator(nn.Module):
    def __init__(self, init_slope: float = 1.0, init_intercept: float = 0.0):
        super().__init__()
        self.slope = nn.Parameter(torch.tensor([init_slope], dtype=torch.float32))
        self.intercept = nn.Parameter(
            torch.tensor([init_intercept], dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.slope * x + self.intercept


class MonoMLPCalibrator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.unsqueeze(-1)).squeeze(-1)


# ---------- Loss ----------


def pairwise_hinge_loss(
    pred: torch.Tensor, y: torch.Tensor, margin: float = MARGIN
) -> torch.Tensor:
    """All-pairs hinge: encourages pred ordering to match y ordering."""
    diff_pred = pred[:, None] - pred[None, :]
    diff_y = y[:, None] - y[None, :]
    sign_y = torch.sign(diff_y)  # +1 if y_i > y_j, -1 if y_i < y_j, 0 if tie
    # We want sign_y * diff_pred >= margin
    raw = torch.relu(margin - sign_y * diff_pred)
    # Mask: only count pairs with non-tie y
    mask = (sign_y != 0).float()
    n_pairs = mask.sum().clamp(min=1.0)
    return (raw * mask).sum() / n_pairs


def combined_loss(
    pred: torch.Tensor,
    y: torch.Tensor,
    sample_w: torch.Tensor,
    lambda_pair: float,
    margin: float = MARGIN,
    score: torch.Tensor | None = None,
    theta_mono: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mae = (sample_w * (pred - y).abs()).sum() / sample_w.sum()
    pair = pairwise_hinge_loss(pred, y, margin)
    total = mae + lambda_pair * pair
    if theta_mono > 0 and score is not None:
        grad = torch.autograd.grad(
            pred.sum(), score, create_graph=True, retain_graph=True
        )[0]
        mono = torch.relu(-grad).mean()
        total = total + theta_mono * mono
    return total, mae.detach(), pair.detach()


# ---------- Training ----------


def train_calibrator(
    model: nn.Module,
    score: np.ndarray,
    y: np.ndarray,
    sample_w: np.ndarray,
    lambda_pair: float,
    *,
    n_epochs: int = N_EPOCHS,
    lr: float = LR,
    wd: float = WD,
    margin: float = MARGIN,
    theta_mono: float = 0.0,
    seed: int = SEED,
) -> nn.Module:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    score_t = torch.tensor(score, dtype=torch.float32, device=DEVICE)
    y_t = torch.tensor(y, dtype=torch.float32, device=DEVICE)
    w_t = torch.tensor(sample_w, dtype=torch.float32, device=DEVICE)

    use_mono = theta_mono > 0
    for ep in range(n_epochs):
        model.train()
        opt.zero_grad()
        if use_mono:
            score_in = score_t.detach().clone().requires_grad_(True)
            pred = model(score_in)
        else:
            pred = model(score_t)
            score_in = None
        loss, _mae, _pair = combined_loss(
            pred,
            y_t,
            w_t,
            lambda_pair,
            margin=margin,
            score=score_in,
            theta_mono=theta_mono if use_mono else 0.0,
        )
        loss.backward()
        opt.step()
    return model


def predict_calib(model: nn.Module, score: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        s = torch.tensor(score, dtype=torch.float32, device=DEVICE)
        pred = model(s)
    return pred.cpu().numpy().astype(np.float64)


# ---------- Main ----------


def main() -> None:
    print("Loading data ...")
    train_df = load_train_smiles_target()
    test_df = load_test_smiles()
    y_train = train_df["pec50"].to_numpy(dtype=np.float64)
    train_smiles = train_df["smiles"].tolist()
    test_smiles = test_df["smiles"].tolist()

    print("Loading 9-pool ...")
    oofs, test_preds, global_w = load_pool()
    norm = sum(global_w.values())
    base_oof = np.zeros_like(y_train)
    base_test = np.zeros(len(test_df), dtype=np.float64)
    for name, w in global_w.items():
        base_oof += (w / norm) * oofs[name]
        base_test += (w / norm) * test_preds[name]

    from importance_weights import compute_importance_weights  # noqa: E402

    iw = compute_importance_weights(train_smiles, test_smiles)

    # Reference: production importance affine
    g_reg = LinearRegression()
    g_reg.fit(base_oof.reshape(-1, 1), y_train, sample_weight=iw)
    slope_g = float(g_reg.coef_[0])
    intercept_g = float(g_reg.intercept_)
    base_oof_cal = slope_g * base_oof + intercept_g
    base_cal_mae = float(np.mean(np.abs(base_oof_cal - y_train)))
    base_sp = float(spearmanr(base_oof_cal, y_train).statistic)
    print(f"  importance affine ref: cal MAE={base_cal_mae:.4f}  Sp={base_sp:.4f}")
    print(f"  slope={slope_g:.4f}  intercept={intercept_g:.4f}")

    folds = umap_split_indices(
        train_smiles, n_splits=N_SPLITS, n_clusters=N_CLUSTERS, seed=SEED
    )

    print("\n=== Pairwise calibrator OOF bake-off ===\n")
    print(
        f"  {'arch':>10}  {'λ':>5}  {'MAE':>7}  {'ΔM2':>8}  {'Sp':>7}  {'ΔSp':>8}  gate"
    )
    print("  " + "-" * 70)

    rows = []
    test_preds_dict: dict[str, np.ndarray] = {}
    for arch_name, ModelClass, theta_mono in [
        (
            "affine",
            lambda: AffineCalibrator(init_slope=slope_g, init_intercept=intercept_g),
            0.0,
        ),
        ("mono_mlp", lambda: MonoMLPCalibrator(), THETA_MONO),
    ]:
        for lam in LAMBDA_PAIRWISE:
            oof_pred = np.zeros_like(y_train)
            for fold_idx, (tr, va) in enumerate(folds):
                model = ModelClass()
                model = train_calibrator(
                    model,
                    base_oof[tr],
                    y_train[tr],
                    iw[tr],
                    lambda_pair=lam,
                    theta_mono=theta_mono,
                    seed=SEED + fold_idx,
                )
                oof_pred[va] = predict_calib(model, base_oof[va])
            cal_mae = float(np.mean(np.abs(oof_pred - y_train)))
            sp = float(spearmanr(oof_pred, y_train).statistic)
            d_m2 = cal_mae - base_cal_mae
            d_sp = sp - base_sp
            m2_pass = d_m2 <= -0.003
            sp_relax_pass = abs(d_m2) <= 0.001 and d_sp >= 0.005
            sp_strict_pass = d_sp >= -0.002
            all_pass_strict = m2_pass and sp_strict_pass
            all_pass_relax = sp_relax_pass

            label = f"{arch_name}_l{lam:g}"
            gate = (
                "STRICT" if all_pass_strict else ("RELAX" if all_pass_relax else "fail")
            )
            print(
                f"  {arch_name:>10}  {lam:>5.2f}  {cal_mae:.4f}  {d_m2:>+8.4f}  "
                f"{sp:.4f}  {d_sp:>+8.4f}  {gate}"
            )
            rows.append(
                {
                    "arch": arch_name,
                    "lambda": lam,
                    "cal_mae": cal_mae,
                    "d_m2": d_m2,
                    "sp": sp,
                    "d_sp": d_sp,
                    "gate_strict": all_pass_strict,
                    "gate_relax": all_pass_relax,
                }
            )

            # Save test pred (only for variants that pass any gate)
            if all_pass_strict or all_pass_relax:
                full_model = ModelClass()
                full_model = train_calibrator(
                    full_model,
                    base_oof,
                    y_train,
                    iw,
                    lambda_pair=lam,
                    theta_mono=theta_mono,
                    seed=SEED,
                )
                test_pred = predict_calib(full_model, base_test)
                test_preds_dict[label] = test_pred

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR.joinpath("2026-04-30-pairwise-calibrator.csv"), index=False)

    # === Save best candidates ===
    print("\n=== Submission CSV generation ===\n")
    test_smiles_list = test_df["smiles"].tolist()
    test_names = test_df["molecule_name"].astype(str).tolist()
    for label, test_pred in test_preds_dict.items():
        out_path = SUBMISSION_DIR.joinpath(f"ens_caruana_bag20_pairwise_{label}.csv")
        out_df = pd.DataFrame(
            {
                "SMILES": test_smiles_list,
                "Molecule Name": test_names,
                "pEC50": test_pred,
            }
        )
        out_df.to_csv(out_path, index=False)
        print(
            f"  saved {out_path.name}  test mean={np.mean(test_pred):.3f} std={np.std(test_pred):.3f}"
        )

    # Recommendation
    print("\n=== Recommendation ===")
    strict_pass = [r for r in rows if r["gate_strict"]]
    relax_pass = [r for r in rows if r["gate_relax"] and not r["gate_strict"]]
    if strict_pass:
        best = min(strict_pass, key=lambda r: r["d_m2"])
        print(
            f"  STRICT gate winner: {best['arch']} λ={best['lambda']}  "
            f"ΔM2={best['d_m2']:+.4f}  ΔSp={best['d_sp']:+.4f}"
        )
    if relax_pass:
        best_relax = max(relax_pass, key=lambda r: r["d_sp"])
        print(
            f"  RELAX gate winner (Sp-targeted, MAE-flat): {best_relax['arch']} "
            f"λ={best_relax['lambda']}  ΔM2={best_relax['d_m2']:+.4f}  "
            f"ΔSp={best_relax['d_sp']:+.4f}"
        )
    if not strict_pass and not relax_pass:
        print(
            "  All variants fail. Pairwise calibrator cannot improve over global affine."
        )


if __name__ == "__main__":
    main()

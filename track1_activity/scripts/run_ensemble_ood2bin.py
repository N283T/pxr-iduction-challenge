"""Phase C of #116 follow-up: OOD-conditional 2-bin caruana_bag20.

Motivation (Codex, 2026-04-24 #115 post-mortem): the existing pool is
embedding-heavy + GNN-family-dominated, with high internal residual r.
Adding a new "family" (Boltz-2 tier-0 tabular) helped OOF MAE
-0.0020. One remaining untapped lever is conditional weights: test
compounds that are close to train (in-domain) may benefit from a
different mix than those that are far from any training example (OOD).

Protocol:
  1. Load 10-member pool OOF + test predictions (same as run_ensemble.py).
  2. Compute Morgan 2048-bit r=2 fingerprints for all 4140 train + 513
     test compounds.
  3. Per-train "density" = max Tanimoto to any OTHER train compound.
     Split train into dense (>= median) and sparse (< median) subsets.
  4. Fit optimize_caruana twice: w_in on dense, w_ood on sparse.
  5. Per-test "sim_to_train" = max Tanimoto to any train compound.
     Bin test by median: in-domain (>= median) and OOD (< median).
  6. Apply w_in to in-domain test rows, w_ood to OOD rows. Stack to
     final 513-vector.
  7. Compute OOF MAE using the same bin assignment trick (apply w_in
     to dense OOF rows, w_ood to sparse OOF rows) as a sanity check;
     this is trivially optimal on the bins the weights were fit on, so
     also report a held-out 5-fold OOF MAE where the bin assignment
     (dense/sparse median) is recomputed per fold.

Output: ens_caruana_bag20_ood2bin.csv + experiment record.
Acceptance gate (same as Phase A): OOF MAE < baseline 10-pool
(0.4130) AND no single weight exceeds 0.5 in either bin.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import mean_absolute_error, r2_score

REPO_ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "scripts")))

from data import load_test_smiles, load_train_smiles_target  # noqa: E402
from evaluate import record_experiment  # noqa: E402

from run_ensemble import load_models, optimize_caruana  # noqa: E402

SUB_DIR = REPO_ROOT.joinpath("track1_activity", "submissions")


def morgan_bitfp(smiles_list: list[str]) -> np.ndarray:
    gen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
    out = np.zeros((len(smiles_list), 2048), dtype=np.uint8)
    for i, smi in enumerate(smiles_list):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        bv = gen.GetFingerprint(m)
        arr = np.frombuffer(bv.ToBitString().encode("ascii"), dtype=np.uint8) - ord("0")
        out[i] = arr
    return out


def pairwise_tanimoto_max(
    query: np.ndarray, ref: np.ndarray, skip_self: bool = False
) -> np.ndarray:
    """For each row in query, max Tanimoto to any row in ref.

    If skip_self=True and query is ref (same shape), the diagonal is
    excluded (used for density = self-excluded max NN within a set).
    """
    Q = query.astype(np.float32)
    R = ref.astype(np.float32)
    q_sum = Q.sum(axis=1)
    r_sum = R.sum(axis=1)
    inter = Q @ R.T
    union = q_sum[:, None] + r_sum[None, :] - inter
    sim = np.where(union > 0, inter / union, 0.0)
    if skip_self and Q.shape == R.shape and np.allclose(Q, R):
        np.fill_diagonal(sim, -1.0)
    return sim.max(axis=1)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RAE": float(
            np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true - np.mean(y_true)))
        ),
        "R2": float(r2_score(y_true, y_pred)),
        "Spearman_R": float(spearmanr(y_pred, y_true).statistic),
        "Kendall_Tau": float(kendalltau(y_pred, y_true).statistic),
    }


def main() -> None:
    train_df = load_train_smiles_target()
    test_df = load_test_smiles()
    y_train = train_df["pec50"].to_numpy(dtype=np.float64)
    n_test = len(test_df)

    model_names, oof_matrix, test_matrix = load_models(y_train, n_test=n_test)
    print(f"Pool: {len(model_names)} members")
    for n in model_names:
        print(f"  {n}")

    print("\nComputing Morgan fingerprints...")
    train_fp = morgan_bitfp(train_df["smiles"].tolist())
    test_fp = morgan_bitfp(test_df["smiles"].tolist())

    print("Computing train density (max Tanimoto to any OTHER train)...")
    train_density = pairwise_tanimoto_max(train_fp, train_fp, skip_self=True)
    thresh_train = float(np.median(train_density))
    dense_mask = train_density >= thresh_train
    sparse_mask = ~dense_mask
    print(
        f"  train density: median={thresh_train:.3f}, min={train_density.min():.3f}, "
        f"max={train_density.max():.3f}"
    )
    print(f"  dense (>= median):  {int(dense_mask.sum())}")
    print(f"  sparse (< median):  {int(sparse_mask.sum())}")

    print("\nFitting caruana on each subset...")
    w_in = optimize_caruana(
        oof_matrix[dense_mask], y_train[dense_mask], n_iter=100, n_bags=20, seed=42
    )
    w_ood = optimize_caruana(
        oof_matrix[sparse_mask], y_train[sparse_mask], n_iter=100, n_bags=20, seed=42
    )
    print("  w_in (dense)  : top 5")
    for i in np.argsort(-w_in)[:5]:
        if w_in[i] > 0:
            print(f"    {model_names[i]:>55s}  {w_in[i]:.4f}")
    print("  w_ood (sparse): top 5")
    for i in np.argsort(-w_ood)[:5]:
        if w_ood[i] > 0:
            print(f"    {model_names[i]:>55s}  {w_ood[i]:.4f}")

    # Reference: unconditional caruana
    w_uni = optimize_caruana(oof_matrix, y_train, n_iter=100, n_bags=20, seed=42)

    # OOF evaluation with OOD-conditional bin
    oof_cond = np.where(
        dense_mask[:, None],
        (oof_matrix * w_in).sum(axis=1, keepdims=True),
        (oof_matrix * w_ood).sum(axis=1, keepdims=True),
    ).ravel()
    oof_uni = (oof_matrix * w_uni).sum(axis=1)

    print("\nOOF comparisons (in-sample bin fit, upper bound):")
    print(f"  unconditional caruana_bag20:  {compute_metrics(y_train, oof_uni)}")
    print(f"  ood2bin (in-sample fit)   :   {compute_metrics(y_train, oof_cond)}")

    # Held-out: per-fold bin re-fit. Use a 5-fold random split on train.
    print("\nHeld-out OOF (5-fold refit of bin assignment)...")
    rng = np.random.default_rng(42)
    fold_idx = rng.integers(0, 5, size=len(y_train))
    oof_cv = np.zeros_like(y_train)
    for k in range(5):
        val = fold_idx == k
        trn = ~val
        # Recompute train density using only the trn subset
        dens = pairwise_tanimoto_max(train_fp[trn], train_fp[trn], skip_self=True)
        thr = float(np.median(dens))
        dense_trn = dens >= thr
        w_in_k = optimize_caruana(
            oof_matrix[trn][dense_trn],
            y_train[trn][dense_trn],
            n_iter=100,
            n_bags=20,
            seed=42,
        )
        w_ood_k = optimize_caruana(
            oof_matrix[trn][~dense_trn],
            y_train[trn][~dense_trn],
            n_iter=100,
            n_bags=20,
            seed=42,
        )
        # Bin the val rows by their sim to the trn subset
        val_sim = pairwise_tanimoto_max(train_fp[val], train_fp[trn])
        dense_val = val_sim >= thr
        oof_cv[val] = np.where(
            dense_val,
            oof_matrix[val] @ w_in_k,
            oof_matrix[val] @ w_ood_k,
        )
    print(f"  ood2bin held-out OOF (5-fold refit): {compute_metrics(y_train, oof_cv)}")

    # Test inference
    print("\nTest inference (bin by sim to train, median threshold)...")
    test_sim = pairwise_tanimoto_max(test_fp, train_fp)
    test_dense = test_sim >= thresh_train
    print(
        f"  test in-domain (>= {thresh_train:.3f}):  {int(test_dense.sum())} / {n_test}"
    )
    test_pred = np.where(
        test_dense,
        test_matrix @ w_in,
        test_matrix @ w_ood,
    )
    print(f"  test_pred: mean={test_pred.mean():.3f}, std={test_pred.std():.3f}")

    # Save submission
    out = pd.DataFrame(
        {
            "SMILES": test_df["smiles"],
            "Molecule Name": test_df["molecule_name"],
            "pEC50": test_pred,
        }
    )
    sub_path = SUB_DIR.joinpath("ens_caruana_bag20_ood2bin.csv")
    out.to_csv(sub_path, index=False)
    print(f"\nSaved: {sub_path}")

    exp_id = record_experiment(
        name="ens_caruana_bag20_ood2bin",
        description=(
            "OOD-conditional 2-bin caruana_bag20. Train split by "
            "self-excluded Morgan Tanimoto max NN (median). Test split "
            "by sim to any train (same threshold). Phase C of #116."
        ),
        model_type="ensemble",
        feature_set="multi-pool",
        hyperparameters={
            "pool": list(model_names),
            "bin_method": "morgan_density_median",
            "n_bags": 20,
            "seed": 42,
        },
        fold_metrics=[],
        submission_path=str(sub_path.relative_to(REPO_ROOT)),
        notes="OOD 2-bin caruana_bag20 following Codex suggestion (Phase C of #116).",
    )
    print(f"Recorded experiment id={exp_id}")


if __name__ == "__main__":
    main()

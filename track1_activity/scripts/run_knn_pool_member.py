"""kNN pool members: sim^2-weighted mean of pEC50 over train anchors.

Codex pivot 2026-04-29 (after PR #151 + PR #152 analog-prior tweak nulls):
add retrieval-based predictors as INDEPENDENT caruana pool members rather
than calibrator/correction add-ons.

Two members produced:
  knn_alltrain_umap   — anchors = all 4140 train compounds (self-exclude)
  knn_potent46_umap   — anchors = 46 potent-46 compounds (self-exclude)

Both use:
  pred(q) = sum_i (sim_i^2 * y_i) / sum_i (sim_i^2)
where sim = Tanimoto on Morgan r=2, 2048 bit FPs.

For each member:
  1. 5-fold UMAP CV (Morgan+Jaccard, k=50, seed=42) -> OOF predictions
     - per fold: anchor pool = (fold-train) intersect (anchor universe),
       predict val rows using sim^2-weighted mean.
  2. Final test predictions: anchor pool = full anchor universe.
  3. Write submission CSV (SMILES, Molecule Name, pEC50).
  4. Record experiment + save OOF to DB (idempotent: on_conflict_replace).

Legacy experiment script; internal design note was removed from the public repository.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

REPO_ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
from data import (  # noqa: E402
    load_test_smiles,
    load_train_smiles_target,
    load_train_smiles_with_counter,
)
from evaluate import (  # noqa: E402
    compute_metrics,
    record_experiment,
    save_oof_predictions,
)
from splits import umap_split_indices  # noqa: E402

POTENT_PEC50_THRESHOLD = 6.0
POTENT_SEL_THRESHOLD = 1.5
WEIGHT_POWER = 2
K_NEIGHBORS = (
    5  # top-k cutoff (without it, sim^2 over many anchors collapses to anchor mean)
)
N_SPLITS = 5
N_CLUSTERS = 50
SEED = 42
SUBMISSION_DIR = REPO_ROOT.joinpath("track1_activity", "submissions")


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
    """Indices into train_df (load_train_smiles_target order) for potent-46."""
    df = load_train_smiles_with_counter()
    sel = df["pec50"] - df["counter_pec50"]
    mask = (df["pec50"] >= POTENT_PEC50_THRESHOLD) & (sel >= POTENT_SEL_THRESHOLD)
    return np.flatnonzero(mask.to_numpy())


def tanimoto_matrix(query_fps: np.ndarray, anchor_fps: np.ndarray) -> np.ndarray:
    """(Nq, Na) Tanimoto similarity. Both inputs uint8 bit matrices."""
    q_pop = query_fps.sum(axis=1).astype(np.int32)
    a_pop = anchor_fps.sum(axis=1).astype(np.int32)
    inter = query_fps.astype(np.int32) @ anchor_fps.T.astype(np.int32)
    union = q_pop[:, None] + a_pop[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1), 0.0).astype(np.float64)


def topk_sim_weighted_mean(
    sim: np.ndarray,
    anchor_y: np.ndarray,
    k: int,
    self_exclude_mask: np.ndarray | None = None,
) -> np.ndarray:
    """For each query: top-k nearest anchors, sim^p-weighted mean of their y.

    Args:
        sim: (Nq, Na) similarity matrix (Tanimoto).
        anchor_y: (Na,) anchor pEC50 values.
        k: top-k cutoff. min(k, Na) is used if Na < k.
        self_exclude_mask: optional (Nq, Na) boolean mask; True positions are
            zeroed before the top-k selection so they cannot be selected.

    Without the top-k cutoff, sim^2 over many anchors collapses each
    prediction to the anchor mean (low-sim anchors dominate by sheer count).
    Top-k limits aggregation to the k most similar anchors per query.
    """
    sim_eff = sim.copy()
    if self_exclude_mask is not None:
        sim_eff = np.where(self_exclude_mask, 0.0, sim_eff)

    n_anchors = sim_eff.shape[1]
    k_eff = min(k, n_anchors)

    # Indices of top-k per query (argpartition is O(Na))
    top_idx = np.argpartition(-sim_eff, kth=k_eff - 1, axis=1)[:, :k_eff]
    rows = np.arange(sim_eff.shape[0])[:, None]
    top_sim = sim_eff[rows, top_idx]
    top_y = anchor_y[top_idx]

    w = top_sim**WEIGHT_POWER
    num = (w * top_y).sum(axis=1)
    den = w.sum(axis=1)
    if np.any(den == 0):
        raise RuntimeError(
            "topk_sim_weighted_mean: a query row has zero total weight in its "
            "top-k anchors (all top-k similarities are 0). Check anchor pool."
        )
    return num / den


def predict_oof_and_test(
    train_fps: np.ndarray,
    test_fps: np.ndarray,
    y_train: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    anchor_universe_idx: np.ndarray | None = None,
    member_label: str = "",
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Compute OOF + test predictions for one kNN pool member.

    Args:
        train_fps: (N_train, 2048).
        test_fps: (N_test, 2048).
        y_train: (N_train,) pEC50.
        folds: list of (train_idx, val_idx) pairs from UMAP split.
        anchor_universe_idx: indices into train (e.g. potent-46 indices)
            to use as anchors. None = all train rows.
        member_label: for log lines.

    Returns:
        oof_preds: (N_train,) — full coverage (UMAP partitions all train).
        test_preds: (N_test,) — predicted using all train rows in
            anchor_universe (no fold restriction).
        fold_metrics: list of compute_metrics dicts per fold.
    """
    if anchor_universe_idx is None:
        anchor_universe_idx = np.arange(len(train_fps))

    n_train = len(train_fps)
    oof_preds = np.zeros(n_train, dtype=np.float64)
    fold_metrics = []
    for fi, (tr_idx, va_idx) in enumerate(folds):
        # Anchor pool for this fold = anchor_universe ∩ fold-train
        fold_anchor_mask = np.isin(anchor_universe_idx, tr_idx)
        fold_anchor_idx = anchor_universe_idx[fold_anchor_mask]
        if len(fold_anchor_idx) < 1:
            raise RuntimeError(
                f"{member_label} fold {fi}: zero fold-train anchors "
                f"(anchor_universe size={len(anchor_universe_idx)})"
            )
        anchor_fps = train_fps[fold_anchor_idx]
        anchor_y = y_train[fold_anchor_idx]
        sim_va = tanimoto_matrix(train_fps[va_idx], anchor_fps)
        # No self-exclude needed: val rows are not in fold-train, so they
        # cannot match any anchor (anchor_idx ⊂ tr_idx, val_idx ∩ tr_idx = ∅).
        oof_preds[va_idx] = topk_sim_weighted_mean(sim_va, anchor_y, k=K_NEIGHBORS)
        m = compute_metrics(y_train[va_idx], oof_preds[va_idx])
        fold_metrics.append(m)
        print(
            f"  {member_label} fold {fi}: train={len(tr_idx)} val={len(va_idx)} "
            f"anchors={len(fold_anchor_idx)} "
            f"MAE={m['MAE']:.4f} Sp={m['Spearman_R']:.4f}"
        )

    # Test prediction: full anchor universe (no fold split)
    full_anchor_fps = train_fps[anchor_universe_idx]
    full_anchor_y = y_train[anchor_universe_idx]
    sim_test = tanimoto_matrix(test_fps, full_anchor_fps)
    test_preds = topk_sim_weighted_mean(sim_test, full_anchor_y, k=K_NEIGHBORS)

    return oof_preds, test_preds, fold_metrics


def write_submission_and_record(
    name: str,
    description: str,
    feature_set: str,
    hyperparameters: dict,
    test_df: pd.DataFrame,
    test_preds: np.ndarray,
    oof_preds: np.ndarray,
    fold_metrics: list[dict],
    notes: str,
) -> int:
    """Write submission CSV + record experiment + save OOF (idempotent)."""
    sub = pd.DataFrame(
        {
            "SMILES": test_df["smiles"],
            "Molecule Name": test_df["molecule_name"],
            "pEC50": test_preds,
        }
    )
    sub_path = SUBMISSION_DIR.joinpath(f"{name}.csv")
    sub.to_csv(sub_path, index=False)
    print(f"  wrote {sub_path}")

    exp_id = record_experiment(
        name=name,
        description=description,
        model_type="knn",
        feature_set=feature_set,
        hyperparameters=hyperparameters,
        fold_metrics=fold_metrics,
        submission_path=f"track1_activity/submissions/{name}.csv",
        notes=notes,
        on_conflict_replace=True,
    )
    save_oof_predictions(exp_id, oof_preds)
    return exp_id


def main() -> None:
    print("Loading data ...")
    train_df = load_train_smiles_target()
    test_df = load_test_smiles()
    y_train = train_df["pec50"].to_numpy(dtype=np.float64)
    print(f"  train={len(train_df)}, test={len(test_df)}")

    print("Computing Morgan fingerprints ...")
    X_train_fp = morgan_matrix(train_df["smiles"].tolist())
    X_test_fp = morgan_matrix(test_df["smiles"].tolist())
    print(f"  X_train_fp {X_train_fp.shape}, X_test_fp {X_test_fp.shape}")

    print("Loading potent-46 anchor indices ...")
    potent_idx = load_potent46_indices()
    print(f"  potent-46 size: {len(potent_idx)}")

    print(
        f"Building UMAP {N_SPLITS}-fold split (Morgan+Jaccard, k={N_CLUSTERS}, seed={SEED}) ..."
    )
    folds = umap_split_indices(
        train_df["smiles"].tolist(),
        n_splits=N_SPLITS,
        n_clusters=N_CLUSTERS,
        seed=SEED,
    )
    print(f"  built {len(folds)} folds")

    # ---- Member 1: knn_alltrain_umap ----
    print("\n=== knn_alltrain_umap ===")
    name_a = "knn_alltrain_umap"
    oof_a, test_a, fm_a = predict_oof_and_test(
        train_fps=X_train_fp,
        test_fps=X_test_fp,
        y_train=y_train,
        folds=folds,
        anchor_universe_idx=None,  # all train
        member_label=name_a,
    )
    overall_a = compute_metrics(y_train, oof_a)
    print(
        f"  full OOF: MAE={overall_a['MAE']:.4f}  Sp={overall_a['Spearman_R']:.4f}  "
        f"RAE={overall_a['RAE']:.4f}  R2={overall_a['R2']:.4f}"
    )
    print(
        f"  test preds: mean={test_a.mean():.4f} std={test_a.std():.4f} "
        f"min={test_a.min():.4f} max={test_a.max():.4f}"
    )
    write_submission_and_record(
        name=name_a,
        description="kNN sim^2-weighted mean over all train (UMAP 5-fold OOF)",
        feature_set="morgan_r2_2048_tanimoto_sim_squared",
        hyperparameters={
            "weight_power": WEIGHT_POWER,
            "k_neighbors": K_NEIGHBORS,
            "anchor_pool": "alltrain",
            "self_exclude": True,
            "n_splits": N_SPLITS,
            "n_clusters": N_CLUSTERS,
            "seed": SEED,
        },
        test_df=test_df,
        test_preds=test_a,
        oof_preds=oof_a,
        fold_metrics=fm_a,
        notes=(
            f"OOF MAE={overall_a['MAE']:.4f}, "
            f"sim^2-weighted mean over fold-train (4140 anchors universe), "
            f"Codex retrieval pivot 2026-04-29"
        ),
    )

    # ---- Member 2: knn_potent46_umap ----
    print("\n=== knn_potent46_umap ===")
    name_p = "knn_potent46_umap"
    oof_p, test_p, fm_p = predict_oof_and_test(
        train_fps=X_train_fp,
        test_fps=X_test_fp,
        y_train=y_train,
        folds=folds,
        anchor_universe_idx=potent_idx,
        member_label=name_p,
    )
    overall_p = compute_metrics(y_train, oof_p)
    print(
        f"  full OOF: MAE={overall_p['MAE']:.4f}  Sp={overall_p['Spearman_R']:.4f}  "
        f"RAE={overall_p['RAE']:.4f}  R2={overall_p['R2']:.4f}"
    )
    print(
        f"  test preds: mean={test_p.mean():.4f} std={test_p.std():.4f} "
        f"min={test_p.min():.4f} max={test_p.max():.4f}"
    )
    write_submission_and_record(
        name=name_p,
        description="kNN sim^2-weighted mean over potent-46 anchors (UMAP 5-fold OOF)",
        feature_set="morgan_r2_2048_tanimoto_sim_squared",
        hyperparameters={
            "weight_power": WEIGHT_POWER,
            "k_neighbors": K_NEIGHBORS,
            "anchor_pool": "potent46",
            "potent_pec50_threshold": POTENT_PEC50_THRESHOLD,
            "potent_sel_threshold": POTENT_SEL_THRESHOLD,
            "self_exclude": True,
            "n_splits": N_SPLITS,
            "n_clusters": N_CLUSTERS,
            "seed": SEED,
        },
        test_df=test_df,
        test_preds=test_p,
        oof_preds=oof_p,
        fold_metrics=fm_p,
        notes=(
            f"OOF MAE={overall_p['MAE']:.4f}, "
            f"sim^2-weighted mean over potent-46 ∩ fold-train (~36-37 anchors per fold), "
            f"Codex retrieval pivot 2026-04-29"
        ),
    )

    # ---- Cross-correlation report ----
    print("\nResidual correlation between the two new members:")
    r_aa_pp = float(np.corrcoef(oof_a, oof_p)[0, 1])
    print(f"  Pearson r(knn_alltrain, knn_potent46) = {r_aa_pp:.4f}")

    print(
        "\nDone. Both members recorded to DB. Next: edit ENSEMBLE_MODELS in run_ensemble.py and bake-off caruana variants."
    )


if __name__ == "__main__":
    main()

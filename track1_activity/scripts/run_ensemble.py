#!/usr/bin/env -S pixi run python
"""Canonical ensemble builder for Track 1.

This is the ONE canonical ensemble script. Previous versioned forks
(run_ensemble_v2.py, run_ensemble_v8.py, and the original run_ensemble.py)
have been moved to track1_activity/scripts/archive/. The v2/v8 naming
jumped (no v3-v7 ever existed) and created confusion about which script
was authoritative — this file resolves that by being the single source
of truth.

Key design choices that differ from the archived scripts:

1. **Explicit allow list**: ``ENSEMBLE_MODELS`` below hard-codes exactly
   which experiments enter the candidate pool. No silent pickup from
   DB queries. Previously the v8 script picked up whatever had OOF +
   a submission CSV, which allowed stale scaffold-split experiments
   (single_*, xgboost_mordred, catboost_mordred) to contaminate the
   pool. With an explicit list, adding or removing a candidate is
   a single-line audit trail.

2. **UMAP fold weights**: fold-based optimization uses
   ``umap_split_indices`` (matching the candidates' training split),
   not scaffold folds. The v2/v8 scripts optimized fold weights on
   scaffold folds even when the underlying OOF predictions were from
   UMAP folds — a subtle inconsistency that biased weight selection.

3. **No v* suffix in submission names**: submissions are
   ``ens_{strategy}.csv`` / DB name ``ens_{strategy}``. Previous
   ens_v7_* and ens_v8_* rows remain in the DB with an
   ``archived_ensemble_v7`` / ``archived_ensemble_v8`` ``model_type``
   prefix (see ``docs/ensemble_cleanup.md``) and are no longer
   generated.

All OOF RAE numbers annotated inline below are snapshots as of
2026-04-09. They will rot if any candidate is re-tuned — re-compute
from ``experiment_summary`` before trusting them for planning.

The candidate list is the union of:
  - ens_v7 UMAP-only members (dropping chemprop_scaffold and
    chemeleon_finetune per plan)
  - The tuned mordred_jazzy model (PR #45)
  - The best tuned 5-task multitask chemprop (chemprop_multitask5_umap
    _aux0.0_tuned; other aux-weight and 2-task variants are intentionally
    excluded)

Threshold: every model in the list has OOF RAE < 0.68 (project policy
for ensemble candidates).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.joinpath("src")))

import numpy as np
import pandas as pd
import psycopg2
from scipy.optimize import minimize

from data import DB_PARAMS, load_test_smiles, load_train_smiles_target
from evaluate import (
    compute_metrics,
    load_oof_predictions,
    record_experiment,
)
from splits import umap_split_indices

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SUBMISSION_DIR = REPO_ROOT.joinpath("track1_activity", "submissions")
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# The allow list — audit this before every ensemble run
# ---------------------------------------------------------------------------

# OOF RAE values below are snapshots from 2026-04-09 and may drift after
# any re-tuning. They are annotations for quick review, not live values.
ENSEMBLE_MODELS: tuple[str, ...] = (
    # 2026-04-18 PM prune: dropped all 13 zero-w_mae LGBMs (fingerprints +
    # embedding LGBMs + rdkit_desc_full + gap1.0 variants), dropped the 3
    # mixed-split LGBMs (LB flat on the morning submission, optimiser
    # re-weights on a redundant family), dropped gin/graphgps (near-zero
    # weight, OOF-weakest graph nets), replaced tabpfn_mordred_jazzy_umap
    # with tabpfn_2d_full_boltz_umap (workhorse upgrade: 2D full + Boltz
    # Tier-0/1 + pose-Jazzy, OOF RAE 0.5303 vs 0.5511 tuned). See
    # track1_activity/scripts/eda_cv_prep/14_prune_dryrun.py for the
    # 5-pool bake-off + 3-variant LGBM check that informed this list.
    #
    # 2026-04-19 PM prune (this edit): dropped 3 redundant members before
    # LB submission:
    #   chemprop_multitask5_umap_aux0.0_tuned (r=0.94 with chemprop_optuna,
    #     aux=0 makes it essentially a second chemprop_optuna seed)
    #   residual_physprop+mordred_umap (r=0.94 with tabpfn_2d_full_boltz,
    #     old (Apr 7) training, subsequent upgrades made its inputs a
    #     subset of the tabpfn workhorse)
    #   tabpfn_chemeleon_umap (r=0.94 with tabpfn_2d_full_boltz; CheMeleon
    #     signal insufficient to compensate for redundancy)
    # All three had caruana weights 0.03-0.06 (combined ~0.13 share).
    # Pool size 12 -> 9.
    # --- DL molecular graph (4) ---
    "chemprop_optuna_umap",  # D-MPNN, pool-top by weight
    "chemprop_chemeleon_umap",  # CheMeleon foundation finetune
    "attentivefp_optuna_umap",
    # gatedgcn_optuna_umap replaced 2026-04-19 by its pretrain+frozen variant
    # (OOF MAE 0.5464 -> 0.5077, r=0.924 = strict supersede).
    "gatedgcn_pretrain_finetune_frozen_umap",  # pretrain+frozen upgrade
    # --- Foundation tabular (2) ---
    # tabpfn_2d_full_boltz_umap replaced 2026-04-20 by its
    # log2fc_pred-augmented variant (same 1801 base features + 2
    # predicted log2_fc scalars from the chemprop pretrain model).
    # OOF MAE 0.4825 -> 0.4429 (-0.04); r=0.975 = strict supersession.
    # Buterez 2024 strategy-2 (predicted LF labels as side features).
    "tabpfn_2d_full_boltz_log2fc_pred_umap_default",
    # TabPFN on chemprop pretrain encoder fingerprint (256d).
    # Buterez 2024 strategy-3 (LF embedding as side feature).
    # OOF MAE 0.4373 / Spearman 0.8102 -- pool single-best.
    # Correlated r=0.962 with the log2fc_pred variant above (both
    # carry pretrain signal), caruana spreads weight between them.
    "tabpfn_chemprop_pretrain_embed_umap_default",
    # --- Boltz trunk embedding pool (3) ---
    # 1024-dim pooled trunk embeddings (s_prot_mean 384 + s_lig_mean 384 +
    # z_interface_mean/max 128+128). Three variants: LGBM Optuna tuned (core
    # pocket), TabPFN default on core pocket, TabPFN default on all protein
    # x ligand pairs. Single OOF MAEs: 0.5116 / 0.4861 / 0.4860.
    # Inter-correlation 0.96-0.99, so they only add diversity via
    # Caruana-style weight spreading.
    "lgbm_pooled_boltz_umap",
    "tabpfn_pooled_boltz_umap_default",
    "tabpfn_pooled_boltz_allpairs_umap_default",
    # 2026-04-20 PM drop: peft_molformer_xl_lora_r32a64_umap_default (added
    # in PR #95) regressed LB rank 9 -> 10 (RAE 0.5545 -> 0.5565, MAE 0.4414
    # -> 0.4430) despite OOF -0.0087. Classic OOF/LB disconnect. Framework
    # + experiment id=581 stay in DB; just dropped from the allow-list.
    # 2026-04-20 PM drop: chemprop_relative_aux_umap_default (added in PR
    # #97) single-model OOF regressed on ALL metrics vs chemprop_optuna
    # baseline (MAE 0.5208 -> 0.5457, Spearman 0.7252 -> 0.7118). Caruana
    # confirmed rejection: wt=0, pool MAE unchanged. FMGCL aux on PXR is
    # falsified; do not revisit this loss family without new evidence.
    # Framework (losses.py + run_chemprop_relative_aux.py) stays on main.
    # --- MoLFormer-c3 pretrain embed + TabPFN (1) ---
    # Transformer-encoder-family analog of tabpfn_chemprop_pretrain_embed.
    # Same recipe (Buterez 2024 strategy-3: pretrain on log2_fc ->
    # frozen embedding -> TabPFN) but different backbone family
    # (MoLFormer-c3 transformer vs chemprop D-MPNN). Single-model OOF
    # MAE 0.4753 (chemprop version 0.4373, but decorrelation hypothesis
    # justifies pool addition). Name lacks "_default" suffix because
    # run_train.py used its 20-trial Optuna default, not --trials 0.
    # PR #98.
    "tabpfn_molformer_c3_pretrain_embed_umap",
    # --- KERMT/GROVER pretrain embed + TabPFN ---
    # Graph-transformer family (continued-pretrain of GROVER_base on
    # single_concentration log2_fc, frozen, then TabPFN with
    # ignore_pretraining_limits=True for 3200d). Added as a
    # decorrelating 11th pool member. Previous pretrain-embed backbones:
    # chemprop (GNN) and MoLFormer-c3 (transformer). See
    # docs/superpowers/specs/2026-04-21-kermt-grover-pretrain-embed-design.md.
    # Single-model OOF MAE 0.4485, Pearson r vs others: chemprop 0.945,
    # molformer_c3 0.918, 2d_full_boltz 0.943. PR <TBD>.
    "tabpfn_kermt_pretrain_embed_umap_default",
    # --- AttentiveFP pretrain embed + TabPFN ---
    # Graph-attention family (PyG AttentiveFP pretrained on
    # single_concentration log2_fc via PR #79 checkpoint, then 512d
    # readout frozen into TabPFN). Added as decorrelating 12th pool
    # member. Complements chemprop (D-MPNN), molformer_c3 (transformer),
    # kermt (graph-transformer). See
    # docs/superpowers/specs/2026-04-21-attentivefp-pretrain-embed-design.md.
    # Single-model OOF MAE 0.4844, Pearson r vs existing:
    # chemprop 0.937, molformer_c3 0.922, 2d_full_boltz 0.941, kermt 0.920.
    # 12-pool caruana_bag20 OOF MAE 0.4242 (-0.0026 vs 11-pool 0.4268),
    # attentivefp caruana weight 0.0223. PR <TBD>.
    "tabpfn_attentivefp_pretrain_embed_umap_default",
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_models(
    y_train: np.ndarray, n_test: int
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Load OOF + test predictions for every model in ``ENSEMBLE_MODELS``.

    Fails loudly if anything is missing: the allow list is authoritative.
    Weight optimization assumes at least two candidates (a single-model
    "ensemble" is just the model itself and all six strategies collapse
    to the same weights), so a pool of < 2 raises.
    """
    if len(ENSEMBLE_MODELS) < 2:
        raise RuntimeError(
            f"ENSEMBLE_MODELS has {len(ENSEMBLE_MODELS)} entries; need >= 2 "
            "for weight optimization to be meaningful."
        )

    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    cur.execute(
        "SELECT id, name, submission_path FROM experiments WHERE name = ANY(%s)",
        (list(ENSEMBLE_MODELS),),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    rows_by_name = {r[1]: r for r in rows}
    missing = [n for n in ENSEMBLE_MODELS if n not in rows_by_name]
    if missing:
        raise RuntimeError(
            f"ENSEMBLE_MODELS not found in experiments table: {missing}. "
            "Fix the allow list or the DB before re-running."
        )

    names: list[str] = []
    oofs: list[np.ndarray] = []
    tests: list[np.ndarray] = []

    for name in ENSEMBLE_MODELS:
        exp_id, _, sub_path = rows_by_name[name]

        # OOF
        oof = load_oof_predictions(exp_id)
        if oof is None:
            raise RuntimeError(f"{name}: no OOF predictions in DB")
        if len(oof) != len(y_train):
            raise RuntimeError(
                f"{name}: OOF length {len(oof)} != train length {len(y_train)}"
            )

        # Test — validate CSV has the expected column and length
        if sub_path is None:
            raise RuntimeError(f"{name}: experiments.submission_path is NULL")
        csv_path = REPO_ROOT.joinpath(sub_path)
        if not csv_path.exists():
            raise RuntimeError(f"{name}: submission CSV not found at {csv_path}")
        test_df = pd.read_csv(csv_path)
        if "pEC50" not in test_df.columns:
            raise RuntimeError(
                f"{name}: submission CSV {csv_path} has no 'pEC50' column "
                f"(got {list(test_df.columns)})"
            )
        if len(test_df) != n_test:
            raise RuntimeError(
                f"{name}: submission CSV has {len(test_df)} rows, "
                f"expected {n_test} (test_activity length)"
            )
        test_pred = test_df["pEC50"].to_numpy()

        names.append(name)
        oofs.append(oof)
        tests.append(test_pred)

    oof_matrix = np.column_stack(oofs)
    test_matrix = np.column_stack(tests)
    return names, oof_matrix, test_matrix


# ---------------------------------------------------------------------------
# Weight optimization strategies
# ---------------------------------------------------------------------------


def normalize_weights(w: np.ndarray) -> np.ndarray:
    """L1-normalize the absolute weights so they sum to 1.

    Raises if the optimizer collapsed to all-zero weights — that outcome
    is pathological and silently turning it into uniform weights would
    hide the failure behind an ensemble that looks like a legitimate
    weighted blend. ``ens_simple_avg`` already covers the "everything
    uniform" strategy as a distinct, labeled submission.
    """
    w_abs = np.abs(w)
    total = w_abs.sum()
    if total < 1e-12:
        raise RuntimeError(
            "normalize_weights: optimizer produced all-zero weights "
            "(|sum| < 1e-12). This is pathological — refusing to fall "
            "back to uniform weights silently. Investigate the objective."
        )
    return w_abs / total


def _check(result, label: str) -> None:
    """Raise if scipy.optimize.minimize did not converge.

    The previous implementation only printed a warning, which meant
    non-converged weights silently flowed into the submission CSV and
    DB record. For a 20-dim Nelder-Mead at maxiter=50000 a non-success
    return is rare but plausible, and its weights are untrustworthy —
    stop the run and surface the failure loudly instead.
    """
    if not result.success:
        raise RuntimeError(
            f"{label}: optimizer did not converge ({result.message}). "
            "Refusing to use the last iterate as production weights."
        )


def _rae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(
        np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true - y_true.mean()))
    )


def optimize_vanilla(oof: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = oof.shape[1]

    def obj(w):
        return _rae(y, oof @ normalize_weights(w))

    result = minimize(
        obj,
        np.ones(n) / n,
        method="Nelder-Mead",
        options={"maxiter": 50000, "xatol": 1e-8, "fatol": 1e-8},
    )
    _check(result, "vanilla")
    return normalize_weights(result.x)


def optimize_l2(oof: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    n = oof.shape[1]
    equal = np.ones(n) / n

    def obj(w):
        wn = normalize_weights(w)
        return _rae(y, oof @ wn) + alpha * np.sum((wn - equal) ** 2)

    result = minimize(
        obj,
        equal.copy(),
        method="Nelder-Mead",
        options={"maxiter": 50000, "xatol": 1e-8, "fatol": 1e-8},
    )
    _check(result, f"l2(alpha={alpha})")
    return normalize_weights(result.x)


def optimize_caruana(
    oof: np.ndarray,
    y: np.ndarray,
    *,
    n_iter: int = 100,
    init_top_n: int = 3,
    bag_frac: float = 0.5,
    n_bags: int = 20,
    seed: int = 42,
) -> np.ndarray:
    """Caruana 2004 forward stepwise selection with replacement,
    sorted initialization, and bagged selection.

    Motivation: continuous weight optimization (vanilla / L2) concentrates
    weight onto whichever member has the best OOF, regardless of how
    correlated it is with other strong members. When that member's OOF
    advantage does not fully reflect LB behaviour, the optimiser
    overpays and the blend regresses. Caruana selection uses discrete
    counts and bagging over library subsets, which naturally spreads
    weight across correlated-but-strong members.

    Returns a simplex-normalized weight vector over all input columns.
    Models not chosen in any bag receive weight 0.
    """
    rng = np.random.RandomState(seed)
    N, M = oof.shape
    counts = np.zeros(M, dtype=np.int64)

    bag_size = max(init_top_n + 1, int(M * bag_frac))
    for _ in range(n_bags):
        members = rng.choice(M, size=bag_size, replace=False)
        bag_arr = oof[:, members]  # (N, bag_size)
        bag_maes = np.mean(np.abs(bag_arr - y[:, None]), axis=0)
        sort_idx = np.argsort(bag_maes)
        top = sort_idx[:init_top_n]

        bag_counts = np.zeros(bag_size, dtype=np.int64)
        if init_top_n > 0:
            bag_counts[top] = 1
            current_sum = bag_arr[:, top].sum(axis=1)
        else:
            current_sum = np.zeros(N)
        current_count = int(bag_counts.sum())

        for _ in range(n_iter):
            cand_pred = (current_sum[:, None] + bag_arr) / (current_count + 1)
            cand_maes = np.mean(np.abs(cand_pred - y[:, None]), axis=0)
            best = int(np.argmin(cand_maes))
            bag_counts[best] += 1
            current_sum = current_sum + bag_arr[:, best]
            current_count += 1

        for idx_in_bag, global_idx in enumerate(members):
            counts[global_idx] += int(bag_counts[idx_in_bag])

    total = int(counts.sum())
    if total == 0:
        raise RuntimeError("Caruana selection produced zero total count")
    return counts.astype(np.float64) / total


def optimize_fold(
    oof: np.ndarray, y: np.ndarray, splits: list, alpha: float = 0.0
) -> np.ndarray:
    n = oof.shape[1]
    equal = np.ones(n) / n
    per_fold = []

    for i, (_, val_idx) in enumerate(splits):
        ov = oof[val_idx]
        yv = y[val_idx]

        def obj(w):
            wn = normalize_weights(w)
            r = _rae(yv, ov @ wn)
            if alpha > 0:
                r += alpha * np.sum((wn - equal) ** 2)
            return r

        result = minimize(
            obj,
            equal.copy(),
            method="Nelder-Mead",
            options={"maxiter": 50000, "xatol": 1e-8, "fatol": 1e-8},
        )
        _check(result, f"fold_{i}(alpha={alpha})")
        per_fold.append(normalize_weights(result.x))

    # Average the per-fold weight vectors, then re-normalize. Each
    # per_fold entry is already simplex-normalized (sums to 1), so the
    # mean also sums to 1 in exact arithmetic; the final division
    # exists only to scrub floating-point drift.
    avg = np.mean(per_fold, axis=0)
    return avg / avg.sum()


# ---------------------------------------------------------------------------
# Evaluation and submission
# ---------------------------------------------------------------------------


def evaluate_and_record(
    name: str,
    weights: np.ndarray,
    oof_matrix: np.ndarray,
    y_train: np.ndarray,
    test_matrix: np.ndarray,
    test_df: pd.DataFrame,
    model_names: list[str],
) -> dict:
    blended_oof = oof_matrix @ weights
    metrics = compute_metrics(y_train, blended_oof)

    print(f"\n  {name}:")
    print(
        f"    OOF RAE={metrics['RAE']:.4f}  MAE={metrics['MAE']:.4f}  "
        f"R2={metrics['R2']:.4f}  Spearman={metrics['Spearman_R']:.4f}"
    )

    # Display threshold: 0.02 keeps the "significant" list focused on
    # models with >= 2% weight. Below that, contributions are within
    # normalization noise for a 20-model pool (1/20 = 5% uniform
    # baseline, so 0.02 is under half of uniform).
    significant = sorted(
        [(n, w) for n, w in zip(model_names, weights) if w > 0.02],
        key=lambda x: -x[1],
    )
    print(f"    Weights ({len(significant)} significant):")
    for n, w in significant:
        print(f"      {n:<45} {w:.4f}")

    blended_test = test_matrix @ weights
    sub = pd.DataFrame(
        {
            "SMILES": test_df["smiles"],
            "Molecule Name": test_df["molecule_name"],
            "pEC50": blended_test,
        }
    )
    sub_filename = f"{name}.csv"
    sub.to_csv(SUBMISSION_DIR.joinpath(sub_filename), index=False)

    # Ensemble names (``ens_*``) are stable across runs, so we pass
    # ``on_conflict_replace=True`` to replace any prior row + its cascaded
    # cv_results / oof_predictions. Regular experiment names still error on
    # duplicate insert. The CSV is written regardless so a hard DB failure
    # is recoverable from disk.
    weight_dict = {n: float(w) for n, w in zip(model_names, weights)}
    try:
        record_experiment(
            name=name,
            description=f"Canonical ensemble ({name})",
            model_type="ensemble",
            feature_set="weighted_blend",
            hyperparameters={
                "weights": weight_dict,
                "strategy": name.replace("ens_", ""),
                "pool_size": len(model_names),
                "pool_models": list(model_names),
            },
            fold_metrics=[metrics],
            submission_path=f"track1_activity/submissions/{sub_filename}",
            notes=f"OOF RAE={metrics['RAE']:.4f}, canonical (UMAP-only)",
            on_conflict_replace=True,
        )
    except Exception as exc:  # noqa: BLE001 — DB errors are diverse
        print(
            f"    [ERROR] {name}: record_experiment failed ({type(exc).__name__}: "
            f"{exc}). CSV still on disk at submissions/{sub_filename}; continuing."
        )

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("Loading data...")
    train_df = load_train_smiles_target()
    test_df = load_test_smiles()
    y_train = train_df["pec50"].to_numpy()

    print(f"Train rows: {len(y_train)}, test rows: {len(test_df)}")
    print(f"Allow list: {len(ENSEMBLE_MODELS)} models")

    model_names, oof_matrix, test_matrix = load_models(y_train, n_test=len(test_df))
    print(f"\nLoaded {len(model_names)} models with aligned OOF + test.")

    # Print per-model OOF RAE as a sanity check
    for i, name in enumerate(model_names):
        rae_i = _rae(y_train, oof_matrix[:, i])
        print(f"  {name:<48} OOF RAE = {rae_i:.4f}")

    # UMAP folds for fold-based weight optimization (matches candidates' split)
    print("\nBuilding UMAP folds for fold-based weight optimization...")
    umap_splits = umap_split_indices(train_df["smiles"].tolist(), n_splits=5, seed=42)

    n = len(model_names)
    equal_w = np.ones(n) / n

    print("\n" + "=" * 70)
    print("  STRATEGY COMPARISON")
    print("=" * 70)

    results: dict[str, dict] = {}

    # 1. Simple average
    results["simple_avg"] = evaluate_and_record(
        "ens_simple_avg",
        equal_w,
        oof_matrix,
        y_train,
        test_matrix,
        test_df,
        model_names,
    )

    # 2. Vanilla unconstrained
    w_vanilla = optimize_vanilla(oof_matrix, y_train)
    results["vanilla"] = evaluate_and_record(
        "ens_vanilla",
        w_vanilla,
        oof_matrix,
        y_train,
        test_matrix,
        test_df,
        model_names,
    )

    # 3. L2-regularized
    for alpha in (0.05, 0.1, 0.3, 0.5):
        w = optimize_l2(oof_matrix, y_train, alpha=alpha)
        results[f"l2_a{alpha}"] = evaluate_and_record(
            f"ens_l2_a{alpha}",
            w,
            oof_matrix,
            y_train,
            test_matrix,
            test_df,
            model_names,
        )

    # 4. Fold-based (UMAP folds)
    w_fold = optimize_fold(oof_matrix, y_train, umap_splits, alpha=0.0)
    results["fold"] = evaluate_and_record(
        "ens_fold",
        w_fold,
        oof_matrix,
        y_train,
        test_matrix,
        test_df,
        model_names,
    )

    # 5. Fold-based + L2 (UMAP folds)
    for alpha in (0.1, 0.3):
        w = optimize_fold(oof_matrix, y_train, umap_splits, alpha=alpha)
        results[f"fold_l2_a{alpha}"] = evaluate_and_record(
            f"ens_fold_l2_a{alpha}",
            w,
            oof_matrix,
            y_train,
            test_matrix,
            test_df,
            model_names,
        )

    # 6. Caruana 2004 forward stepwise selection with bagging
    # Structural regularization: discrete count-based weights + bagged
    # library subsets prevent any single model from capturing > ~0.3
    # weight, which mitigates the OOF-LB gap from correlated members.
    # See docs/papers/shotgun_raw/shotgun.txt for the paper.
    w_caru = optimize_caruana(
        oof_matrix,
        y_train,
        n_iter=100,
        init_top_n=3,
        bag_frac=0.5,
        n_bags=20,
        seed=42,
    )
    results["caruana_bag20"] = evaluate_and_record(
        "ens_caruana_bag20",
        w_caru,
        oof_matrix,
        y_train,
        test_matrix,
        test_df,
        model_names,
    )

    # Summary table
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(
        f"  {'Strategy':<25}  {'OOF RAE':>8}  {'OOF MAE':>8}  {'OOF R2':>8}  {'Spearman':>9}"
    )
    print("  " + "-" * 66)
    sorted_results = sorted(results.items(), key=lambda kv: kv[1]["RAE"])
    for strategy, m in sorted_results:
        print(
            f"  {strategy:<25}  {m['RAE']:>8.4f}  {m['MAE']:>8.4f}  "
            f"{m['R2']:>8.4f}  {m['Spearman_R']:>9.4f}"
        )

    best_strategy, best_metrics = sorted_results[0]
    print(f"\n  Best (by OOF RAE): {best_strategy} = {best_metrics['RAE']:.4f}")
    print(
        "  LB RAE reference: 0.62 (ens_v7, 2026-04-07 snapshot — "
        "update after next submission)"
    )
    print(
        "\n  Remember: lower OOF RAE may not translate to better LB if "
        "the optimizer overfits weights."
    )


if __name__ == "__main__":
    main()

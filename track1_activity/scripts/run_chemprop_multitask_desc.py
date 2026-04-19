#!/usr/bin/env -S pixi run python
"""ChemProp multi-task with descriptor aux heads (top-N from importance).

Fork of run_chemprop_multitask.py: instead of assay readouts as aux, uses
the top-N physicochemical/Boltz descriptors from
``reports/multitask_aux/top_aux_candidates.csv``. Each aux target is
z-scored over train; main pEC50 is left unscaled. Per-family loss weight
multipliers let us keep the prior "Boltz = high signal" / "Mordred =
regularizer" intuition without having to tune 25 individual knobs.

Inherits NaN masking, predict_pec50 validation, and training loop
plumbing from run_chemprop_multitask. Only data loading and the
task-weight tensor change.

Usage:
    pixi run python track1_activity/scripts/run_chemprop_multitask_desc.py \
        --split umap --n-aux 25 --base-aux-weight 0.04 --use-tuned
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
import torch
from lightning import pytorch as pl

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))

from chemprop import data as chemprop_data  # noqa: E402
from chemprop import models, nn  # noqa: E402
from chemprop.nn.metrics import MSE  # noqa: E402

from data import (  # noqa: E402
    DB_PARAMS,
    JAZZY_FEATURE_COLS,
    get_conn,
    load_jazzy,
    load_mordred,
    load_rdkit_full,
    load_test_smiles,
    load_train_mordred,
    load_train_smiles_target,
)
from evaluate import (  # noqa: E402
    compute_metrics,
    print_fold_summary,
    print_metrics,
    record_experiment,
    save_oof_predictions,
)
from splits import scaffold_split_indices, umap_split_indices  # noqa: E402

# Reuse plumbing
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "scripts")))
from run_chemprop_multitask import (  # noqa: E402
    DEFAULT_PARAMS,
    PEC50_PLAUSIBLE_RANGE,
    TUNED_PARAMS,
    make_dataloader,
    predict_pec50,
)

torch.set_float32_matmul_precision("medium")


SUBMISSION_DIR = REPO_ROOT.joinpath("track1_activity", "submissions")
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

REPORT_DIR = REPO_ROOT.joinpath("track1_activity", "reports", "multitask_aux")
TOP_AUX_CSV = REPORT_DIR.joinpath("top_aux_candidates.csv")

# Compounds to drop from training. Auranofin (cid 1657) is an Au complex
# that Boltz-2 preprocessing rejects, so all Boltz-derived aux targets
# are NaN and the 3D-desc aux columns are zero-imputed -- adds noise
# to the multi-task signal with no way to learn from it. Dropping is
# safe for test-time generalization since no test compound contains Au.
DROP_TRAIN_CIDS = (1657,)

# Family-level loss weight multipliers. Applied on top of --base-aux-weight.
# Rationale:
#   * Boltz tier-0/1 carry the richest PXR-specific signal (distillation
#     from 3D+MSA into 2D graph), so modest up-weight.
#   * 3D families ask the encoder to learn conformer-aware structure, the
#     most ambitious regularization → mild up-weight.
#   * Mordred/RDKit are predominantly graph-derived, so baseline weight.
#   * Jazzy is small-count but physically meaningful → baseline.
FAMILY_WEIGHT_MULT = {
    "boltz_tier0": 1.50,
    "boltz_tier1": 1.50,
    "d3_usrcat": 1.20,
    "d3_mordred3d": 1.20,
    "d3_morse": 1.20,
    "d3_getaway": 1.20,
    "d3_rdf": 1.20,
    "d3_scalar": 1.20,
    "d3_whim": 1.20,
    "d3_usr": 1.20,
    "d3_electroshape": 1.20,
    "d3_autocorr3d": 1.20,
    "mordred": 1.00,
    "rdkit": 1.00,
    "jazzy_pose": 1.00,
    "jazzy_self": 1.00,
}

AGG_REGISTRY = {
    "mean": nn.MeanAggregation,
    "sum": nn.SumAggregation,
    "norm": nn.NormAggregation,
}

BOLTZ_TIER0_COLS = (
    "affinity_pred_value",
    "affinity_pred_value_1",
    "affinity_pred_value_2",
    "affinity_probability_binary",
    "affinity_probability_binary_1",
    "affinity_probability_binary_2",
    "confidence_score",
    "ptm",
    "iptm",
    "ligand_iptm",
    "protein_iptm",
    "complex_plddt",
    "complex_iplddt",
    "complex_pde",
    "complex_ipde",
    "ligand_atom_count",
    "ligand_to_pocket_distance_a",
)


def load_aux_values(feature_full_names: list[str], train_ids: list[int]) -> np.ndarray:
    """Load per-compound values for a list of ``family.column`` features.

    Returns a (N_train, N_aux) float32 matrix. NaNs are preserved so the
    MPNN training step can mask them in the aux loss -- the main pEC50
    column is never produced here, so no NaN-masking concern for task 0.
    """
    by_family: dict[str, list[tuple[int, str]]] = {}
    for i, full in enumerate(feature_full_names):
        fam, _, col = full.partition(".")
        by_family.setdefault(fam, []).append((i, col))

    X = np.full((len(train_ids), len(feature_full_names)), np.nan, dtype=np.float32)

    def _fill(positions: list[tuple[int, str]], df: pd.DataFrame) -> None:
        df = df.reindex(train_ids)
        for i, col in positions:
            if col not in df.columns:
                raise KeyError(f"column '{col}' not found in loaded dataframe")
            X[:, i] = df[col].to_numpy(dtype=np.float32)

    for fam, positions in by_family.items():
        cols = [c for _, c in positions]
        if fam == "mordred":
            df, _ = load_train_mordred()
        elif fam == "rdkit":
            df = load_rdkit_full(train_ids)
        elif fam == "jazzy_self":
            df = load_jazzy(train_ids).reindex(index=train_ids)
        elif fam == "jazzy_pose":
            with psycopg2.connect(**DB_PARAMS) as conn:
                df = pd.read_sql(
                    "SELECT compound_id, sdc, sdx, sa, dga, dgp, dgtot "
                    "FROM compound_boltz2_jazzy",
                    conn,
                ).set_index("compound_id")
        elif fam == "boltz_tier0":
            col_sql = ", ".join(f"b.{c}" for c in BOLTZ_TIER0_COLS)
            with psycopg2.connect(**DB_PARAMS) as conn:
                df = pd.read_sql(
                    f"""
                    SELECT c.id AS compound_id, {col_sql},
                           (b.affinity_pred_value_1 - b.affinity_pred_value_2)
                               AS ensemble_diff_affinity,
                           (b.affinity_probability_binary_1
                              - b.affinity_probability_binary_2)
                               AS ensemble_diff_prob
                    FROM compounds c
                    LEFT JOIN compound_boltz2 b ON b.compound_id = c.id
                    """,
                    conn,
                ).set_index("compound_id")
        elif fam == "boltz_tier1":
            tier1_path = REPO_ROOT.joinpath(
                "data", "boltz2_confidence_features.parquet"
            )
            df = pd.read_parquet(tier1_path)
        elif fam == "d3_mordred3d":
            with psycopg2.connect(**DB_PARAMS) as conn:
                raw = pd.read_sql(
                    "SELECT compound_id, descriptors FROM compound_boltz2_mordred3d",
                    conn,
                ).set_index("compound_id")
            rows = {}
            for cid, rec in raw["descriptors"].items():
                if rec is None or (isinstance(rec, float) and np.isnan(rec)):
                    rows[cid] = {}
                else:
                    rows[cid] = rec
            df = pd.DataFrame(rows).T
        elif fam == "d3_scalar":
            scalar_cols = [
                "asphericity",
                "eccentricity",
                "inertial_shape_factor",
                "npr1",
                "npr2",
                "pmi1",
                "pmi2",
                "pmi3",
                "radius_of_gyration",
                "spherocity_index",
                "pbf",
            ]
            with psycopg2.connect(**DB_PARAMS) as conn:
                df = pd.read_sql(
                    f"SELECT compound_id, {', '.join(scalar_cols)} "
                    "FROM compound_boltz2_desc3d",
                    conn,
                ).set_index("compound_id")
        elif fam.startswith("d3_") and fam != "d3_mordred3d" and fam != "d3_scalar":
            # Vector families: autocorr3d, getaway, morse, rdf, whim, usr,
            # usrcat, electroshape. Each stored as a float8[] column; the
            # importance-CSV column is a synthetic integer suffix (e.g. "64"
            # means position 64 of that vector).
            vec_col = fam.removeprefix("d3_")
            if vec_col == "electroshape":
                table = "compound_boltz2_skfp3d"
            else:
                table = "compound_boltz2_desc3d_vector"
            with psycopg2.connect(**DB_PARAMS) as conn:
                raw = pd.read_sql(
                    f"SELECT compound_id, {vec_col} FROM {table}",
                    conn,
                ).set_index("compound_id")
            raw = raw.reindex(train_ids)
            for i, col_str in positions:
                pos = int(col_str)
                col_vals = []
                for v in raw[vec_col]:
                    if v is None or (isinstance(v, float) and np.isnan(v)):
                        col_vals.append(np.nan)
                    else:
                        arr = np.asarray(v, dtype=np.float64)
                        col_vals.append(float(arr[pos]) if pos < len(arr) else np.nan)
                X[:, i] = np.asarray(col_vals, dtype=np.float32)
            continue
        else:
            raise ValueError(f"Unknown family: {fam}")

        _fill(positions, df)

    return X


def build_model(params: dict, task_weights: torch.Tensor):
    """Like run_chemprop_multitask.build_model but takes a full weights tensor.

    The main task sits at index 0 with weight 1.0; aux indices 1.. carry
    whatever weights the caller chose (per-family multiplier x base).
    """
    mp = nn.BondMessagePassing(
        d_h=params["message_hidden_dim"],
        depth=params["depth"],
        dropout=params["mp_dropout"],
        activation=params["activation"],
    )
    agg = AGG_REGISTRY[params["aggregation"]]()

    criterion = MSE(task_weights=task_weights)
    ffn = nn.RegressionFFN(
        n_tasks=int(task_weights.numel()),
        input_dim=mp.output_dim,
        hidden_dim=params["ffn_hidden_dim"],
        n_layers=params["ffn_num_layers"],
        dropout=params["ffn_dropout"],
        criterion=criterion,
    )

    return models.MPNN(
        message_passing=mp,
        agg=agg,
        predictor=ffn,
        batch_norm=True,
        warmup_epochs=params["warmup_epochs"],
        init_lr=params["learning_rate"],
        max_lr=params["learning_rate"] * params["lr_ratio"],
        final_lr=params["learning_rate"] * 0.1,
    )


def assert_chemprop_masks_nan(params: dict, task_weights: torch.Tensor) -> None:
    """One-batch sanity: NaN aux values must not poison the loss."""
    n_tasks = int(task_weights.numel())
    smiles = ["CCO", "CCC", "CCCO", "CCCC"]
    targets = np.zeros((4, n_tasks), dtype=np.float32)
    targets[:, 0] = [4.0, 5.0, 4.5, 5.5]
    if n_tasks > 1:
        targets[:, 1:] = 0.5
        targets[0, 1:] = np.nan

    loader = make_dataloader(smiles, targets, batch_size=4, shuffle=False)
    model = build_model(params, task_weights)
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
    )
    try:
        trainer.fit(model, loader, loader)
        loss_metric = trainer.callback_metrics.get("train_loss")
        if loss_metric is None or not torch.isfinite(loss_metric).all():
            raise RuntimeError(
                f"ChemProp NaN-mask sanity failed: train_loss={loss_metric}"
            )
    finally:
        del model, trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def train_fold(
    params, train_smiles, train_targets, val_smiles, val_targets, task_weights
):
    train_loader = make_dataloader(
        train_smiles, train_targets, params["batch_size"], shuffle=True
    )
    val_loader = make_dataloader(
        val_smiles, val_targets, params["batch_size"], shuffle=False
    )
    model = build_model(params, task_weights)
    early_stop = pl.callbacks.EarlyStopping(
        monitor="val_loss", patience=params["patience"], mode="min"
    )
    trainer = pl.Trainer(
        max_epochs=params["max_epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        callbacks=[early_stop],
    )
    trainer.fit(model, train_loader, val_loader)
    val_preds = predict_pec50(
        trainer,
        model,
        val_smiles,
        params["batch_size"],
        int(task_weights.numel()),
    )
    return val_preds, model, trainer


def load_compound_ids(split: str) -> list[int]:
    with get_conn() as conn:
        cur = conn.cursor()
        table = "train_activity" if split == "train" else "test_activity"
        cur.execute(f"SELECT compound_id FROM {table} ORDER BY id")
        return [r[0] for r in cur.fetchall()]


def run(args):
    print(
        f"ChemProp multi-task DESC | split={args.split} | n_aux={args.n_aux} "
        f"| base_w={args.base_aux_weight} | max_epochs={args.max_epochs}"
    )

    # Select top-N aux features from the clustering CSV
    if not TOP_AUX_CSV.exists():
        raise FileNotFoundError(
            f"Missing {TOP_AUX_CSV}. Run "
            f"track1_activity/scripts/multitask_aux/{{01,02}}_*.py first."
        )
    top = pd.read_csv(TOP_AUX_CSV)
    if not args.include_boltz_aux:
        # Drop Boltz-distillation-style aux before taking top-N: these cannot
        # be learned from 2D graph alone, cause negative transfer in practice
        # (Optuna #39 confirmed: include_boltz=False beats True by ~0.01 MAE).
        top = top[~top["family"].isin({"boltz_tier0", "boltz_tier1"})]
    top = top.head(args.n_aux).reset_index(drop=True)
    aux_features = top["feature"].tolist()
    aux_families = top["family"].tolist()
    print(f"  Loaded {len(aux_features)} aux features from {TOP_AUX_CSV.name}")

    # Load main pec50 + aux matrix
    train_df = load_train_smiles_target()
    test_df = load_test_smiles()
    train_ids = load_compound_ids("train")
    assert len(train_ids) == len(train_df)

    # Drop compounds whose aux is structurally broken (e.g. Auranofin Au
    # complex rejected by Boltz preprocessing -- all Boltz-derived aux is
    # NaN and 3D-desc columns are zero-imputed).
    keep_mask = np.array([cid not in DROP_TRAIN_CIDS for cid in train_ids], dtype=bool)
    dropped = [cid for cid in train_ids if cid in DROP_TRAIN_CIDS]
    if dropped:
        print(f"  Dropping {len(dropped)} train compound(s): {dropped}")
    train_df = train_df.loc[keep_mask].reset_index(drop=True)
    train_ids = [cid for cid in train_ids if cid not in DROP_TRAIN_CIDS]

    smiles = train_df["smiles"].tolist()
    y_main = train_df["pec50"].to_numpy(dtype=np.float32)
    Xaux = load_aux_values(aux_features, train_ids)

    # z-score aux columns (use nanmean/nanstd to survive sparse NaNs)
    aux_means = np.nanmean(Xaux, axis=0)
    aux_stds = np.nanstd(Xaux, axis=0)
    aux_stds[aux_stds < 1e-6] = 1.0
    Xaux_z = (Xaux - aux_means) / aux_stds

    targets = np.concatenate([y_main.reshape(-1, 1), Xaux_z.astype(np.float32)], axis=1)
    n_tasks = targets.shape[1]

    # Build task weights
    task_weights = np.ones(n_tasks, dtype=np.float32)
    task_weights[0] = 1.0
    for i, fam in enumerate(aux_families, start=1):
        mult = FAMILY_WEIGHT_MULT.get(fam, 1.0)
        task_weights[i] = args.base_aux_weight * mult
    task_weights_t = torch.tensor(task_weights, dtype=torch.float32)

    print(f"  n_tasks={n_tasks}, task_weights:")
    print(f"    [main]                                  pec50  w=1.000")
    for i, (f, fam) in enumerate(zip(aux_features, aux_families), start=1):
        n_nan = int(np.isnan(Xaux[:, i - 1]).sum())
        print(
            f"    [aux {i:>2d}/{n_tasks - 1:>2d}] "
            f"{f[:42]:<42s}  w={task_weights[i]:.3f}  "
            f"nan={n_nan}"
        )

    # NaN main-task guard
    if np.isnan(y_main).any():
        raise ValueError(
            f"Main pEC50 has {int(np.isnan(y_main).sum())} NaN rows -- refusing"
        )

    # CV splits
    if args.split == "scaffold":
        outer = scaffold_split_indices(smiles, n_splits=5, seed=42)
    else:
        outer = umap_split_indices(smiles, n_splits=5, n_clusters=50, seed=42)

    params = (TUNED_PARAMS if args.use_tuned else DEFAULT_PARAMS).copy()
    if args.max_epochs is not None:
        params["max_epochs"] = args.max_epochs

    exp_name = (
        f"chemprop_multitask_desc{args.n_aux}_{args.split}_w{args.base_aux_weight}"
    )
    if not args.include_boltz_aux:
        exp_name += "_noboltz"
    if args.use_tuned:
        exp_name += "_tuned"
    print(f"  Experiment: {exp_name}")

    print("  Running ChemProp NaN-mask sanity check...")
    assert_chemprop_masks_nan(params, task_weights_t)
    print("  Sanity check passed.")

    oof_preds = np.zeros(len(smiles), dtype=np.float32)
    fold_metrics = []
    test_pred_per_fold = []

    for fold, (tr_idx, va_idx) in enumerate(outer):
        print(f"\n[Fold {fold}] train={len(tr_idx)}, val={len(va_idx)}")
        tr_smi = [smiles[i] for i in tr_idx]
        va_smi = [smiles[i] for i in va_idx]
        tr_y = targets[tr_idx]
        va_y = targets[va_idx]

        model = trainer = None
        try:
            val_preds, model, trainer = train_fold(
                params, tr_smi, tr_y, va_smi, va_y, task_weights_t
            )
            if not np.isfinite(val_preds).all():
                raise RuntimeError(
                    f"Fold {fold}: val_preds contain "
                    f"{int((~np.isfinite(val_preds)).sum())} NaN/Inf values"
                )
            oof_preds[va_idx] = val_preds

            metrics = compute_metrics(va_y[:, 0], val_preds)
            fold_metrics.append(metrics)
            print_metrics(metrics, label=f"Fold {fold}")

            test_preds = predict_pec50(
                trainer,
                model,
                test_df["smiles"].tolist(),
                params["batch_size"],
                n_tasks,
            )
            if not np.isfinite(test_preds).all():
                raise RuntimeError(f"Fold {fold}: test_preds contain NaN")
            test_pred_per_fold.append(test_preds)
        finally:
            del model, trainer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    oof_metrics = compute_metrics(targets[:, 0], oof_preds)
    print("\n  Overall OOF:")
    print_metrics(oof_metrics)
    print_fold_summary(fold_metrics)

    test_preds_mean = np.mean(test_pred_per_fold, axis=0)
    if not np.isfinite(test_preds_mean).all():
        raise RuntimeError("test_preds_mean contains NaN/Inf")
    print(
        f"\n  Test preds: mean={test_preds_mean.mean():.3f}, "
        f"std={test_preds_mean.std():.3f}"
    )

    sub = pd.DataFrame(
        {
            "SMILES": test_df["smiles"],
            "Molecule Name": test_df["molecule_name"],
            "pEC50": test_preds_mean,
        }
    )
    sub_path = SUBMISSION_DIR.joinpath(f"{exp_name}.csv")
    sub_tmp = sub_path.with_suffix(sub_path.suffix + ".tmp")
    sub.to_csv(sub_tmp, index=False)

    try:
        exp_id = record_experiment(
            name=exp_name,
            description=(
                f"ChemProp multi-task DESC ({args.n_aux} aux heads) "
                f"on {args.split} split, base_aux_w={args.base_aux_weight}"
            ),
            model_type="chemprop",
            feature_set="smiles",
            hyperparameters={
                **params,
                "base_aux_weight": args.base_aux_weight,
                "n_tasks": n_tasks,
                "aux_features": aux_features,
                "aux_families": aux_families,
                "family_weight_mult": {
                    fam: FAMILY_WEIGHT_MULT.get(fam, 1.0) for fam in set(aux_families)
                },
            },
            fold_metrics=fold_metrics,
            submission_path=f"track1_activity/submissions/{exp_name}.csv",
            num_boost_rounds=[0] * len(fold_metrics),
            notes=(
                f"OOF RAE={oof_metrics['RAE']:.4f}, "
                f"n_aux={args.n_aux}, base_aux_weight={args.base_aux_weight}"
            ),
        )
    except Exception:
        sub_tmp.unlink(missing_ok=True)
        raise

    sub_tmp.replace(sub_path)
    print(f"  Saved submission: {sub_path}")
    save_oof_predictions(exp_id, oof_preds)
    print(f"\n  Done: {exp_name} -> RAE={oof_metrics['RAE']:.4f}")
    return oof_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="ChemProp descriptor-aux multitask")
    parser.add_argument("--split", choices=["scaffold", "umap"], default="umap")
    parser.add_argument("--n-aux", type=int, default=25)
    parser.add_argument("--base-aux-weight", type=float, default=0.04)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--use-tuned", action="store_true")
    parser.add_argument(
        "--no-boltz-aux",
        dest="include_boltz_aux",
        action="store_false",
        help="Drop boltz_tier0/tier1 from aux target set (Optuna-best config)",
    )
    parser.set_defaults(include_boltz_aux=True)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

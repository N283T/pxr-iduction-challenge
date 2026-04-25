"""Stacker bakeoff: 2-stage ensembles over the 9-member pool.

Current caruana_bag20 is capped at OOF MAE ~0.4150 because it is a
linear non-negative average that cannot reallocate weight across
correlated members without zero-summing (see issue #82 + 5-case
OOF -0.002 ceiling memory). Stacking aims to break this by letting a
second-stage learner apply compound-conditional weighting.

For each meta-learner we run 5-fold outer CV (same UMAP seed=42 as
pool members, so the stacker train/val split is leak-free wrt the
member OOFs) and report OOF MAE / RAE / Spearman.

Meta-learners evaluated (Phase 1):
  simple_avg       uniform 1/9 (pool sanity check, no learning)
  linear           sklearn LinearRegression, no constraint
  ridge            sklearn Ridge, cv-tuned alpha (1e-4..10)
  lgbm_default     LightGBM mae objective, default hyperparams
  lgbm_optuna      LightGBM mae objective, 20-trial Optuna
  tabpfn           TabPFN v7 (Bayesian on 9d input, small tabular)

Reference baselines shown alongside:
  caruana_bag20    current production weight scheme (~0.4150)

Phase 2 (context meta-features) and Phase 3 (calibration + LB) are
separate scripts; this file only runs the stage-2 regression.

Usage:
    pixi run python track1_activity/scripts/run_stacker.py
    pixi run python track1_activity/scripts/run_stacker.py --learners linear,lgbm_default
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "scripts")))
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))

from data import load_train_smiles_target  # noqa: E402
from evaluate import compute_metrics  # noqa: E402
from splits import umap_split_indices  # noqa: E402

import run_ensemble  # noqa: E402


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_pool_oof() -> tuple[list[str], np.ndarray, np.ndarray, list[str]]:
    """Return (member names, OOF matrix [n_train, n_pool], y_train, smiles list).

    Reuses run_ensemble.load_models so the allow-list stays authoritative.
    """
    train_df = load_train_smiles_target()
    y_arr = train_df["pec50"].to_numpy(dtype=np.float64)
    smiles = train_df["smiles"].tolist()
    names, oof, _ = run_ensemble.load_models(y_arr, n_test=513)
    return names, oof, y_arr, smiles


# ---------------------------------------------------------------------------
# Meta features (Phase 2: context-aware stacking)
# ---------------------------------------------------------------------------

# Physprop descriptors selected for stacker meta features. Small set
# (8 cols) to avoid overfit on 4140 rows; lipophilicity-heavy because
# logP dominates feature importance in single-feature LGBMs on PXR.
PHYSPROP_COLS: tuple[str, ...] = (
    "logp",
    "tpsa",
    "hba",
    "hbd",
    "num_heavy_atoms",
    "num_aromatic_rings",
    "fractioncsp3",
    "amw",
)


def load_physprop(n_train: int) -> np.ndarray:
    """Physprop descriptors for train_activity rows, ordered by train_activity.id."""
    import psycopg2

    from data import DB_PARAMS

    cols = ", ".join(f"cd.{c}" for c in PHYSPROP_COLS)
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    cur.execute(
        f"""SELECT {cols}
           FROM train_activity t
           JOIN compound_descriptors cd ON cd.compound_id = t.compound_id
           ORDER BY t.id"""
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    arr = np.array(rows, dtype=np.float64)
    if arr.shape[0] != n_train:
        raise RuntimeError(
            f"Physprop row count {arr.shape[0]} != train rows {n_train}; "
            "check compound_descriptors coverage."
        )
    # Replace any NaN with column median (metal-containing compounds
    # occasionally miss descriptors; column median is a safe imputation).
    for j in range(arr.shape[1]):
        col = arr[:, j]
        nan_mask = ~np.isfinite(col)
        if nan_mask.any():
            median = float(np.nanmedian(col))
            col[nan_mask] = median
            arr[:, j] = col
    return arr


def load_physprop_test(n_test: int) -> np.ndarray:
    import psycopg2

    from data import DB_PARAMS

    cols = ", ".join(f"cd.{c}" for c in PHYSPROP_COLS)
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    cur.execute(
        f"""SELECT {cols}
           FROM test_activity t
           JOIN compound_descriptors cd ON cd.compound_id = t.compound_id
           ORDER BY t.id"""
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    arr = np.array(rows, dtype=np.float64)
    if arr.shape[0] != n_test:
        raise RuntimeError(
            f"Physprop test row count {arr.shape[0]} != test rows {n_test}."
        )
    for j in range(arr.shape[1]):
        col = arr[:, j]
        nan_mask = ~np.isfinite(col)
        if nan_mask.any():
            median = float(np.nanmedian(col))
            col[nan_mask] = median
            arr[:, j] = col
    return arr


def pred_disagreement_features(oof_mat: np.ndarray) -> np.ndarray:
    """4 row-wise summary statistics of the 9 pool predictions.

    These are pure functions of the OOF matrix, no label / split info.
    """
    q25 = np.quantile(oof_mat, 0.25, axis=1)
    q75 = np.quantile(oof_mat, 0.75, axis=1)
    feats = np.column_stack(
        [
            oof_mat.mean(axis=1),
            oof_mat.std(axis=1),
            oof_mat.max(axis=1) - oof_mat.min(axis=1),
            q75 - q25,
        ]
    )
    return feats


def build_meta_features(
    oof_mat: np.ndarray,
    n: int,
    split: str,
    *,
    kind: str = "physprop",
    smiles: list[str] | None = None,
    train_smiles: list[str] | None = None,
    n_clusters: int = 50,
    umap_seed: int = 42,
) -> np.ndarray:
    """Meta-feature matrix appended to the pool OOF columns.

    ``kind`` selects which bundle:
      physprop   4 disagreement + 8 physprop                = 12 cols
      disagree   4 disagreement only                        =  4 cols
      cluster    4 disagreement + 50 UMAP cluster one-hot   = 54 cols
                 (clusters use the same UMAP+Morgan+Jaccard recipe as
                 ``splits.umap_split_indices``; pool members have no
                 explicit cluster-ID feature, so this is net-new signal.)

    ``split`` is ``"train"`` or ``"test"`` — controls SQL source when
    physprop is pulled and determines which SMILES list UMAP fits on.
    For ``kind="cluster"`` on ``split="test"`` we reuse the UMAP
    embedding fit on the train SMILES and ``.transform`` the test
    SMILES so the cluster IDs are consistent.
    """
    disagree = pred_disagreement_features(oof_mat)
    if kind == "disagree":
        return disagree
    if kind == "physprop":
        if split == "train":
            physprop = load_physprop(n)
        elif split == "test":
            physprop = load_physprop_test(n)
        else:
            raise ValueError(f"Unknown split: {split}")
        return np.concatenate([disagree, physprop], axis=1)
    if kind == "cluster":
        if smiles is None:
            raise ValueError("kind='cluster' requires smiles argument")
        cluster_oh = _umap_cluster_onehot(
            smiles,
            n_clusters=n_clusters,
            seed=umap_seed,
            train_smiles=train_smiles,
        )
        return np.concatenate([disagree, cluster_oh], axis=1)
    raise ValueError(f"Unknown meta kind: {kind}")


def _umap_cluster_onehot(
    smiles: list[str],
    *,
    n_clusters: int,
    seed: int,
    train_smiles: list[str] | None = None,
) -> np.ndarray:
    """Project SMILES onto ``n_clusters`` UMAP+KMeans buckets, return one-hot.

    Reuses the canonical UMAP+Morgan+Jaccard recipe from
    ``splits.umap_split_indices`` so the clusters align with the
    production CV geometry. The UMAP embedding is fit on ``train_smiles``
    when provided (test-time branch); otherwise fit on ``smiles``
    themselves (train branch).
    """
    import umap
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    from sklearn.cluster import KMeans

    fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    def _fp_matrix(smi_list: list[str]) -> np.ndarray:
        rows = []
        for s in smi_list:
            mol = Chem.MolFromSmiles(s)
            rows.append(fp_gen.GetFingerprintAsNumPy(mol).astype(np.uint8))
        return np.asarray(rows, dtype=np.uint8)

    fit_smiles = train_smiles if train_smiles is not None else smiles
    fit_fp = _fp_matrix(fit_smiles)
    reducer = umap.UMAP(
        n_components=16,
        metric="jaccard",
        n_neighbors=15,
        min_dist=0.1,
        random_state=seed,
    )
    emb_fit = reducer.fit_transform(fit_fp)
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    kmeans.fit(emb_fit)
    if train_smiles is None:
        labels = kmeans.labels_
    else:
        emb_query = reducer.transform(_fp_matrix(smiles))
        labels = kmeans.predict(emb_query)
    # one-hot
    oh = np.zeros((len(labels), n_clusters), dtype=np.float32)
    oh[np.arange(len(labels)), labels] = 1.0
    return oh


# ---------------------------------------------------------------------------
# Meta-learners
# ---------------------------------------------------------------------------


def fit_predict(
    learner: str,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    seed: int = 42,
) -> np.ndarray:
    """Train the named meta-learner on (X_tr, y_tr), predict X_val."""
    if learner == "simple_avg":
        return X_val.mean(axis=1)

    if learner == "linear":
        from sklearn.linear_model import LinearRegression

        model = LinearRegression()
        model.fit(X_tr, y_tr)
        return model.predict(X_val)

    if learner == "ridge":
        from sklearn.linear_model import RidgeCV

        model = RidgeCV(alphas=np.logspace(-4, 1, 30))
        model.fit(X_tr, y_tr)
        return model.predict(X_val)

    if learner == "lgbm_default":
        import lightgbm as lgb

        params = {
            "objective": "regression",
            "metric": "mae",
            "verbose": -1,
            "seed": seed,
            "num_leaves": 15,
            "learning_rate": 0.03,
            "feature_fraction": 1.0,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_child_samples": 20,
            "lambda_l1": 0.01,
            "lambda_l2": 1.0,
        }
        dtrain = lgb.Dataset(X_tr, label=y_tr)
        model = lgb.train(params, dtrain, num_boost_round=500)
        return model.predict(X_val)

    if learner == "lgbm_optuna":
        import lightgbm as lgb
        import optuna
        from sklearn.model_selection import KFold

        def objective(trial):
            params = {
                "objective": "regression",
                "metric": "mae",
                "verbose": -1,
                "seed": seed,
                "num_leaves": trial.suggest_int("num_leaves", 7, 31),
                "learning_rate": trial.suggest_float("lr", 0.01, 0.15, log=True),
                "feature_fraction": trial.suggest_float("ff", 0.5, 1.0),
                "bagging_fraction": trial.suggest_float("bf", 0.5, 1.0),
                "bagging_freq": trial.suggest_int("bq", 1, 10),
                "min_child_samples": trial.suggest_int("mcs", 5, 80),
                "lambda_l1": trial.suggest_float("l1", 1e-6, 1.0, log=True),
                "lambda_l2": trial.suggest_float("l2", 1e-6, 5.0, log=True),
            }
            kf = KFold(n_splits=3, shuffle=True, random_state=seed)
            maes = []
            for tr_idx, va_idx in kf.split(X_tr):
                dtr = lgb.Dataset(X_tr[tr_idx], label=y_tr[tr_idx])
                dva = lgb.Dataset(X_tr[va_idx], label=y_tr[va_idx])
                model = lgb.train(
                    params,
                    dtr,
                    num_boost_round=600,
                    valid_sets=[dva],
                    callbacks=[lgb.early_stopping(30, verbose=False)],
                )
                pred = model.predict(X_tr[va_idx])
                maes.append(float(np.mean(np.abs(pred - y_tr[va_idx]))))
            return float(np.mean(maes))

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=seed),
        )
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=20, show_progress_bar=False)
        best = dict(study.best_params)
        params = {
            "objective": "regression",
            "metric": "mae",
            "verbose": -1,
            "seed": seed,
            "num_leaves": best["num_leaves"],
            "learning_rate": best["lr"],
            "feature_fraction": best["ff"],
            "bagging_fraction": best["bf"],
            "bagging_freq": best["bq"],
            "min_child_samples": best["mcs"],
            "lambda_l1": best["l1"],
            "lambda_l2": best["l2"],
        }
        dtrain = lgb.Dataset(X_tr, label=y_tr)
        model = lgb.train(params, dtrain, num_boost_round=500)
        return model.predict(X_val)

    if learner == "tabpfn":
        from tabpfn import TabPFNRegressor

        model = TabPFNRegressor(
            n_estimators=8,
            device="cuda",
            softmax_temperature=0.9,
            random_state=seed,
            ignore_pretraining_limits=True,
        )
        model.fit(X_tr, y_tr)
        return model.predict(X_val)

    raise ValueError(f"Unknown learner: {learner}")


# ---------------------------------------------------------------------------
# CV runner
# ---------------------------------------------------------------------------


def outer_cv(
    learner: str,
    oof_mat: np.ndarray,
    y: np.ndarray,
    smiles: list[str],
    seed: int = 42,
    n_splits: int = 5,
    n_clusters: int = 50,
) -> tuple[dict, np.ndarray]:
    """Run outer 5-fold CV using the canonical UMAP+Morgan+Jaccard split.

    Returns (mean metrics dict, OOF prediction vector).
    """
    splits = umap_split_indices(
        smiles,
        n_splits=n_splits,
        n_clusters=n_clusters,
        seed=seed,
    )

    oof_pred = np.full_like(y, np.nan, dtype=np.float64)
    fold_metrics: list[dict] = []

    for fold, (tr_idx, va_idx) in enumerate(splits):
        X_tr = oof_mat[tr_idx]
        y_tr = y[tr_idx]
        X_val = oof_mat[va_idx]
        y_val = y[va_idx]

        pred = fit_predict(learner, X_tr, y_tr, X_val, seed=seed)
        oof_pred[va_idx] = pred

        m = compute_metrics(y_val, pred)
        fold_metrics.append(m)
        print(
            f"  [{learner:14s} Fold {fold}] "
            f"MAE={m['MAE']:.4f} RAE={m['RAE']:.4f} "
            f"R2={m['R2']:.4f} Sp={m['Spearman_R']:.4f}"
        )

    # Overall metrics on concatenated OOF
    overall = compute_metrics(y, oof_pred)
    mean_metrics = {
        k: float(np.mean([m[k] for m in fold_metrics])) for k in fold_metrics[0]
    }
    mean_metrics_std = {
        k: float(np.std([m[k] for m in fold_metrics])) for k in fold_metrics[0]
    }
    return overall, mean_metrics, mean_metrics_std, oof_pred


def caruana_baseline_oof(
    oof_mat: np.ndarray, y: np.ndarray, seed: int = 42
) -> tuple[dict, np.ndarray]:
    """Reproduce caruana_bag20 on the same OOF matrix for a fair baseline.

    Unlike the learners above, caruana does NOT split the OOF matrix into
    train/val — it directly fits weights against y on the full matrix.
    That is the same protocol run_ensemble uses, and it is what we
    compare against.
    """
    from run_ensemble import optimize_caruana

    w = optimize_caruana(oof_mat, y, n_bags=20, seed=seed)
    pred = oof_mat @ w
    overall = compute_metrics(y, pred)
    return overall, w, pred


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--learners",
        default="simple_avg,linear,ridge,lgbm_default,lgbm_optuna,tabpfn",
        help="Comma-separated meta-learners to evaluate",
    )
    parser.add_argument(
        "--meta",
        choices=["off", "disagree", "physprop", "cluster"],
        default="off",
        help="Meta-feature bundle to append. 'off' is Phase 1 (9-col OOF "
        "only), others are Phase 2 variants. See build_meta_features.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    learners = [s.strip() for s in args.learners.split(",") if s.strip()]

    print("Loading pool OOFs...")
    names, oof_mat, y, smiles = load_pool_oof()
    print(f"Pool: {len(names)} members, OOF shape {oof_mat.shape}, y {y.shape}")

    pool_only = oof_mat  # always keep 9-col pool view for caruana baseline
    for i, n in enumerate(names):
        single_mae = float(np.mean(np.abs(pool_only[:, i] - y)))
        print(f"  [{i}] {n[:60]:60s} MAE={single_mae:.4f}")

    if args.meta != "off":
        meta = build_meta_features(
            pool_only,
            n=len(y),
            split="train",
            kind=args.meta,
            smiles=smiles,
        )
        oof_mat = np.concatenate([pool_only, meta], axis=1)
        print(
            f"+meta[{args.meta}]: added {meta.shape[1]} cols "
            f"→ total features {oof_mat.shape[1]}"
        )

    # Caruana reference (same protocol as run_ensemble.main) — always on
    # the 9-column pool view, even when --meta augments the stacker input.
    print("\n===== Baseline: caruana_bag20 (no outer CV) =====")
    overall, w, _ = caruana_baseline_oof(pool_only, y, seed=args.seed)
    print(
        f"  caruana_bag20  MAE={overall['MAE']:.4f}  RAE={overall['RAE']:.4f}  "
        f"Sp={overall['Spearman_R']:.4f}"
    )
    print("  weights:", {names[i][:40]: round(float(w[i]), 3) for i in range(len(w))})
    caruana_mae = overall["MAE"]

    # Stacker outer CV
    results: list[tuple[str, dict, dict, dict]] = []
    for learner in learners:
        print(f"\n===== {learner} (outer 5-fold UMAP split, seed={args.seed}) =====")
        overall, mean_m, std_m, _ = outer_cv(
            learner, oof_mat, y, smiles, seed=args.seed
        )
        print(
            f"  overall OOF  MAE={overall['MAE']:.4f}  RAE={overall['RAE']:.4f}  "
            f"Sp={overall['Spearman_R']:.4f}"
        )
        print(
            f"  fold mean    MAE={mean_m['MAE']:.4f}±{std_m['MAE']:.4f}  "
            f"RAE={mean_m['RAE']:.4f}±{std_m['RAE']:.4f}"
        )
        results.append((learner, overall, mean_m, std_m))

    # Summary
    print("\n===== Summary (overall OOF, lower MAE is better) =====")
    print(
        f"  {'learner':>14s}  {'MAE':>7s}  {'Δ vs caruana':>14s}  "
        f"{'RAE':>7s}  {'Sp':>7s}"
    )
    print(
        f"  {'caruana_bag20':>14s}  {caruana_mae:>7.4f}  {'—':>14s}  "
        f"{overall['RAE']:>7.4f}  {overall['Spearman_R']:>7.4f}"
    )
    for name, overall, _, _ in results:
        delta = overall["MAE"] - caruana_mae
        gate = "PASS" if delta <= -0.003 else "fail"
        print(
            f"  {name:>14s}  {overall['MAE']:>7.4f}  "
            f"{delta:>+8.4f} [{gate:>4s}]  "
            f"{overall['RAE']:>7.4f}  {overall['Spearman_R']:>7.4f}"
        )


if __name__ == "__main__":
    main()

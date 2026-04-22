"""Mixed-source feature compression + TabPFN bakeoff.

Concatenates the feature bases of all current pool members (7 sources)
into an 8375-dim mega-feature, then reduces per-fold by LGBM gain to
top-500 and runs TabPFN. Motivation: LGBM selects the strongest signals
across different feature families (Mordred descriptors + chemeleon
foundation FP + Boltz trunk + chemprop/KERMT/MoLFormer/AttentiveFP/
GatedGCN learned embeddings), potentially extracting cross-family
synergy that no single feature-source top-K captures.

Feature concatenation order (8375d total):
  cheme_2d_full_boltz_log2fc_pred : 2103
  chemprop_pretrain_embed         :  256
  pooled_boltz_allpairs           : 1024
  molformer_c3_pretrain_embed     :  768
  kermt_pretrain_embed            : 3200
  attentivefp_pretrain_embed      :  512
  gatedgcn_pretrain_embed         :  512
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import psycopg2
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import mean_absolute_error, r2_score

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "scripts")))

from data import DB_PARAMS  # noqa: E402
from evaluate import record_experiment, save_oof_predictions  # noqa: E402
from splits import umap_split_indices  # noqa: E402

SOURCES = [
    "cheme_2d_full_boltz_log2fc_pred",
    "chemprop_pretrain_embed",
    "pooled_boltz_allpairs",
    "molformer_c3_pretrain_embed",
    "kermt_pretrain_embed",
    "attentivefp_pretrain_embed",
    "gatedgcn_pretrain_embed",
]

SUBMISSION_DIR = REPO_ROOT.joinpath("track1_activity", "submissions")


def rae(y_true, y_pred):
    num = np.sum(np.abs(y_true - y_pred))
    den = np.sum(np.abs(y_true - np.mean(y_true)))
    return float(num / den) if den > 0 else float("nan")


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RAE": rae(y_true, y_pred),
        "R2": float(r2_score(y_true, y_pred)),
        "Spearman_R": float(spearmanr(y_pred, y_true).statistic),
        "Kendall_Tau": float(kendalltau(y_pred, y_true).statistic),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from tabpfn import TabPFNRegressor

    import run_train

    with psycopg2.connect(**DB_PARAMS) as conn:
        tr_df = pd.read_sql(
            """SELECT t.id AS compound_id, t.pec50, c.std_smiles AS smiles, c.molecule_name
                 FROM train_activity t JOIN compounds c ON c.id = t.id
                 ORDER BY t.id""",
            conn,
        )
        te_df = pd.read_sql(
            """SELECT t.id AS compound_id, c.std_smiles AS smiles, c.molecule_name
                 FROM test_activity t JOIN compounds c ON c.id = t.id
                 ORDER BY t.id""",
            conn,
        )

    y = tr_df["pec50"].to_numpy(dtype=np.float32)

    # Load all sources and track slice boundaries
    X_tr_blocks, X_te_blocks, block_info = [], [], []
    for src in SOURCES:
        Xtr, Xte = run_train.load_features(src, tr_df, te_df)
        col_mean = np.nanmean(Xtr, axis=0)
        col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
        Xtr = np.where(np.isfinite(Xtr), Xtr, col_mean).astype(np.float32)
        Xte = np.where(np.isfinite(Xte), Xte, col_mean).astype(np.float32)
        d = Xtr.shape[1]
        block_info.append((src, d))
        X_tr_blocks.append(Xtr)
        X_te_blocks.append(Xte)
        print(f"  loaded {src}: {d}d")

    X_tr_full = np.concatenate(X_tr_blocks, axis=1)
    X_te_full = np.concatenate(X_te_blocks, axis=1)
    d_total = X_tr_full.shape[1]
    print(
        f"\nConcatenated: {d_total}d across {len(SOURCES)} sources\n"
        f"  train {X_tr_full.shape[0]}, test {X_te_full.shape[0]}"
    )

    splits = umap_split_indices(
        tr_df["smiles"].tolist(), n_splits=5, n_clusters=50, seed=args.seed
    )

    oof = np.zeros(len(y), dtype=np.float32)
    test_preds_per_fold: list[np.ndarray] = []
    fold_metrics: list[dict] = []
    # Track which source contributed how many of the top-K selections
    source_hits = {s: 0 for s, _ in block_info}
    # Cumulative block indices for source-attribution
    block_ends = np.cumsum([d for _, d in block_info])

    def src_of_idx(i: int) -> str:
        for j, end in enumerate(block_ends):
            if i < end:
                return block_info[j][0]
        return "?"

    for fold, (tr_idx, va_idx) in enumerate(splits):
        lgbm = lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=10,
            random_state=args.seed,
            verbose=-1,
        )
        lgbm.fit(X_tr_full[tr_idx], y[tr_idx])
        gain = lgbm.booster_.feature_importance(importance_type="gain")
        sel = np.argsort(-gain)[: args.K]

        for i in sel:
            source_hits[src_of_idx(int(i))] += 1

        reg = TabPFNRegressor(
            device="cuda",
            n_estimators=8,
            random_state=args.seed,
            ignore_pretraining_limits=False,
        )
        reg.fit(X_tr_full[tr_idx][:, sel], y[tr_idx])
        oof[va_idx] = reg.predict(X_tr_full[va_idx][:, sel])
        test_preds_per_fold.append(reg.predict(X_te_full[:, sel]))

        m = compute_metrics(y[va_idx], oof[va_idx])
        fold_metrics.append(m)
        print(
            f"  [Fold {fold}] d_sel={args.K}  MAE={m['MAE']:.4f} "
            f"RAE={m['RAE']:.4f} Sp={m['Spearman_R']:.4f}"
        )

    overall = compute_metrics(y, oof)
    print(f"\nOverall OOF (mixed compression top-{args.K}):")
    for k, v in overall.items():
        print(f"  {k}={v:.4f}")

    # Source-attribution summary (cumulative across 5 folds, so 2500 total picks)
    total_picks = sum(source_hits.values())
    print(f"\nSource attribution (5-fold cumulative, {total_picks} total picks):")
    for src, d in block_info:
        pct = source_hits[src] / total_picks * 100
        print(f"  {src:40s}  d={d:<5d}  picks={source_hits[src]:5d}  share={pct:5.1f}%")

    # Compare against top references
    refs = [
        "tabpfn_cheme_2d_full_boltz_log2fc_pred_top500_umap",
        "tabpfn_cheme_2d_full_boltz_log2fc_pred_umap_default",
    ]
    print("\nReferences:")
    with psycopg2.connect(**DB_PARAMS) as conn, conn.cursor() as cur:
        for ref in refs:
            cur.execute(
                "SELECT mae_mean, spearman_mean FROM experiment_summary WHERE name=%s",
                (ref,),
            )
            row = cur.fetchone()
            if row:
                ref_mae = float(row[0])
                print(f"  {ref}: MAE={ref_mae:.4f}  Δ={overall['MAE'] - ref_mae:+.4f}")

    # Save experiment + submission
    test_preds_mean = np.mean(np.stack(test_preds_per_fold), axis=0)
    exp_name = f"tabpfn_mixed_pool_top{args.K}_umap"
    sub = pd.DataFrame(
        {
            "SMILES": te_df["smiles"],
            "Molecule Name": te_df["molecule_name"],
            "pEC50": test_preds_mean,
        }
    )
    sub_path = SUBMISSION_DIR.joinpath(f"{exp_name}.csv")
    sub.to_csv(sub_path, index=False)
    print(f"\nSaved: {sub_path}")

    exp_id = record_experiment(
        name=exp_name,
        description=(
            f"TabPFN on top-{args.K} LGBM-gain features from the 7-source "
            f"concatenated pool feature space ({d_total}d total), per-fold "
            f"selection, UMAP 5-fold CV."
        ),
        model_type="tabpfn",
        feature_set="mixed_pool_sources",
        hyperparameters={
            "K": args.K,
            "sources": SOURCES,
            "d_total": d_total,
            "lgbm_n_estimators": 500,
            "lgbm_learning_rate": 0.05,
            "seed": args.seed,
        },
        fold_metrics=fold_metrics,
        submission_path=f"track1_activity/submissions/{exp_name}.csv",
        notes=(
            f"Mixed compression top-{args.K} from {d_total}d pool-concat. "
            f"OOF MAE={overall['MAE']:.4f} RAE={overall['RAE']:.4f} "
            f"Sp={overall['Spearman_R']:.4f}. "
            f"Source shares (5-fold sum): "
            + ", ".join(f"{s.split('_')[0]}:{source_hits[s]}" for s in SOURCES)
        ),
    )
    save_oof_predictions(exp_id, oof)
    print(f"Recorded experiment id={exp_id}: {exp_name}")


if __name__ == "__main__":
    main()

#!/usr/bin/env -S pixi run python
"""Run Phase 2 train+AS1 OOF for current TabPFN ensemble members.

The outputs are development artifacts for the Phase 2 validation matrix, not
submission CSVs. Each row is predicted out-of-fold from a model that did not see
that labeled row during training.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "scripts")))

from build_phase2_validation_matrix import (  # noqa: E402
    OUT_DIR,
    TRUE_BIN_LABELS,
    build_labeled_pool,
    build_phase2_feature_matrix,
    build_phase2_splits,
)

MEMBER_OUT_DIR = OUT_DIR / "tabpfn_members"
SCOREBOARD_SLICES = ["all", "source_as1", "true_lt3", "true_gte6"]


@dataclass(frozen=True)
class MemberConfig:
    name: str
    feature: str
    top_k: int | None = None


MEMBERS = [
    MemberConfig(
        name="tabpfn_cheme_2d_full_boltz_log2fc_pred_optuna_trial10_seed5ens_umap_default",
        feature="cheme_2d_full_boltz_log2fc_pred_optuna_trial10_seed5ens",
    ),
    MemberConfig(
        name="tabpfn_chemprop_pretrain_embed_umap_default",
        feature="chemprop_pretrain_embed",
    ),
    MemberConfig(
        name="tabpfn_pooled_boltz_umap_default",
        feature="pooled_boltz",
    ),
    MemberConfig(
        name="tabpfn_pooled_boltz_allpairs_umap_default",
        feature="pooled_boltz_allpairs",
    ),
    MemberConfig(
        name="tabpfn_molformer_c3_pretrain_embed_umap",
        feature="molformer_c3_pretrain_embed",
    ),
    MemberConfig(
        name="tabpfn_kermt_pretrain_embed_umap_default",
        feature="kermt_pretrain_embed",
    ),
    MemberConfig(
        name="tabpfn_attentivefp_pretrain_embed_umap_default",
        feature="attentivefp_pretrain_embed",
    ),
    MemberConfig(
        name="tabpfn_gatedgcn_pretrain_embed_umap_default",
        feature="gatedgcn_pretrain_embed",
    ),
    MemberConfig(
        name="tabpfn_cheme_2d_full_boltz_log2fc_pred_optuna_trial10_seed5ens_top500_umap",
        feature="cheme_2d_full_boltz_log2fc_pred_optuna_trial10_seed5ens",
        top_k=500,
    ),
    MemberConfig(
        name="tabpfn_cheme_2d_full_boltz_log2fc_pred_seed10ens_top500_umap",
        feature="cheme_2d_full_boltz_log2fc_pred_seed10ens",
        top_k=500,
    ),
]


def metric_row(y: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    err = pred - y
    spearman = stats.spearmanr(y, pred).statistic if len(y) >= 2 else np.nan
    return {
        "n": int(len(y)),
        "mae": float(np.mean(np.abs(err))),
        "bias_pred_minus_true": float(np.mean(err)),
        "spearman": float(spearman),
        "pred_mean": float(np.mean(pred)),
        "true_mean": float(np.mean(y)),
    }


def summarize_oof(oof: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    rows = []
    masks = {
        "all": pd.Series(True, index=oof.index),
        "source_train": oof["source"].eq("train"),
        "source_as1": oof["source"].eq("as1"),
        "true_lt3": oof["pec50"] < 3.0,
        "true_gte6": oof["pec50"] >= 6.0,
    }
    for label in TRUE_BIN_LABELS:
        masks[f"bin_{label}"] = oof["true_bin"].eq(label)
    for name, mask in masks.items():
        sub = oof.loc[mask]
        rows.append(
            {
                "slice": name,
                **metric_row(
                    sub["pec50"].to_numpy(dtype=np.float64),
                    sub[pred_col].to_numpy(dtype=np.float64),
                ),
            }
        )
    return pd.DataFrame(rows)


def select_topk_per_fold(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    top_k: int,
    seed: int,
) -> np.ndarray:
    ranker = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=10,
        random_state=seed,
        verbose=-1,
    )
    ranker.fit(X[train_idx], y[train_idx])
    gain = ranker.booster_.feature_importance(importance_type="gain")
    return np.argsort(-gain)[:top_k]


def run_member(
    member: MemberConfig,
    pool: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion

    version_enum = {
        "v3": ModelVersion.V3,
        "v2_6": ModelVersion.V2_6,
        "v2_5": ModelVersion.V2_5,
        "v2": ModelVersion.V2,
    }[args.tabpfn_version]
    model_path = TabPFNRegressor.create_default_for_version(version_enum).model_path

    print(f"\n=== {member.name} ===")
    print(f"feature={member.feature} top_k={member.top_k}")
    X = build_phase2_feature_matrix(member.feature, pool)
    y = pool["pec50"].to_numpy(dtype=np.float32)
    oof_pred = np.full(len(pool), np.nan, dtype=np.float64)
    fold_rows = []

    for fold, (train_idx, val_idx) in enumerate(splits):
        if member.top_k is None:
            selected = np.arange(X.shape[1], dtype=np.int64)
        else:
            selected = select_topk_per_fold(
                X, y, train_idx, member.top_k, seed=args.seed + fold
            )

        print(
            f"fold {fold}: train={len(train_idx)} val={len(val_idx)} "
            f"d={len(selected)} as1_val={int((pool.iloc[val_idx]['source'] == 'as1').sum())}"
        )
        model = TabPFNRegressor(
            device=args.device,
            n_estimators=args.n_estimators,
            softmax_temperature=args.softmax_temperature,
            average_before_softmax=args.average_before_softmax,
            random_state=args.seed + fold,
            model_path=model_path,
            ignore_pretraining_limits=len(selected) > 500,
        )
        model.fit(X[train_idx][:, selected], y[train_idx])
        pred = model.predict(X[val_idx][:, selected]).astype(np.float64)
        oof_pred[val_idx] = pred
        fold_rows.append(
            {
                "member": member.name,
                "fold": fold,
                "feature": member.feature,
                "top_k": member.top_k or 0,
                **metric_row(y[val_idx].astype(np.float64), pred),
            }
        )

    if np.isnan(oof_pred).any():
        raise RuntimeError(f"{member.name} produced incomplete OOF predictions")

    pred_col = "phase2_oof_pred"
    oof = pool.copy()
    oof[pred_col] = oof_pred
    oof["phase2_oof_error"] = oof[pred_col] - oof["pec50"]
    oof["phase2_oof_abs_error"] = oof["phase2_oof_error"].abs()
    summary = summarize_oof(oof, pred_col)
    summary.insert(0, "member", member.name)
    summary.insert(1, "feature", member.feature)
    summary.insert(2, "top_k", member.top_k or 0)
    return oof, pd.DataFrame(fold_rows), summary


def choose_members(names: list[str]) -> list[MemberConfig]:
    if not names or names == ["all"]:
        return MEMBERS
    by_name = {m.name: m for m in MEMBERS}
    by_feature = {m.feature: m for m in MEMBERS}
    chosen = []
    for name in names:
        if name in by_name:
            chosen.append(by_name[name])
        elif name in by_feature:
            chosen.append(by_feature[name])
        else:
            raise SystemExit(
                f"Unknown member {name!r}. Known: {', '.join(m.name for m in MEMBERS)}"
            )
    return chosen


def write_combined_summary() -> None:
    summaries = []
    for path in sorted(MEMBER_OUT_DIR.glob("*__summary.csv")):
        summaries.append(pd.read_csv(path))
    if not summaries:
        return
    combined = pd.concat(summaries, ignore_index=True)
    combined.to_csv(MEMBER_OUT_DIR / "combined_summary.csv", index=False)
    all_rows = combined[combined["slice"] == "all"].sort_values("mae")
    scoreboard = (
        combined[combined["slice"].isin(SCOREBOARD_SLICES)]
        .pivot_table(
            index=["member", "feature", "top_k"],
            columns="slice",
            values=["mae", "bias_pred_minus_true", "spearman"],
            aggfunc="first",
        )
        .sort_values(("mae", "all"))
    )
    scoreboard.columns = [f"{slice_name}_{metric}" for metric, slice_name in scoreboard.columns]
    scoreboard = scoreboard.reset_index()
    scoreboard.to_csv(MEMBER_OUT_DIR / "phase2_member_slice_scoreboard.csv", index=False)
    report = [
        "# Phase 2 TabPFN member OOF",
        "",
        "These are `train + AS1` cross-fit OOF results for current TabPFN",
        "ensemble members. TabPFN uses v2.6 by default here.",
        "",
        "## All labeled rows",
        "",
        all_rows[
            ["member", "feature", "top_k", "n", "mae", "bias_pred_minus_true", "spearman"]
        ].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Slice scoreboard",
        "",
        scoreboard[
            [
                "member",
                "feature",
                "top_k",
                "all_mae",
                "source_as1_mae",
                "true_lt3_mae",
                "true_gte6_mae",
                "all_spearman",
                "source_as1_spearman",
            ]
        ].to_markdown(index=False, floatfmt=".4f"),
    ]
    (MEMBER_OUT_DIR / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("member", nargs="*", default=["all"])
    parser.add_argument("--tabpfn-version", choices=["v3", "v2_6", "v2_5", "v2"], default="v2_6")
    parser.add_argument("--n-estimators", type=int, default=8)
    parser.add_argument("--softmax-temperature", type=float, default=0.9)
    parser.add_argument("--average-before-softmax", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-clusters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    MEMBER_OUT_DIR.mkdir(parents=True, exist_ok=True)

    pool = build_labeled_pool()
    splits, split_summary = build_phase2_splits(
        pool, n_splits=args.n_splits, n_clusters=args.n_clusters, seed=args.seed
    )
    split_summary.to_csv(MEMBER_OUT_DIR / "split_summary.csv", index=False)

    selected_members = choose_members(args.member)
    for member in selected_members:
        safe_name = member.name.replace("/", "_")
        oof_path = MEMBER_OUT_DIR / f"{safe_name}__oof.csv"
        fold_path = MEMBER_OUT_DIR / f"{safe_name}__fold_metrics.csv"
        summary_path = MEMBER_OUT_DIR / f"{safe_name}__summary.csv"
        meta_path = MEMBER_OUT_DIR / f"{safe_name}__metadata.json"
        if oof_path.exists() and not args.force:
            print(f"skip existing {member.name}")
            continue
        oof, fold_metrics, summary = run_member(member, pool, splits, args)
        oof.to_csv(oof_path, index=False)
        fold_metrics.to_csv(fold_path, index=False)
        summary.to_csv(summary_path, index=False)
        meta = {
            "member": asdict(member),
            "tabpfn_version": args.tabpfn_version,
            "n_estimators": args.n_estimators,
            "softmax_temperature": args.softmax_temperature,
            "average_before_softmax": args.average_before_softmax,
            "device": args.device,
            "n_splits": args.n_splits,
            "n_clusters": args.n_clusters,
            "seed": args.seed,
        }
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {oof_path}")
        print(f"wrote {summary_path}")

    write_combined_summary()
    print(f"wrote {MEMBER_OUT_DIR / 'combined_summary.csv'}")
    print(f"wrote {MEMBER_OUT_DIR / 'report.md'}")


if __name__ == "__main__":
    main()

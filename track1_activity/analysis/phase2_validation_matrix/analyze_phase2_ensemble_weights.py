#!/usr/bin/env -S pixi run python
"""Analyze old ensemble weights on Phase 2 member OOF predictions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy import text

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "scripts")))

from data import get_engine  # noqa: E402
from run_ensemble import optimize_caruana, optimize_l2, optimize_vanilla  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.joinpath("outputs", "tabpfn_members")
REPORT_PATH = OUT_DIR / "phase2_ensemble_weight_analysis.md"
SCOREBOARD_SLICES = ["all", "source_as1", "true_lt3", "true_gte6"]


def metric_row(y: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    err = pred - y
    return {
        "n": int(len(y)),
        "mae": float(np.mean(np.abs(err))),
        "bias_pred_minus_true": float(np.mean(err)),
        "spearman": float(stats.spearmanr(y, pred).statistic),
        "pred_mean": float(np.mean(pred)),
        "true_mean": float(np.mean(y)),
    }


def load_old_weights(name: str = "ens_caruana_bag20") -> pd.Series:
    row = pd.read_sql(
        text(
            """
            SELECT hyperparameters
            FROM experiments
            WHERE name = :name
            """
        ),
        get_engine(),
        params={"name": name},
    )
    if row.empty:
        raise RuntimeError(f"experiment not found: {name}")
    hp = row.iloc[0]["hyperparameters"]
    if isinstance(hp, str):
        hp = json.loads(hp)
    weights = pd.Series(hp["weights"], dtype=np.float64)
    return weights.sort_values(ascending=False)


def load_phase2_oof(member_names: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    frames = []
    base = None
    for name in member_names:
        path = OUT_DIR / f"{name}__oof.csv"
        if not path.exists():
            raise RuntimeError(f"missing Phase2 OOF for old member: {path}")
        df = pd.read_csv(path)
        keep = [
            "pool_idx",
            "source",
            "compound_id",
            "molecule_name",
            "pec50",
            "true_bin",
        ]
        if base is None:
            base = df[keep].copy()
        elif not np.array_equal(base["pool_idx"].to_numpy(), df["pool_idx"].to_numpy()):
            raise RuntimeError(f"pool_idx mismatch for {name}")
        frames.append(df["phase2_oof_pred"].to_numpy(dtype=np.float64))
    if base is None:
        raise RuntimeError("no members loaded")
    return base, np.column_stack(frames)


def summarize_prediction(base: pd.DataFrame, pred: np.ndarray, label: str) -> list[dict]:
    y = base["pec50"].to_numpy(dtype=np.float64)
    masks = {
        "all": np.ones(len(base), dtype=bool),
        "source_train": base["source"].eq("train").to_numpy(),
        "source_as1": base["source"].eq("as1").to_numpy(),
        "true_lt3": (base["pec50"] < 3.0).to_numpy(),
        "true_gte6": (base["pec50"] >= 6.0).to_numpy(),
    }
    rows = []
    for slice_name, mask in masks.items():
        rows.append({"setting": label, "slice": slice_name, **metric_row(y[mask], pred[mask])})
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    old_weights = load_old_weights()
    old_member_names = old_weights.index.tolist()
    base, oof = load_phase2_oof(old_member_names)
    y = base["pec50"].to_numpy(dtype=np.float64)
    old_w = old_weights.to_numpy(dtype=np.float64)
    old_w = old_w / old_w.sum()

    settings: dict[str, np.ndarray] = {
        "old_ens_caruana_bag20_weights": old_w,
        "simple_average_old_members": np.ones(len(old_w), dtype=np.float64) / len(old_w),
    }
    settings["phase2_vanilla_opt"] = optimize_vanilla(oof, y)
    settings["phase2_l2_0p3"] = optimize_l2(oof, y, alpha=0.3)
    settings["phase2_caruana_bag20"] = optimize_caruana(
        oof,
        y,
        n_iter=100,
        init_top_n=3,
        bag_frac=0.5,
        n_bags=20,
        seed=42,
    )

    weight_rows = []
    summary_rows = []
    for setting, weights in settings.items():
        pred = oof @ weights
        summary_rows.extend(summarize_prediction(base, pred, setting))
        for member, weight in zip(old_member_names, weights):
            weight_rows.append({"setting": setting, "member": member, "weight": float(weight)})

    weight_df = pd.DataFrame(weight_rows)
    summary_df = pd.DataFrame(summary_rows)
    old_weight_df = old_weights.rename("weight").reset_index()
    old_weight_df.columns = ["member", "weight"]
    old_weight_df.to_csv(OUT_DIR / "old_ens_caruana_bag20_weights.csv", index=False)
    weight_df.to_csv(OUT_DIR / "phase2_weight_comparison_weights.csv", index=False)
    summary_df.to_csv(OUT_DIR / "phase2_weight_comparison_summary.csv", index=False)
    scoreboard = (
        summary_df[summary_df["slice"].isin(SCOREBOARD_SLICES)]
        .pivot_table(
            index="setting",
            columns="slice",
            values=["mae", "bias_pred_minus_true", "spearman"],
            aggfunc="first",
        )
        .sort_values(("mae", "all"))
    )
    scoreboard.columns = [f"{slice_name}_{metric}" for metric, slice_name in scoreboard.columns]
    scoreboard = scoreboard.reset_index()
    scoreboard.to_csv(OUT_DIR / "phase2_weight_slice_scoreboard.csv", index=False)

    lines = [
        "# Phase 2 ensemble weight analysis",
        "",
        "Old `ens_caruana_bag20` weights applied to the Phase 2 `train + AS1`",
        "OOF matrix, compared with re-optimized Phase 2 weights.",
        "",
        "## Old weights",
        "",
        old_weight_df.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Summary",
        "",
        summary_df[summary_df["slice"].eq("all")]
        .sort_values("mae")
        .to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Slice scoreboard",
        "",
        scoreboard[
            [
                "setting",
                "all_mae",
                "source_as1_mae",
                "true_lt3_mae",
                "true_gte6_mae",
                "all_spearman",
                "source_as1_spearman",
            ]
        ].to_markdown(index=False, floatfmt=".4f"),
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_DIR / 'old_ens_caruana_bag20_weights.csv'}")
    print(f"wrote {OUT_DIR / 'phase2_weight_comparison_summary.csv'}")
    print(f"wrote {OUT_DIR / 'phase2_weight_comparison_weights.csv'}")
    print(f"wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()

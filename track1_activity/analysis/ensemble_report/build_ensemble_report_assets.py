#!/usr/bin/env -S pixi run python
"""Build figures for the Track 1 ensemble/calibration explanation report."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psycopg2
from sklearn.decomposition import TruncatedSVD

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))

from data import DB_PARAMS  # noqa: E402

OUT_DIR = REPO_ROOT.joinpath(
    "docs", "track1_explain", "assets", "ensemble_calibration"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSION_DIR = REPO_ROOT.joinpath("track1_activity", "submissions")


MODEL_LABELS = {
    "tabpfn_cheme_2d_full_boltz_log2fc_pred_optuna_trial10_seed5ens_top500_umap": "top500 optuna",
    "tabpfn_cheme_2d_full_boltz_log2fc_pred_seed10ens_top500_umap": "top500 seed10",
    "tabpfn_cheme_2d_full_boltz_log2fc_pred_optuna_trial10_seed5ens_umap_default": "2D/Boltz/LF optuna",
    "tabpfn_chemprop_pretrain_embed_umap_default": "ChemProp LF embed",
    "tabpfn_kermt_pretrain_embed_umap_default": "KERMT LF embed",
    "tabpfn_pooled_boltz_umap_default": "Boltz pooled",
    "tabpfn_pooled_boltz_allpairs_umap_default": "Boltz allpairs",
    "tabpfn_molformer_c3_pretrain_embed_umap": "MoLFormer LF embed",
    "tabpfn_gatedgcn_pretrain_embed_umap_default": "GatedGCN LF embed",
    "tabpfn_attentivefp_pretrain_embed_umap_default": "AttentiveFP LF embed",
}

STRATEGY_NAMES = [
    "ens_vanilla",
    "ens_l2_a0.1",
    "ens_fold_l2_a0.1",
    "ens_caruana_bag20",
]

STRATEGY_LABELS = {
    "ens_vanilla": "vanilla",
    "ens_l2_a0.1": "L2 alpha=0.1",
    "ens_fold_l2_a0.1": "fold L2 alpha=0.1",
    "ens_caruana_bag20": "Caruana bag20",
}

KEY_SUBMISSIONS = [
    13,
    15,
    16,
    17,
    18,
    19,
    23,
    30,
    31,
    32,
    38,
    39,
    43,
    51,
    55,
    56,
    57,
    58,
    59,
]

PCA_SUBMISSION_IDS = [43, 48, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]

SUBMISSION_LABELS = {
    13: "raw Caruana",
    16: "linear_pos",
    19: "importance affine",
    30: "restore 9-pool",
    31: "5-seed LF",
    39: "Optuna swap",
    43: "hybrid meta",
    51: "reverse probe",
    55: "best anchor",
    56: "bad top500 swap",
    59: "late lift",
}

PCA_LABELS = {
    43: "43 hybrid",
    48: "48 meta axis",
    50: "50 decor",
    51: "51 reverse",
    52: "52 repool swap",
    53: "53 repool core",
    54: "54 potent noaux",
    55: "55 potent gate",
    56: "56 bad swap",
    57: "57 stronger gate",
    58: "58 combo gate",
    59: "59 high lift",
}

PATH_FALLBACKS = {
    "track1_activity/analysis/compound_level_lb/outputs/meta_axis_candidates/ens_meta_axis_a343.csv": (
        "track1_activity/submissions/ens_hybrid_meta_baseline_5050.csv"
    ),
}


def _load_latest_experiment(cur, name: str) -> tuple[int, dict, dict]:
    cur.execute(
        """
        SELECT id, hyperparameters, notes
          FROM experiments
         WHERE name = %s
         ORDER BY id DESC
         LIMIT 1
        """,
        (name,),
    )
    row = cur.fetchone()
    if row is None:
        raise RuntimeError(f"missing experiment: {name}")
    exp_id, hp, notes = row
    if isinstance(hp, str):
        hp = json.loads(hp)
    cur.execute(
        """
        SELECT mae, rae, r2, spearman_r, kendall_tau
          FROM experiment_cv_results
         WHERE experiment_id = %s
         ORDER BY fold
        """,
        (exp_id,),
    )
    metrics_rows = cur.fetchall()
    if not metrics_rows:
        raise RuntimeError(f"missing CV metrics for experiment id={exp_id}")
    metrics = {
        "mae": float(np.mean([r[0] for r in metrics_rows])),
        "rae": float(np.mean([r[1] for r in metrics_rows])),
        "r2": float(np.mean([r[2] for r in metrics_rows])),
        "spearman": float(np.mean([r[3] for r in metrics_rows])),
        "kendall": float(np.mean([r[4] for r in metrics_rows])),
        "notes": notes or "",
    }
    return exp_id, hp, metrics


def _short_name(name: str) -> str:
    return MODEL_LABELS.get(name, name.replace("tabpfn_", "").replace("_umap_default", ""))


def plot_strategy_weights(cur) -> None:
    rows = []
    for strategy in STRATEGY_NAMES:
        _, hp, metrics = _load_latest_experiment(cur, strategy)
        for model, weight in hp["weights"].items():
            rows.append(
                {
                    "strategy": strategy,
                    "strategy_label": STRATEGY_LABELS[strategy],
                    "model": _short_name(model),
                    "weight": float(weight),
                    "mae": metrics["mae"],
                    "rae": metrics["rae"],
                    "spearman": metrics["spearman"],
                }
            )
    df = pd.DataFrame(rows)
    pivot = (
        df.pivot_table(
            index="model", columns="strategy_label", values="weight", fill_value=0.0
        )
        .reindex(columns=[STRATEGY_LABELS[s] for s in STRATEGY_NAMES])
        .sort_values("Caruana bag20", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    y = np.arange(len(pivot))
    height = 0.18
    colors = ["#6b7280", "#4c78a8", "#59a14f", "#e15759"]
    for i, col in enumerate(pivot.columns):
        ax.barh(y + (i - 1.5) * height, pivot[col], height=height, label=col, color=colors[i])
    ax.set_yticks(y)
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_xlabel("Ensemble weight")
    ax.set_title("Continuous Optimizers Concentrate; Caruana Spreads Discrete Counts")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right", frameon=True)
    ax.set_xlim(0, max(0.85, float(pivot.max().max()) + 0.05))

    strategy_metrics = (
        df.drop_duplicates("strategy_label")
        .set_index("strategy_label")
        .reindex(pivot.columns)
    )
    lines = [
        f"{idx}: MAE {row.mae:.4f}, Sp {row.spearman:.4f}"
        for idx, row in strategy_metrics.iterrows()
    ]
    ax.text(
        0.98,
        0.50,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="right",
        va="center",
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#d1d5db"},
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR.joinpath("ensemble_strategy_weights_compare.png"), dpi=180)
    plt.close(fig)


def plot_lb_timeline(cur) -> None:
    cur.execute(
        """
        SELECT id, submission_name, lb_mae, lb_spearman, lb_rank, submitted_at
          FROM lb_submissions
         WHERE track = 'activity'
           AND id = ANY(%s)
           AND lb_mae IS NOT NULL
         ORDER BY id
        """,
        (KEY_SUBMISSIONS,),
    )
    df = pd.DataFrame(
        cur.fetchall(),
        columns=["id", "submission", "mae", "spearman", "rank", "submitted_at"],
    )
    fig, ax1 = plt.subplots(figsize=(11.5, 5.2))
    ax1.plot(df["id"], df["mae"], marker="o", color="#2563eb", lw=2.2, label="Public LB MAE")
    ax1.set_xlabel("Local submission id")
    ax1.set_ylabel("Public LB MAE", color="#2563eb")
    ax1.tick_params(axis="y", labelcolor="#2563eb")
    ax1.grid(alpha=0.25)
    ax1.invert_yaxis()
    x_min = float(df["id"].min()) - 3.0
    x_max = float(df["id"].max()) + 3.0
    mae_min = float(df["mae"].min())
    mae_max = float(df["mae"].max())
    mae_pad = max(0.0025, (mae_max - mae_min) * 0.18)
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(mae_max + mae_pad, mae_min - mae_pad)

    ax2 = ax1.twinx()
    ax2.plot(df["id"], df["spearman"], marker="s", color="#059669", lw=1.8, label="Spearman")
    ax2.set_ylabel("Public LB Spearman", color="#059669")
    ax2.tick_params(axis="y", labelcolor="#059669")
    sp_min = float(df["spearman"].min())
    sp_max = float(df["spearman"].max())
    sp_pad = max(0.0012, (sp_max - sp_min) * 0.20)
    ax2.set_ylim(sp_min - sp_pad, sp_max + sp_pad)

    for _, row in df.iterrows():
        label = SUBMISSION_LABELS.get(int(row.id))
        if not label:
            continue
        yoff = -18 if row.id in {56, 58, 59} else 14
        xoff = 0
        if row.id in {55, 57}:
            yoff = 18
        if row.id == 19:
            xoff = -12
        if row.id == 16:
            xoff = -8
        ax1.annotate(
            label,
            xy=(row.id, row.mae),
            xytext=(xoff, yoff),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#d1d5db"},
            arrowprops={"arrowstyle": "-", "color": "#9ca3af", "lw": 0.8},
        )

    ax1.set_title("Track 1 Ensemble and Calibration Public-LB Trajectory")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines + lines2,
        labels + labels2,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.02),
        ncol=2,
        frameon=True,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT_DIR.joinpath("ensemble_calibration_lb_timeline.png"), dpi=180)
    plt.close(fig)


def plot_calibration_schematic() -> None:
    fig, ax = plt.subplots(figsize=(11.2, 5.0))
    ax.axis("off")

    boxes = [
        (0.04, 0.50, 0.18, 0.26, "Member OOF\npredictions", "#dbeafe"),
        (0.29, 0.50, 0.18, 0.26, "Caruana\nweighted blend", "#fee2e2"),
        (0.54, 0.50, 0.18, 0.26, "Post-hoc\ncalibrator", "#dcfce7"),
        (0.78, 0.50, 0.18, 0.26, "Submission\npEC50", "#fef3c7"),
    ]
    for x, y, w, h, text, color in boxes:
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="#374151", lw=1.4)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=12, weight="bold")

    for x0, x1 in [(0.22, 0.29), (0.47, 0.54), (0.72, 0.78)]:
        ax.annotate(
            "",
            xy=(x1, 0.63),
            xytext=(x0, 0.63),
            arrowprops={"arrowstyle": "->", "lw": 1.8, "color": "#374151"},
        )

    ax.text(
        0.25,
        0.22,
        "OOF side\nreconstruct train predictions\nfit/evaluate with nested UMAP CV",
        ha="left",
        va="center",
        fontsize=10.0,
        color="#111827",
    )
    ax.text(
        0.67,
        0.22,
        "Test side\napply the same fitted transform\nto raw ensemble test predictions",
        ha="left",
        va="center",
        fontsize=10.0,
        color="#111827",
    )
    ax.text(
        0.63,
        0.40,
        "linear_pos:\ny = 1.05 * pred - 0.25",
        ha="center",
        va="center",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#86efac"},
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR.joinpath("calibration_pipeline_schematic.png"), dpi=180)
    plt.close(fig)


def _load_submission(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"SMILES", "Molecule Name", "pEC50"}
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(f"{path} missing columns: {sorted(missing)}")
    return df[["SMILES", "Molecule Name", "pEC50"]].copy()


def _aligned_prediction(path: Path, reference: pd.DataFrame) -> np.ndarray:
    df = _load_submission(path)
    if df[["SMILES", "Molecule Name"]].equals(reference[["SMILES", "Molecule Name"]]):
        return df["pEC50"].to_numpy(dtype=np.float64)
    merged = reference[["SMILES", "Molecule Name"]].merge(
        df[["SMILES", "Molecule Name", "pEC50"]],
        on=["SMILES", "Molecule Name"],
        how="left",
        validate="one_to_one",
    )
    if merged["pEC50"].isna().any():
        raise RuntimeError(f"{path} did not align to the reference submission")
    return merged["pEC50"].to_numpy(dtype=np.float64)


def _resolve_submission_path(file_path: str) -> Path:
    path = REPO_ROOT.joinpath(file_path)
    if path.exists():
        return path
    fallback = PATH_FALLBACKS.get(file_path)
    if fallback:
        fallback_path = REPO_ROOT.joinpath(fallback)
        if fallback_path.exists():
            return fallback_path
    raise FileNotFoundError(path)


def plot_id51_delta_svd(cur) -> None:
    cur.execute(
        """
        SELECT id, submission_name, file_path, lb_mae, lb_spearman, notes
          FROM lb_submissions
         WHERE track = 'activity'
           AND id = ANY(%s)
           AND lb_mae IS NOT NULL
         ORDER BY id
        """,
        (PCA_SUBMISSION_IDS,),
    )
    rows = cur.fetchall()
    meta = pd.DataFrame(
        rows,
        columns=["id", "submission_name", "file_path", "lb_mae", "lb_spearman", "notes"],
    )
    if 51 not in set(meta["id"]):
        raise RuntimeError("id51 is required as the delta anchor")

    id51_path = _resolve_submission_path(
        str(meta.loc[meta["id"] == 51, "file_path"].iloc[0])
    )
    ref = _load_submission(id51_path)
    id51_pred = ref["pEC50"].to_numpy(dtype=np.float64)

    preds: dict[int, np.ndarray] = {}
    for row in meta.itertuples(index=False):
        preds[int(row.id)] = _aligned_prediction(
            _resolve_submission_path(str(row.file_path)), ref
        )

    ids = [int(i) for i in meta["id"].tolist()]
    deltas = np.vstack([preds[i] - id51_pred for i in ids])
    svd = TruncatedSVD(n_components=2, random_state=42)
    coords = svd.fit_transform(deltas)
    coord_df = meta.copy()
    coord_df["x"] = coords[:, 0]
    coord_df["y"] = coords[:, 1]
    id51_mae = float(coord_df.loc[coord_df["id"] == 51, "lb_mae"].iloc[0])
    coord_df["delta_mae_vs_id51"] = coord_df["lb_mae"].astype(float) - id51_mae
    coord_df["mean_abs_delta_vs_id51"] = np.mean(np.abs(deltas), axis=1)
    coord_df["p90_abs_delta_vs_id51"] = np.quantile(np.abs(deltas), 0.90, axis=1)
    coord_df.to_csv(
        OUT_DIR.joinpath("id51_submission_delta_svd_coordinates.csv"), index=False
    )

    fig, ax = plt.subplots(figsize=(10.8, 7.0))
    colors = []
    for row in coord_df.itertuples(index=False):
        if row.id == 51:
            colors.append("#111827")
        elif row.delta_mae_vs_id51 < -1e-9:
            colors.append("#2563eb")
        else:
            colors.append("#dc2626")

    sizes = 90 + 1400 * np.clip(coord_df["mean_abs_delta_vs_id51"].to_numpy(), 0, 0.08)
    ax.scatter(
        coord_df["x"],
        coord_df["y"],
        s=sizes,
        c=colors,
        alpha=0.88,
        edgecolor="white",
        linewidth=1.0,
        zorder=3,
    )

    def point(sub_id: int) -> tuple[float, float]:
        row = coord_df.loc[coord_df["id"] == sub_id].iloc[0]
        return float(row["x"]), float(row["y"])

    arrows = [
        (48, 50, "#9ca3af", "bad decor direction"),
        (50, 51, "#2563eb", "reverse away"),
        (51, 55, "#2563eb", "potent46 gate"),
        (55, 57, "#f59e0b", "more gate"),
        (55, 58, "#dc2626", "combo gate"),
        (57, 59, "#dc2626", "high lift"),
        (51, 56, "#dc2626", "bad top500 swap"),
    ]
    for start, end, color, _ in arrows:
        if start not in ids or end not in ids:
            continue
        x0, y0 = point(start)
        x1, y1 = point(end)
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops={
                "arrowstyle": "->",
                "color": color,
                "lw": 1.8,
                "shrinkA": 8,
                "shrinkB": 8,
            },
            zorder=2,
        )

    label_offsets = {
        43: (8, 10),
        48: (8, 14),
        50: (8, 10),
        51: (8, 16),
        52: (8, 12),
        53: (8, 12),
        54: (8, 12),
        55: (8, 10),
        56: (8, -14),
        57: (8, -16),
        58: (8, -12),
        59: (8, -16),
    }
    for row in coord_df.itertuples(index=False):
        label = PCA_LABELS.get(int(row.id), str(row.id))
        offset = label_offsets.get(int(row.id), (8, 8))
        ax.annotate(
            label,
            xy=(row.x, row.y),
            xytext=offset,
            textcoords="offset points",
            fontsize=8.5,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "#d1d5db"},
            zorder=4,
        )

    ax.axhline(0, color="#d1d5db", lw=0.8, zorder=1)
    ax.axvline(0, color="#d1d5db", lw=0.8, zorder=1)
    ax.set_xlabel(f"Delta SVD 1 ({svd.explained_variance_ratio_[0] * 100:.1f}% var)")
    ax.set_ylabel(f"Delta SVD 2 ({svd.explained_variance_ratio_[1] * 100:.1f}% var)")
    ax.set_title("Submission Prediction Movements Around id51")
    ax.grid(alpha=0.22)

    legend_text = [
        "blue: public MAE improved vs id51",
        "red: public MAE worsened vs id51",
        "point size: mean |prediction shift| vs id51",
    ]
    ax.text(
        0.98,
        0.02,
        "\n".join(legend_text),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#d1d5db"},
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR.joinpath("id51_submission_delta_svd.png"), dpi=180)
    plt.close(fig)

    zoom_ids = [51, 55, 57, 58, 59]
    zoom = coord_df[coord_df["id"].isin(zoom_ids)].copy()
    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    zoom_colors = [
        "#111827" if row.id == 51 else ("#2563eb" if row.delta_mae_vs_id51 < 0 else "#dc2626")
        for row in zoom.itertuples(index=False)
    ]
    zoom_sizes = 120 + 1800 * np.clip(
        zoom["mean_abs_delta_vs_id51"].to_numpy(), 0, 0.04
    )
    ax.scatter(
        zoom["x"],
        zoom["y"],
        s=zoom_sizes,
        c=zoom_colors,
        alpha=0.9,
        edgecolor="white",
        linewidth=1.1,
        zorder=3,
    )
    for start, end, color, _ in arrows:
        if start not in zoom_ids or end not in zoom_ids:
            continue
        x0, y0 = point(start)
        x1, y1 = point(end)
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops={
                "arrowstyle": "->",
                "color": color,
                "lw": 2.0,
                "shrinkA": 9,
                "shrinkB": 9,
            },
            zorder=2,
        )
    zoom_offsets = {
        51: (-58, 14),
        55: (-76, 10),
        57: (12, -12),
        58: (10, 8),
        59: (12, -14),
    }
    for row in zoom.itertuples(index=False):
        ax.annotate(
            f"{int(row.id)}\nMAE {row.lb_mae:.6f}",
            xy=(row.x, row.y),
            xytext=zoom_offsets.get(int(row.id), (8, 8)),
            textcoords="offset points",
            fontsize=8.5,
            bbox={"boxstyle": "round,pad=0.24", "facecolor": "white", "edgecolor": "#d1d5db"},
        )
    pad_x = 0.045
    pad_y = 0.08
    ax.set_xlim(float(zoom["x"].min()) - pad_x, float(zoom["x"].max()) + pad_x)
    ax.set_ylim(float(zoom["y"].min()) - pad_y, float(zoom["y"].max()) + pad_y)
    ax.axhline(0, color="#d1d5db", lw=0.8)
    ax.axvline(0, color="#d1d5db", lw=0.8)
    ax.grid(alpha=0.24)
    ax.set_xlabel("Delta SVD 1")
    ax.set_ylabel("Delta SVD 2")
    ax.set_title("Zoom: id51 Local Gate Path")
    fig.tight_layout()
    fig.savefig(OUT_DIR.joinpath("id51_gate_path_svd_zoom.png"), dpi=180)
    plt.close(fig)

    vectors = []
    for start, end, _, label in arrows:
        if start not in ids or end not in ids:
            continue
        delta = preds[end] - preds[start]
        vectors.append(
            {
                "direction": f"id{start}->id{end}",
                "label": label,
                "lb_delta_mae": float(
                    coord_df.loc[coord_df["id"] == end, "lb_mae"].iloc[0]
                    - coord_df.loc[coord_df["id"] == start, "lb_mae"].iloc[0]
                ),
                "mean_abs_shift": float(np.mean(np.abs(delta))),
                "p90_abs_shift": float(np.quantile(np.abs(delta), 0.90)),
                "max_abs_shift": float(np.max(np.abs(delta))),
            }
        )
    pd.DataFrame(vectors).to_csv(
        OUT_DIR.joinpath("id51_submission_direction_summary.csv"), index=False
    )


def main() -> None:
    with psycopg2.connect(**DB_PARAMS) as conn:
        cur = conn.cursor()
        plot_strategy_weights(cur)
        plot_lb_timeline(cur)
        plot_id51_delta_svd(cur)
    plot_calibration_schematic()
    print(f"wrote assets to {OUT_DIR}")


if __name__ == "__main__":
    main()

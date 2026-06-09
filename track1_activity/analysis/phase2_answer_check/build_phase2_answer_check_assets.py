#!/usr/bin/env -S pixi run python
"""Build figures for the Track 1 Phase 2 answer-check report."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))

from data import get_engine, load_test_smiles, load_train_smiles_target  # noqa: E402
from splits import _morgan_fp_matrix  # noqa: E402

ASSET_DIR = REPO_ROOT / "docs" / "track1_explain" / "assets" / "phase2_answer_check"
SUBMISSION_DIR = REPO_ROOT / "track1_activity" / "submissions"
LF_PATH = (
    REPO_ROOT
    / "data"
    / "chemprop_pretrain_log2fc_predictions_optuna_trial10_seed5ens.parquet"
)

ANCHOR_CASES = {
    "id55 g35": "ens_id51_top500_potent46_t40_soft_g35.csv",
    "id57 g50": "ens_id51_top500_potent46_t40_soft_g50.csv",
    "id58 combo": "ens_id55_combo_gate_rank1.csv",
    "id59 lift": "ens_id57_high_activity_lift_rank2.csv",
    "id56 bad swap": "ens_swap_optuna_t10_top500_calibrated_importance.csv",
}

SELECTED_MODELS = {
    "id55 anchor": SUBMISSION_DIR / "ens_id51_top500_potent46_t40_soft_g35.csv",
    "best single temp0.7": SUBMISSION_DIR
    / "tabpfn_cheme_2d_full_boltz_log2fc_pred_seed10ens_top500_umap_v3_temp0p7.csv",
    "top500 seed10": SUBMISSION_DIR
    / "tabpfn_cheme_2d_full_boltz_log2fc_pred_seed10ens_top500_umap.csv",
    "broad optuna": SUBMISSION_DIR
    / "tabpfn_cheme_2d_full_boltz_log2fc_pred_optuna_trial10_seed5ens_umap_default.csv",
    "bad OOF top500": SUBMISSION_DIR
    / "tabpfn_cheme_2d_full_boltz_log2fc_pred_optuna_trial10_seed5ens_top500_umap.csv",
    "ChemProp LF embed": SUBMISSION_DIR
    / "tabpfn_chemprop_pretrain_embed_umap_default.csv",
    "KERMT LF embed": SUBMISSION_DIR / "tabpfn_kermt_pretrain_embed_umap_default.csv",
    "MoLFormer LF embed": SUBMISSION_DIR
    / "tabpfn_molformer_c3_pretrain_embed_umap.csv",
    "direct MoLFormer LoRA": SUBMISSION_DIR
    / "peft_molformer_xl_lora_r32a64_umap_default.csv",
}


def savefig(fig: plt.Figure, name: str) -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(ASSET_DIR / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def load_as1() -> pd.DataFrame:
    engine = get_engine()
    return pd.read_sql(
        """
        SELECT
            t.id AS test_id,
            c.id AS compound_id,
            c.molecule_name,
            c.std_smiles AS smiles,
            l.pec50 AS true_pec50,
            l.emax_estimate,
            l.emax_vs_pos_ctrl
        FROM test_activity_phase1_labels l
        JOIN test_activity t ON t.compound_id = l.compound_id
        JOIN compounds c ON c.id = l.compound_id
        ORDER BY t.id
        """,
        engine,
    )


def load_submission(path: Path, col: str = "pred") -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.rename(columns={"Molecule Name": "molecule_name", "pEC50": col})[
        ["molecule_name", col]
    ]


def add_id55_context(as1: pd.DataFrame) -> pd.DataFrame:
    df = as1.merge(
        load_submission(
            SUBMISSION_DIR / "ens_id51_top500_potent46_t40_soft_g35.csv", "pred_id55"
        ),
        on="molecule_name",
        how="left",
    )
    df["error"] = df["pred_id55"] - df["true_pec50"]
    df["abs_error"] = df["error"].abs()

    train = load_train_smiles_target()
    train_fp = _morgan_fp_matrix(train["smiles"].tolist()).astype(bool)
    test_fp = _morgan_fp_matrix(load_test_smiles()["smiles"].tolist()).astype(bool)
    query_fp = test_fp[df["test_id"].to_numpy(dtype=int) - 1]

    inter = query_fp.astype(np.uint16) @ train_fp.astype(np.uint16).T
    union = (
        query_fp.sum(axis=1, keepdims=True)
        + train_fp.sum(axis=1, keepdims=True).T
        - inter
    )
    sim = np.divide(
        inter, union, out=np.zeros_like(inter, dtype=np.float32), where=union > 0
    )
    nn_idx = sim.argmax(axis=1)
    df["nn_train_tanimoto"] = sim[np.arange(len(df)), nn_idx]
    df["nn_train_pec50"] = train["pec50"].to_numpy()[nn_idx]

    potent_mask = train["pec50"].to_numpy() >= 6.0
    potent_fp = train_fp[potent_mask]
    inter_p = query_fp.astype(np.uint16) @ potent_fp.astype(np.uint16).T
    union_p = (
        query_fp.sum(axis=1, keepdims=True)
        + potent_fp.sum(axis=1, keepdims=True).T
        - inter_p
    )
    sim_p = np.divide(
        inter_p,
        union_p,
        out=np.zeros_like(inter_p, dtype=np.float32),
        where=union_p > 0,
    )
    df["nn_potent_tanimoto"] = sim_p.max(axis=1)

    lf = pd.read_parquet(LF_PATH).loc[df["compound_id"].astype(int).tolist()].copy()
    lf["lf_mean"] = 0.5 * (lf["log2fc_8p25_pred"] + lf["log2fc_33_pred"])
    lf = lf.reset_index()
    return df.merge(lf, on="compound_id", how="left")


def metrics(y: pd.Series, pred: pd.Series) -> dict[str, float]:
    err = pred.to_numpy(dtype=float) - y.to_numpy(dtype=float)
    return {
        "mae": float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),
        "spearman": float(stats.spearmanr(y, pred).statistic),
    }


def plot_anchor_replay(as1: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, filename in ANCHOR_CASES.items():
        sub = load_submission(SUBMISSION_DIR / filename, "pred")
        joined = as1.merge(sub, on="molecule_name", how="inner")
        row = {"case": label, **metrics(joined["true_pec50"], joined["pred"])}
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(ASSET_DIR / "anchor_replay_summary.csv", index=False)

    colors = ["#4c78a8", "#72b7b2", "#f58518", "#e45756", "#b279a2"]
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    bars = ax.bar(out["case"], out["mae"], color=colors, alpha=0.92)
    ax.axhline(out.loc[out["case"] == "id55 g35", "mae"].iloc[0], color="#333333", lw=1)
    ax.bar_label(bars, labels=[f"{v:.4f}" for v in out["mae"]], padding=3, fontsize=9)
    ax.set_ylabel("AS1 MAE")
    ax.set_title("Recent Phase 1 anchors replayed on released AS1 labels")
    ax.set_ylim(0.400, max(out["mae"]) + 0.006)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=20)
    savefig(fig, "anchor_as1_mae.png")
    return out


def plot_id55_error_shape(df: pd.DataFrame) -> pd.DataFrame:
    labels = ["<3", "3-4", "4-5", "5-6", ">=6"]
    df = df.copy()
    df["true_bin"] = pd.cut(
        df["true_pec50"], [-np.inf, 3, 4, 5, 6, np.inf], labels=labels
    )
    summary = (
        df.groupby("true_bin", observed=True)
        .agg(
            n=("error", "size"),
            mae=("abs_error", "mean"),
            bias=("error", "mean"),
            true_mean=("true_pec50", "mean"),
            pred_mean=("pred_id55", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(ASSET_DIR / "id55_error_by_true_bin.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    x = np.arange(len(summary))
    axes[0].bar(x, summary["mae"], color="#4c78a8", alpha=0.9)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(summary["true_bin"])
    axes[0].set_ylabel("MAE")
    axes[0].set_title("id55 AS1 error by true pEC50 bin")
    axes[0].grid(axis="y", alpha=0.25)
    for i, row in summary.iterrows():
        axes[0].text(
            i, row["mae"] + 0.025, f"n={int(row['n'])}", ha="center", fontsize=9
        )

    bias_colors = np.where(summary["bias"] >= 0, "#e45756", "#4c78a8")
    axes[1].bar(x, summary["bias"], color=bias_colors, alpha=0.9)
    axes[1].axhline(0, color="#333333", lw=1)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(summary["true_bin"])
    axes[1].set_ylabel("Prediction - true")
    axes[1].set_title("Compression: low overpredicted, high underpredicted")
    axes[1].grid(axis="y", alpha=0.25)
    savefig(fig, "id55_error_by_true_bin.png")
    return summary


def plot_true_vs_pred(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    sc = ax.scatter(
        df["true_pec50"],
        df["pred_id55"],
        c=df["log2fc_33_pred"],
        cmap="viridis",
        s=34,
        alpha=0.88,
        linewidths=0,
    )
    lims = [1.6, 6.9]
    ax.plot(lims, lims, color="#333333", lw=1, ls="--")
    ax.set_xlim(lims)
    ax.set_ylim(2.2, 6.2)
    ax.set_xlabel("Released AS1 pEC50")
    ax.set_ylabel("id55 prediction")
    ax.set_title("id55 predictions compress AS1 extremes")
    fig.colorbar(sc, ax=ax, label="predicted log2fc_33")
    ax.grid(alpha=0.22)
    savefig(fig, "id55_true_vs_pred_log2fc.png")


def plot_oof_vs_as1(as1: pd.DataFrame) -> pd.DataFrame:
    engine = get_engine()
    exp = pd.read_sql(
        """
        SELECT e.name, e.submission_path, s.mae_mean AS oof_mae, s.spearman_mean AS oof_spearman
        FROM experiments e
        JOIN experiment_summary s ON s.id = e.id
        WHERE e.submission_path IS NOT NULL
        ORDER BY e.id
        """,
        engine,
    )
    rows = []
    seen = set()
    for row in exp.itertuples(index=False):
        if row.name in seen:
            continue
        seen.add(row.name)
        path = REPO_ROOT / row.submission_path
        if not path.exists():
            continue
        try:
            sub = load_submission(path, "pred")
        except Exception:
            continue
        joined = as1.merge(sub, on="molecule_name", how="inner")
        if len(joined) != len(as1):
            continue
        m = metrics(joined["true_pec50"], joined["pred"])
        rows.append(
            {
                "name": row.name,
                "oof_mae": float(row.oof_mae),
                "oof_spearman": float(row.oof_spearman),
                "as1_mae": m["mae"],
                "as1_bias": m["bias"],
                "as1_spearman": m["spearman"],
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(ASSET_DIR / "oof_vs_as1_experiments.csv", index=False)

    finite = out[np.isfinite(out["oof_mae"]) & np.isfinite(out["as1_mae"])].copy()
    plot_df = finite[(finite["oof_mae"] <= 0.85) & (finite["as1_mae"] <= 0.90)].copy()

    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    ax.scatter(
        plot_df["oof_mae"],
        plot_df["as1_mae"],
        s=16,
        alpha=0.42,
        color="#6b7280",
        lw=0,
    )
    highlights = {
        "tabpfn_cheme_2d_full_boltz_log2fc_pred_seed10ens_top500_umap_v3_temp0p7": "best single",
        "tabpfn_cheme_2d_full_boltz_log2fc_pred_optuna_trial10_seed5ens_top500_umap": "bad OOF top500",
        "peft_molformer_xl_lora_r32a64_umap_default": "direct LoRA",
        "tabpfn_chemprop_assay_shape_drlatent_embed_umap_default": "AS1 surprise",
    }
    for name, label in highlights.items():
        sub = plot_df[plot_df["name"] == name]
        if sub.empty:
            continue
        ax.scatter(sub["oof_mae"], sub["as1_mae"], s=60, color="#e45756", zorder=5)
        ax.annotate(
            label,
            (sub["oof_mae"].iloc[0], sub["as1_mae"].iloc[0]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    r = stats.pearsonr(finite["oof_mae"], finite["as1_mae"]).statistic
    ax.set_xlabel("OOF MAE")
    ax.set_ylabel("AS1 MAE")
    ax.set_title(f"OOF is useful globally, weak for tiny top-end deltas (r={r:.3f})")
    ax.set_xlim(0.36, 0.82)
    ax.set_ylim(0.38, 0.86)
    ax.grid(alpha=0.22)
    savefig(fig, "oof_vs_as1_scatter.png")
    return out


def plot_selected_model_tails(as1: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, path in SELECTED_MODELS.items():
        if not path.exists():
            continue
        joined = as1.merge(
            load_submission(path, "pred"), on="molecule_name", how="inner"
        )
        joined["error"] = joined["pred"] - joined["true_pec50"]
        joined["abs_error"] = joined["error"].abs()
        low = joined[joined["true_pec50"] < 3]
        high = joined[joined["true_pec50"] >= 6]
        rows.append(
            {
                "model": label,
                "as1_mae": float(joined["abs_error"].mean()),
                "low_tail_mae": float(low["abs_error"].mean()),
                "high_tail_mae": float(high["abs_error"].mean()),
            }
        )
    out = pd.DataFrame(rows).sort_values("as1_mae")
    out.to_csv(ASSET_DIR / "selected_model_tail_errors.csv", index=False)

    fig, ax = plt.subplots(figsize=(9.4, 5.6))
    y = np.arange(len(out))
    ax.barh(
        y - 0.18, out["low_tail_mae"], height=0.34, color="#e45756", label="true <3"
    )
    ax.barh(
        y + 0.18, out["high_tail_mae"], height=0.34, color="#4c78a8", label="true >=6"
    )
    ax.set_yticks(y)
    ax.set_yticklabels(out["model"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Tail MAE on AS1")
    ax.set_title("Extreme-tail errors are shared across model families")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=True)
    savefig(fig, "selected_model_tail_errors.png")
    return out


def plot_log2fc_proxy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["lf33_quartile"] = pd.qcut(
        df["log2fc_33_pred"], 4, labels=["Q1 low", "Q2", "Q3", "Q4 high"]
    )
    summary = (
        df.groupby("lf33_quartile", observed=True)
        .agg(
            n=("error", "size"),
            mae=("abs_error", "mean"),
            bias=("error", "mean"),
            true_mean=("true_pec50", "mean"),
            pred_mean=("pred_id55", "mean"),
            lf33=("log2fc_33_pred", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(ASSET_DIR / "id55_error_by_log2fc_quartile.csv", index=False)

    fig, ax1 = plt.subplots(figsize=(7.4, 4.8))
    x = np.arange(len(summary))
    ax1.bar(x, summary["mae"], color="#4c78a8", alpha=0.88, label="MAE")
    ax1.set_ylabel("id55 AS1 MAE")
    ax1.set_xticks(x)
    ax1.set_xticklabels(summary["lf33_quartile"])
    ax1.grid(axis="y", alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(x, summary["true_mean"], color="#f58518", marker="o", label="true mean")
    ax2.plot(x, summary["pred_mean"], color="#54a24b", marker="o", label="pred mean")
    ax2.set_ylabel("pEC50 mean")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", frameon=True)
    ax1.set_title("Predicted log2fc separates activity, but low-LF tail is hard")
    savefig(fig, "id55_error_by_log2fc_quartile.png")
    return summary


def plot_proxy_correlations(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "nn_train_tanimoto",
        "nn_train_pec50",
        "nn_potent_tanimoto",
        "log2fc_8p25_pred",
        "log2fc_33_pred",
        "lf_mean",
        "emax_estimate",
        "emax_vs_pos_ctrl",
    ]
    rows = []
    for col in cols:
        rows.append(
            {
                "proxy": col,
                "pearson_error": float(stats.pearsonr(df[col], df["error"]).statistic),
                "pearson_abs_error": float(
                    stats.pearsonr(df[col], df["abs_error"]).statistic
                ),
                "spearman_true": float(
                    stats.spearmanr(df[col], df["true_pec50"]).statistic
                ),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(ASSET_DIR / "proxy_correlations.csv", index=False)

    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    y = np.arange(len(out))
    ax.barh(
        y - 0.18,
        out["spearman_true"],
        height=0.34,
        color="#54a24b",
        label="Spearman vs true",
    )
    ax.barh(
        y + 0.18,
        out["pearson_abs_error"],
        height=0.34,
        color="#e45756",
        label="Pearson vs abs error",
    )
    ax.axvline(0, color="#333333", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(out["proxy"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("correlation")
    ax.set_title(
        "Predicted log2fc is activity-aligned; NN proxies are weak global error gates"
    )
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=True, loc="lower right")
    savefig(fig, "proxy_correlations.png")
    return out


def plot_label_support_density(df: pd.DataFrame) -> pd.DataFrame:
    labels = ["<3", "3-4", "4-5", "5-6", ">=6"]
    bins = [-np.inf, 3, 4, 5, 6, np.inf]
    train = load_train_smiles_target()
    train = train.copy()
    train["true_bin"] = pd.cut(train["pec50"], bins, labels=labels)

    test_fp = _morgan_fp_matrix(load_test_smiles()["smiles"].tolist()).astype(bool)
    train_fp = _morgan_fp_matrix(train["smiles"].tolist()).astype(bool)
    query_fp = test_fp[df["test_id"].to_numpy(dtype=int) - 1]
    inter = query_fp.astype(np.uint16) @ train_fp.astype(np.uint16).T
    union = (
        query_fp.sum(axis=1, keepdims=True)
        + train_fp.sum(axis=1, keepdims=True).T
        - inter
    )
    sim = np.divide(
        inter, union, out=np.zeros_like(inter, dtype=np.float32), where=union > 0
    )

    rows = []
    for i, row in enumerate(df.itertuples(index=False)):
        true_bin = pd.cut([row.true_pec50], bins, labels=labels)[0]
        for threshold in (0.40, 0.50, 0.60):
            mask = sim[i] >= threshold
            near = train.loc[mask]
            same_bin = near[near["true_bin"] == true_bin]
            low = near[near["pec50"] < 3]
            high = near[near["pec50"] >= 6]
            rows.append(
                {
                    "molecule_name": row.molecule_name,
                    "true_pec50": row.true_pec50,
                    "true_bin": str(true_bin),
                    "abs_error": row.abs_error,
                    "threshold": threshold,
                    "n_neighbors": int(mask.sum()),
                    "n_same_bin": int(len(same_bin)),
                    "n_low_train": int(len(low)),
                    "n_high_train": int(len(high)),
                    "mean_neighbor_pec50": float(near["pec50"].mean())
                    if len(near)
                    else np.nan,
                    "max_sim": float(sim[i].max()),
                }
            )
    density = pd.DataFrame(rows)
    density.to_csv(ASSET_DIR / "as1_local_train_support_density.csv", index=False)

    summary = (
        density.groupby(["true_bin", "threshold"], observed=True)
        .agg(
            n_as1=("molecule_name", "size"),
            median_neighbors=("n_neighbors", "median"),
            mean_neighbors=("n_neighbors", "mean"),
            any_neighbor_rate=(
                "n_neighbors",
                lambda values: float((values > 0).mean()),
            ),
            median_same_bin=("n_same_bin", "median"),
            mean_same_bin=("n_same_bin", "mean"),
            same_bin_support_rate=(
                "n_same_bin",
                lambda values: float((values > 0).mean()),
            ),
            median_low_train=("n_low_train", "median"),
            median_high_train=("n_high_train", "median"),
            mean_abs_error=("abs_error", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(ASSET_DIR / "as1_local_train_support_summary.csv", index=False)

    train_counts = train["true_bin"].value_counts().reindex(labels, fill_value=0)
    as1_counts = (
        pd.cut(df["true_pec50"], bins, labels=labels)
        .value_counts()
        .reindex(labels, fill_value=0)
    )
    counts = pd.DataFrame(
        {
            "true_bin": labels,
            "train_count": train_counts.values,
            "as1_count": as1_counts.values,
        }
    )
    counts.to_csv(ASSET_DIR / "train_as1_label_bin_counts.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    x = np.arange(len(labels))
    axes[0].bar(
        x - 0.18, counts["train_count"], width=0.36, color="#4c78a8", label="train"
    )
    axes[0].bar(x + 0.18, counts["as1_count"], width=0.36, color="#f58518", label="AS1")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("compound count")
    axes[0].set_title("Label density is thickest in the middle")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=True)

    heat = (
        summary.pivot(
            index="true_bin", columns="threshold", values="same_bin_support_rate"
        )
        .reindex(labels)
        .reindex(columns=[0.40, 0.50, 0.60])
    )
    image = axes[1].imshow(
        heat.to_numpy(dtype=float),
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
        cmap="viridis",
    )
    axes[1].set_xticks(np.arange(len(heat.columns)))
    axes[1].set_xticklabels([f"{threshold:.2f}" for threshold in heat.columns])
    axes[1].set_yticks(np.arange(len(labels)))
    axes[1].set_yticklabels(labels)
    axes[1].set_xlabel("Morgan Tanimoto threshold")
    axes[1].set_ylabel("AS1 true bin")
    axes[1].set_title("AS1 with >=1 same-bin train neighbor")
    for row_i, label in enumerate(labels):
        for col_i, threshold in enumerate(heat.columns):
            value = heat.loc[label, threshold]
            axes[1].text(
                col_i,
                row_i,
                f"{100 * value:.0f}%",
                ha="center",
                va="center",
                color="white" if value < 0.45 else "black",
                fontsize=9,
            )
    colorbar = fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)
    colorbar.set_label("support rate")
    savefig(fig, "label_support_density.png")
    return summary


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    as1 = load_as1()
    context = add_id55_context(as1)

    plot_anchor_replay(as1)
    plot_id55_error_shape(context)
    plot_true_vs_pred(context)
    plot_oof_vs_as1(as1)
    plot_selected_model_tails(as1)
    plot_log2fc_proxy(context)
    plot_proxy_correlations(context)
    plot_label_support_density(context)

    largest = context.sort_values("abs_error", ascending=False).head(30)
    largest.to_csv(
        ASSET_DIR / "id55_largest_errors_with_proxy_context.csv", index=False
    )
    print(f"Wrote assets to {ASSET_DIR}")


if __name__ == "__main__":
    main()

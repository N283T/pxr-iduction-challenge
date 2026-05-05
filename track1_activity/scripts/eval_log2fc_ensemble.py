"""Phase 2 of issue #115: evaluate log2fc prediction quality per encoder
and compare 4 ensemble strategies on the seed=42 pretrain val split.

Evaluation protocol:
- Reconstruct the 4-encoder shared val_idx (np.random.default_rng(42)
  on 13136 std_smiles compounds, val_frac=0.1 → 1313 compound_ids).
- Intersect with the 4653 train+test union that has log2fc predictions
  (~442 compounds fall in both sets).
- Load ground-truth log2fc_8p25 / log2fc_33 per compound_id from
  single_concentration (averaged per concentration band).
- For each of 5 encoders (chemprop, molformer_c3, attentivefp, gatedgcn,
  kermt), compute MAE against ground truth on the intersection.
- Test 4 ensemble strategies:
    mean_5        = simple mean of all 5 predictions
    mean_4        = simple mean excluding weakest encoder (attentivefp or gatedgcn, auto-selected from val loss)
    weighted_val  = inverse-val-loss weighted mean (chemprop gets highest weight)
    top2          = (chemprop + molformer_c3) mean

Caveat: KERMT's internal split is scaffold_balanced (not our seed=42),
so its predictions on val_ids may include some compounds it trained on.
This is flagged in the output but doesn't invalidate downstream use —
Phase 3 (downstream OOF) is the real acceptance gate.

Output: stdout table + track1_activity/reports/log2fc_ensemble_phase2.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))

from data import DB_PARAMS  # noqa: E402

ENCODERS = {
    "chemprop": (
        REPO_ROOT.joinpath("data", "chemprop_pretrain_log2fc_predictions.parquet"),
        0.3647,
    ),
    "molformer_c3": (
        REPO_ROOT.joinpath("data", "molformer_c3_pretrain_log2fc_predictions.parquet"),
        0.5065,
    ),
    "attentivefp": (
        REPO_ROOT.joinpath("data", "attentivefp_pretrain_log2fc_predictions.parquet"),
        0.7012,
    ),
    "gatedgcn": (
        REPO_ROOT.joinpath("data", "gatedgcn_pretrain_log2fc_predictions.parquet"),
        0.7478,
    ),
    "kermt": (
        REPO_ROOT.joinpath("data", "kermt_pretrain_log2fc_predictions.parquet"),
        None,  # external val, incomparable val loss
    ),
}
OUT_CSV = REPO_ROOT.joinpath("track1_activity", "reports", "log2fc_ensemble_phase2.csv")


def get_pretrain_val_ids(seed: int = 42, val_frac: float = 0.1) -> list[int]:
    """Reconstruct the 4-encoder shared val split on 13136 std_smiles compounds."""
    sql = (
        "SELECT c.id AS compound_id FROM compounds c "
        "WHERE c.std_smiles IS NOT NULL ORDER BY c.id"
    )
    with psycopg2.connect(**DB_PARAMS) as conn:
        all_ids = pd.read_sql(sql, conn)["compound_id"].to_numpy()
    n = len(all_ids)
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = int(n * val_frac)
    return sorted(all_ids[idx[:n_val]].tolist())


def load_ground_truth() -> pd.DataFrame:
    sql = """
    SELECT compound_id,
      AVG(CASE WHEN concentration_m BETWEEN 8.2e-6 AND 8.3e-6
               THEN log2_fc_estimate END) AS log2fc_8p25,
      AVG(CASE WHEN concentration_m BETWEEN 3.28e-5 AND 3.32e-5
               THEN log2_fc_estimate END) AS log2fc_33
    FROM single_concentration
    GROUP BY compound_id
    """
    with psycopg2.connect(**DB_PARAMS) as conn:
        return pd.read_sql(sql, conn).set_index("compound_id")


def mae(pred: pd.Series, truth: pd.Series) -> tuple[float, int]:
    m = pred.notna() & truth.notna()
    n = int(m.sum())
    if n == 0:
        return float("nan"), 0
    return float(np.mean(np.abs(pred[m].values - truth[m].values))), n


def main() -> None:
    val_ids = get_pretrain_val_ids()
    print(f"Pretrain val split: {len(val_ids)} compound_ids")

    truth = load_ground_truth()
    print(f"Single_concentration ground truth: {len(truth)} compound_ids")

    preds_by_enc: dict[str, pd.DataFrame] = {}
    for name, (path, _val_loss) in ENCODERS.items():
        if not path.exists():
            print(f"  missing: {name} at {path}")
            continue
        df = pd.read_parquet(path)
        preds_by_enc[name] = df
        print(f"  loaded {name}: {df.shape}")

    shared_ids = set(preds_by_enc["chemprop"].index)
    for name, df in preds_by_enc.items():
        shared_ids &= set(df.index)
    eval_ids = sorted(set(val_ids) & shared_ids & set(truth.index))
    print(f"Eval compounds (val ∩ all 5 encoders ∩ ground truth): {len(eval_ids)}")

    truth_eval = truth.loc[eval_ids]

    # LEAK WARNING: prepare_kermt_pretrain_csv.py uses split = int(len*0.9)
    # with rng=default_rng(42) and takes idx[:0.9] as KERMT-train, idx[0.9:]
    # as KERMT-val. The 4 other pretrains use the SAME seed=42 shuffle but
    # take idx[:0.1] as val, idx[0.1:] as train. So KERMT's train set ⊇
    # other 4's val set → KERMT log2fc predictions on our eval set are
    # memorized, not held-out. Keep KERMT's solo MAE in the report for
    # transparency but exclude it from clean ensemble candidates.

    rows: list[dict] = []
    for name, df in preds_by_enc.items():
        sub = df.loc[eval_ids]
        m8, n8 = mae(sub["log2fc_8p25_pred"], truth_eval["log2fc_8p25"])
        m33, n33 = mae(sub["log2fc_33_pred"], truth_eval["log2fc_33"])
        rows.append(
            {
                "variant": f"{name} (solo)",
                "mae_8p25": m8,
                "n_8p25": n8,
                "mae_33": m33,
                "n_33": n33,
                "val_loss_pretrain": ENCODERS[name][1],
                "note": ("LEAKY: KERMT-train ⊇ eval set" if name == "kermt" else ""),
            }
        )

    clean_order = ["chemprop", "molformer_c3", "attentivefp", "gatedgcn"]

    def stack(enc_list: list[str], col: str) -> np.ndarray:
        return np.stack(
            [preds_by_enc[e].loc[eval_ids, col].values for e in enc_list], axis=1
        )

    stack8 = stack(clean_order, "log2fc_8p25_pred")
    stack33 = stack(clean_order, "log2fc_33_pred")

    mean4_8 = stack8.mean(axis=1)
    mean4_33 = stack33.mean(axis=1)

    mask3 = [i for i, e in enumerate(clean_order) if e != "gatedgcn"]
    mean3_8 = stack8[:, mask3].mean(axis=1)
    mean3_33 = stack33[:, mask3].mean(axis=1)

    val_losses = np.array([ENCODERS[e][1] for e in clean_order])
    w = 1.0 / val_losses
    w = w / w.sum()
    wmean4_8 = (stack8 * w).sum(axis=1)
    wmean4_33 = (stack33 * w).sum(axis=1)

    idx_top2 = [clean_order.index("chemprop"), clean_order.index("molformer_c3")]
    top2_8 = stack8[:, idx_top2].mean(axis=1)
    top2_33 = stack33[:, idx_top2].mean(axis=1)

    # Informational (LEAKY KERMT mixed in) — do NOT pick from these for
    # Phase 3. Reported only because they represent what would happen if
    # KERMT's train-set predictions carry into the 4653-compound feature.
    leaky_order = ["chemprop", "molformer_c3", "kermt", "attentivefp", "gatedgcn"]
    stack8L = stack(leaky_order, "log2fc_8p25_pred")
    stack33L = stack(leaky_order, "log2fc_33_pred")
    mean5_8 = stack8L.mean(axis=1)
    mean5_33 = stack33L.mean(axis=1)

    val_lossesL = np.array(
        [ENCODERS[e][1] if ENCODERS[e][1] is not None else 0.45 for e in leaky_order]
    )
    wL = 1.0 / val_lossesL
    wL = wL / wL.sum()
    wmean5_8 = (stack8L * wL).sum(axis=1)
    wmean5_33 = (stack33L * wL).sum(axis=1)

    idx_top3 = [
        leaky_order.index("chemprop"),
        leaky_order.index("molformer_c3"),
        leaky_order.index("kermt"),
    ]
    top3_8 = stack8L[:, idx_top3].mean(axis=1)
    top3_33 = stack33L[:, idx_top3].mean(axis=1)

    truth8 = truth_eval["log2fc_8p25"].values
    truth33 = truth_eval["log2fc_33"].values

    def mae_vs(pred: np.ndarray, truth_vec: np.ndarray) -> tuple[float, int]:
        mask = np.isfinite(truth_vec)
        n = int(mask.sum())
        return float(np.mean(np.abs(pred[mask] - truth_vec[mask]))), n

    clean_variants = [
        ("mean_4_clean", mean4_8, mean4_33, ""),
        ("mean_3_cp_mc3_afp", mean3_8, mean3_33, ""),
        ("weighted_val_loss_4_clean", wmean4_8, wmean4_33, ""),
        ("top2_cp_mc3", top2_8, top2_33, ""),
    ]
    leaky_variants = [
        ("mean_5_LEAKY", mean5_8, mean5_33, "includes KERMT (leaky)"),
        ("weighted_val_loss_5_LEAKY", wmean5_8, wmean5_33, "includes KERMT (leaky)"),
        ("top3_cp_mc3_kermt_LEAKY", top3_8, top3_33, "includes KERMT (leaky)"),
    ]

    for variant, p8, p33, note in clean_variants + leaky_variants:
        m8, n8 = mae_vs(p8, truth8)
        m33, n33 = mae_vs(p33, truth33)
        rows.append(
            {
                "variant": variant,
                "mae_8p25": m8,
                "n_8p25": n8,
                "mae_33": m33,
                "n_33": n33,
                "val_loss_pretrain": None,
                "note": note,
            }
        )

    df_out = pd.DataFrame(rows)
    df_out["mae_mean"] = (df_out["mae_8p25"] + df_out["mae_33"]) / 2
    df_out = df_out.sort_values("mae_mean").reset_index(drop=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUT_CSV, index=False)

    print("\n=== Per-variant log2fc MAE on pretrain val ∩ train+test set ===")
    print(df_out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved: {OUT_CSV}")


if __name__ == "__main__":
    main()

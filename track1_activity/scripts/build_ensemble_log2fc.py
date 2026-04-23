"""Phase 3 of issue #115: build the ensembled log2fc prediction parquet.

Strategy: inverse-val-loss weighted mean of 4 clean encoders (chemprop,
molformer_c3, attentivefp, gatedgcn). KERMT is excluded because its
training set overlaps other encoders' val set by construction, making
KERMT's predictions on any val-derived evaluation leaky — see
eval_log2fc_ensemble.py output.

Output: data/ensemble4_log2fc_predictions.parquet
        index=compound_id, columns=[log2fc_8p25_pred, log2fc_33_pred]
        same schema as chemprop_pretrain_log2fc_predictions.parquet
        so the run_train.py feature loader can swap it in place.

Usage:
    pixi run python track1_activity/scripts/build_ensemble_log2fc.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

ENCODERS_VAL_LOSS = {
    "chemprop": 0.3647,
    "molformer_c3": 0.5065,
    "attentivefp": 0.7012,
    "gatedgcn": 0.7478,
}
OUT_PATH = REPO_ROOT.joinpath("data", "ensemble4_log2fc_predictions.parquet")


def main() -> None:
    preds: dict[str, pd.DataFrame] = {}
    for enc in ENCODERS_VAL_LOSS:
        path = REPO_ROOT.joinpath("data", f"{enc}_pretrain_log2fc_predictions.parquet")
        if not path.exists():
            raise SystemExit(f"missing {path}")
        preds[enc] = pd.read_parquet(path)
        print(f"  loaded {enc}: {preds[enc].shape}")

    common = sorted(set.intersection(*[set(p.index) for p in preds.values()]))
    assert len(common) == 4653, f"expected 4653 common compound_ids, got {len(common)}"
    print(f"common compound_ids: {len(common)}")

    weights = np.array([1.0 / ENCODERS_VAL_LOSS[e] for e in ENCODERS_VAL_LOSS])
    weights = weights / weights.sum()
    for e, w in zip(ENCODERS_VAL_LOSS, weights):
        print(f"  weight[{e}] = {w:.4f}")

    enc_list = list(ENCODERS_VAL_LOSS.keys())
    out_cols = {}
    for col in ["log2fc_8p25_pred", "log2fc_33_pred"]:
        stack = np.stack(
            [preds[e].loc[common, col].to_numpy(dtype=np.float32) for e in enc_list],
            axis=1,
        )
        out_cols[col] = (stack * weights).sum(axis=1)

    out = pd.DataFrame(
        {
            "compound_id": common,
            "log2fc_8p25_pred": out_cols["log2fc_8p25_pred"],
            "log2fc_33_pred": out_cols["log2fc_33_pred"],
        }
    ).set_index("compound_id")
    out.to_parquet(OUT_PATH)

    print(f"Saved {out.shape} to {OUT_PATH}")
    print(
        f"  log2fc_8p25_pred: mean={out.log2fc_8p25_pred.mean():.3f} "
        f"std={out.log2fc_8p25_pred.std():.3f}"
    )
    print(
        f"  log2fc_33_pred:   mean={out.log2fc_33_pred.mean():.3f} "
        f"std={out.log2fc_33_pred.std():.3f}"
    )


if __name__ == "__main__":
    main()

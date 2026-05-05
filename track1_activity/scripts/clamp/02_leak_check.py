#!/usr/bin/env python
"""Behavioral leak check for CLAMP on PXR test compounds.

CLAMP was pretrained on PubChem18 (21k assays) + FS-Mol (ChEMBL27).
PXR assays very likely in pretrain corpus. We test how much PXR-specific
knowledge CLAMP has via zero-shot prediction with PXR-themed assay text.

Method:
1. Use forward_dense(test_smiles, pxr_assay_descriptions) to get
   per-(compound, assay-text) logits.
2. Map logits → predicted activity, compare with test pEC50 (cheating
   here for diagnostic only, since blinded test labels would normally
   be unknown — but here we compute MAE against train labels for
   compounds we have labels for, OR proxy via train/test split).

Interpretation:
- MAE < 0.50 → strong PXR knowledge in pretrain (test-internal label leak risk)
- MAE 0.55-0.65 → moderate (chemistry prior + PXR domain knowledge)
- MAE > 0.70 → just chemistry prior, treat as standard pretrained encoder

Note: this script uses train compounds (where we have labels) as
the validation set for the leak diagnosis. If CLAMP can predict train
pEC50 from PXR assay text *without seeing PXR labels during fine-tune*,
it has PXR knowledge in pretrain. The diagnostic is identical for
both train and test compounds.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
import torch


PXR_ASSAY_DESCRIPTIONS = [
    # Try multiple phrasings to see if model has PXR concept
    "PXR (Pregnane X Receptor) activation EC50: agonist potency at the human "
    "pregnane X receptor (NR1I2), measured by reporter assay in HepG2 cells.",
    "Activation of human pregnane X receptor (PXR / NR1I2) reporter assay.",
    "Activator of pregnane X receptor PXR.",
    "qHTS Assay for Activators of the Pregnane X Receptor signaling pathway.",
    "Compounds that activate the PXR nuclear receptor in cell-based reporter "
    "assays for ADMET screening.",
]


def fetch_train_with_labels(host: str, port: int, dbname: str) -> pd.DataFrame:
    conn = psycopg2.connect(dbname=dbname, host=host, port=port)
    df = pd.read_sql(
        """
        SELECT t.compound_id, c.std_smiles, t.pec50
        FROM train_activity t
        JOIN compounds c ON c.id = t.compound_id
        WHERE c.std_smiles IS NOT NULL AND t.pec50 IS NOT NULL
        ORDER BY t.compound_id;
        """,
        conn,
    )
    conn.close()
    return df


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-sample", type=int, default=500, help="random sample for speed; 0 = all"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--db-host", default="/tmp")
    parser.add_argument("--db-port", type=int, default=5433)
    parser.add_argument("--db-name", default="pxr_challenge")
    parser.add_argument(
        "--out",
        default="track1_activity/reports/clamp_leak_check.csv",
    )
    parser.add_argument(
        "--repo-root",
        default=os.environ.get("PXR_REPO", "/home/nagaet/pxr-iduction-challenge"),
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    out_path = repo_root.joinpath(args.out)

    import clamp

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = clamp.CLAMP(device=device)
    model.to(device)
    model.eval()

    df = fetch_train_with_labels(args.db_host, args.db_port, args.db_name)
    print(f"fetched {len(df)} labeled train compounds", file=sys.stderr)

    if args.n_sample > 0 and len(df) > args.n_sample:
        df = df.sample(n=args.n_sample, random_state=args.seed).reset_index(drop=True)
        print(f"sampled {len(df)}", file=sys.stderr)

    smiles = df["std_smiles"].tolist()
    pec50 = df["pec50"].to_numpy(dtype=np.float32)

    rows = []
    for desc_idx, desc in enumerate(PXR_ASSAY_DESCRIPTIONS):
        with torch.no_grad():
            # forward_dense returns (N_mol, N_assay) logits
            logits = model.forward_dense(smiles, [desc]).cpu().numpy().squeeze(-1)
        # logits sign: higher = active. Compare ranking with pec50.
        from scipy.stats import spearmanr, pearsonr

        rho, _ = spearmanr(logits, pec50)
        r, _ = pearsonr(logits, pec50)
        # crude MAE: linear-fit logits→pec50, compute residual MAE
        slope, intercept = np.polyfit(logits, pec50, 1)
        pred = slope * logits + intercept
        mae = float(np.mean(np.abs(pred - pec50)))

        print(
            f"desc[{desc_idx}] '{desc[:60]}...' "
            f"Sp={rho:+.3f} Pr={r:+.3f} fit_MAE={mae:.3f}"
        )
        rows.append(
            {
                "desc_idx": desc_idx,
                "description": desc,
                "spearman_rho": rho,
                "pearson_r": r,
                "fit_mae": mae,
                "n": len(df),
            }
        )

    res = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_path, index=False)
    print(f"wrote {out_path}")

    print("\nINTERPRETATION:")
    best_mae = res["fit_mae"].min()
    if best_mae < 0.50:
        print(f"  best fit_MAE={best_mae:.3f} — STRONG PXR knowledge in pretrain")
    elif best_mae < 0.65:
        print(f"  best fit_MAE={best_mae:.3f} — moderate PXR domain knowledge")
    else:
        print(f"  best fit_MAE={best_mae:.3f} — only chemistry prior, no leak concern")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env -S pixi run python
"""Canonical ensemble builder for Track 1.

This is the ONE canonical ensemble script. Previous versioned forks
(run_ensemble_v2.py, run_ensemble_v8.py, and the original run_ensemble.py)
have been moved to track1_activity/scripts/archive/. The v2/v8 naming
jumped (no v3-v7 ever existed) and created confusion about which script
was authoritative — this file resolves that by being the single source
of truth.

Key design choices that differ from the archived scripts:

1. **Explicit allow list**: ``ENSEMBLE_MODELS`` below hard-codes exactly
   which experiments enter the candidate pool. No silent pickup from
   DB queries. Previously the v8 script picked up whatever had OOF +
   a submission CSV, which allowed stale scaffold-split experiments
   (single_*, xgboost_mordred, catboost_mordred) to contaminate the
   pool. With an explicit list, adding or removing a candidate is
   a single-line audit trail.

2. **UMAP fold weights**: fold-based optimization uses
   ``umap_split_indices`` (matching the candidates' training split),
   not scaffold folds. The v2/v8 scripts optimized fold weights on
   scaffold folds even when the underlying OOF predictions were from
   UMAP folds — a subtle inconsistency that biased weight selection.

3. **No v* suffix in submission names**: submissions are
   ``ens_{strategy}.csv`` / DB name ``ens_{strategy}``. Previous
   ens_v7_* and ens_v8_* rows remain in the DB with an
   ``archived_ensemble_v7`` / ``archived_ensemble_v8`` ``model_type``
   prefix (see ``https://github.com/N283T/pxr-iduction-challenge/issues/172``) and are no longer
   generated.

All OOF RAE numbers annotated inline below are snapshots as of
2026-04-09. They will rot if any candidate is re-tuned — re-compute
from ``experiment_summary`` before trusting them for planning.

The candidate list is the union of:
  - ens_v7 UMAP-only members (dropping chemprop_scaffold and
    chemeleon_finetune per plan)
  - The tuned mordred_jazzy model (PR #45)
  - The best tuned 5-task multitask chemprop (chemprop_multitask5_umap
    _aux0.0_tuned; other aux-weight and 2-task variants are intentionally
    excluded)

Threshold: every model in the list has OOF RAE < 0.68 (project policy
for ensemble candidates).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.joinpath("src")))

import numpy as np
import pandas as pd
import psycopg2
from scipy.optimize import minimize

from data import DB_PARAMS, load_test_smiles, load_train_smiles_target
from evaluate import (
    compute_metrics,
    load_oof_predictions,
    record_experiment,
)
from splits import umap_split_indices

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SUBMISSION_DIR = REPO_ROOT.joinpath("track1_activity", "submissions")
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# The allow list — audit this before every ensemble run
# ---------------------------------------------------------------------------

# OOF RAE values below are snapshots from 2026-04-09 and may drift after
# any re-tuning. They are annotations for quick review, not live values.
ENSEMBLE_MODELS: tuple[str, ...] = (
    # 2026-04-18 PM prune: dropped all 13 zero-w_mae LGBMs (fingerprints +
    # embedding LGBMs + rdkit_desc_full + gap1.0 variants), dropped the 3
    # mixed-split LGBMs (LB flat on the morning submission, optimiser
    # re-weights on a redundant family), dropped gin/graphgps (near-zero
    # weight, OOF-weakest graph nets), replaced tabpfn_mordred_jazzy_umap
    # with tabpfn_2d_full_boltz_umap (workhorse upgrade: 2D full + Boltz
    # Tier-0/1 + pose-Jazzy, OOF RAE 0.5303 vs 0.5511 tuned). See
    # track1_activity/scripts/eda_cv_prep/14_prune_dryrun.py for the
    # 5-pool bake-off + 3-variant LGBM check that informed this list.
    #
    # 2026-04-19 PM prune (this edit): dropped 3 redundant members before
    # LB submission:
    #   chemprop_multitask5_umap_aux0.0_tuned (r=0.94 with chemprop_optuna,
    #     aux=0 makes it essentially a second chemprop_optuna seed)
    #   residual_physprop+mordred_umap (r=0.94 with tabpfn_2d_full_boltz,
    #     old (Apr 7) training, subsequent upgrades made its inputs a
    #     subset of the tabpfn workhorse)
    #   tabpfn_chemeleon_umap (r=0.94 with tabpfn_2d_full_boltz; CheMeleon
    #     signal insufficient to compensate for redundancy)
    # All three had caruana weights 0.03-0.06 (combined ~0.13 share).
    # Pool size 12 -> 9.
    # 2026-04-21 PM drop: 4 direct-GNN members all at caruana weight=0
    # in the post-GatedGCN 12-pool. Each was redundant with its
    # pretrain-embed sibling (chemprop_optuna/chemeleon vs
    # tabpfn_chemprop_pretrain_embed, attentivefp_optuna vs
    # tabpfn_attentivefp_pretrain_embed, gatedgcn_pretrain_finetune_frozen
    # vs tabpfn_gatedgcn_pretrain_embed). Bakeoff showed removing all 4
    # together IMPROVES OOF (-0.0005), whereas keeping any single member
    # REGRESSED (+0.002) -- caruana's bagged search is non-monotone in
    # partial-direct-GNN presence. Framework + DB experiments stay;
    # dropped from allow-list only. Pool size 12 -> 8.
    #   chemprop_optuna_umap (dropped)
    #   chemprop_chemeleon_umap (dropped)
    #   attentivefp_optuna_umap (dropped)
    #   gatedgcn_pretrain_finetune_frozen_umap (dropped)
    # --- Foundation tabular (2) ---
    # tabpfn_2d_full_boltz_umap replaced 2026-04-20 by its
    # log2fc_pred-augmented variant (same 1801 base features + 2
    # predicted log2_fc scalars from the chemprop pretrain model).
    # 2026-04-21 PM: that 1803-feature variant further superseded by
    # chemeleon (300d foundation FP) + 2d_full_boltz_log2fc_pred =
    # 2103d concat, discovered via mix-feature bakeoff. Same feature
    # universe but with chemeleon's foundation-model signal stacked in.
    # Single-model OOF MAE 0.4427 -> 0.4213 (Δ -0.021, strict
    # supersession). Post-swap 8-pool caruana MAE 0.4181 (Δ -0.0065
    # vs 0.4246) with cheme_2df carrying caruana weight 0.388 (pool-top).
    # Chemeleon alone with TabPFN was empirically iffy, but mixed with
    # the 2D/Boltz descriptors it adds decorrelating signal.
    #
    # 2026-04-25 SWAP: default cheme_2df_lf -> seedNens variant (Plan A
    # multi-seed pretrain). chemprop pretrain re-run with 5 seeds [43..46]
    # then 5 more [47..51], log2fc_pred parquets averaged per row. Same
    # 2103d feature shape, only the 2 log2fc cols change to a multi-seed
    # mean.
    #
    # 5-seed (id=31, rank 1, MAE 0.4084 / Sp 0.846): 9-pool caruana MAE
    # 0.4150 -> 0.4034 (Δ -0.0116) and Sp 0.840 -> 0.840.
    #
    # 10-seed (id=32 candidate, 2026-04-25 evening, this entry):
    # additional 5 seeds [47..51]. Single-model OOF MAE 0.4068 -> 0.4056
    # (Δ -0.0012) AND Sp 0.836 -> 0.840 (Δ +0.004). Caruana MAE 0.4034
    # -> 0.4019 (Δ -0.0015) AND Sp 0.8403 -> 0.8424 (Δ +0.0021).
    # Spearman gain matches the gap to rank 2 (sia 0.850 vs us 0.846)
    # — primary motivation for the extension.
    #
    # Same SWAP semantics as 5-seed: structurally an upgrade of the same
    # encoder family, residual r ~0.99 with the prior 5-seed variant.
    #
    # 2026-04-26 Phase 4 (log2fc Optuna PR): SWAPPED for
    # `tabpfn_cheme_2d_full_boltz_log2fc_pred_optuna_trial10_seed5ens_umap_default`.
    # Optuna re-tuned chemprop pretrain hparams (16-trial TPE +
    # multi-seed validate) against downstream TabPFN OOF MAE.
    # Trial 10 5-seed: hidden_dim=384, mp_dropout~0, lr=4.6e-4, w_33=0.98.
    # Single-model OOF: MAE 0.4055 -> 0.3959 (Δ -0.0096), Sp 0.8380 -> 0.8425.
    # Caruana_bag20 (with t11 ADD below): MAE 0.4019 -> 0.3920 (Δ -0.0099),
    # Sp 0.8424 -> 0.8504 (Δ +0.0080, ties sia 0.8506). See
    # 31_optuna_pretrain_swap.py + run_chemprop_pretrain_optuna.py.
    "tabpfn_cheme_2d_full_boltz_log2fc_pred_optuna_trial10_seed5ens_umap_default",
    # 2026-04-27: trial11 ADD removed after Phase 4 LB regression (id=38
    # MAE +0.0031 / Sp -0.0057 vs id=32 baseline despite OOF Δ -0.0099 /
    # +0.0080). Hypothesis: ADD 2 correlated cheme members concentrated
    # chemprop-family weight 0.59 -> 0.85, biasing the ensemble in the
    # wrong direction on the test distribution. Now testing the more
    # conservative swap_default_t10 (single SWAP, no ADD, 9 pool members)
    # as LB A/B vs id=38 to isolate which change was the culprit.
    # bakeoff OOF MAE 0.3971 / Sp 0.8451 (vs Phase 4 0.3920 / 0.8504).
    # See feedback_phase4_oof_lb_full_reverse.md.
    # "tabpfn_cheme_2d_full_boltz_log2fc_pred_optuna_trial11_seed5ens_umap_default",
    # TabPFN on chemprop pretrain encoder fingerprint (256d).
    # Buterez 2024 strategy-3 (LF embedding as side feature).
    # OOF MAE 0.4373 / Spearman 0.8102 -- pool single-best.
    # Correlated r=0.962 with the log2fc_pred variant above (both
    # carry pretrain signal), caruana spreads weight between them.
    "tabpfn_chemprop_pretrain_embed_umap_default",
    # 2026-04-27 NULL: chemprop_pretrain_optuna_trial10_embed ADD test
    # (384d frozen embed from Optuna Trial 10 ckpt) failed at LB despite
    # passing OOF gate. Bakeoff caruana_bag20 Δ MAE -0.0052 (just above
    # bag noise +-0.003); single-run 10-pool was bag-noise tied. LB id=40
    # MAE 0.4110 / Sp 0.8437 vs id=32 baseline 0.4080 / 0.8465 (Δ MAE
    # +0.0030 / Sp -0.0028, rank 1 -> 3, gap to sia +0.0027 -> +0.0058).
    # Pattern matches Phase 4 trial10/11 log2fc_pred ADD (id=38 same
    # family-concentration reverse amp). Confirms gate 2 r > 0.85 rule
    # really does block LB-meaningful adds even when caruana finds weight
    # for the new member. Framework retained (33_chemprop_embed_swap.py +
    # eval_chemprop_optuna_embed.py + chemprop_pretrain_optuna_trial{N}_
    # embed feature pattern). See feedback_chemprop_optuna_embed_null.md.
    # "tabpfn_chemprop_pretrain_optuna_trial10_embed_umap",
    # --- Boltz trunk embedding pool (2) ---
    # 1024-dim pooled trunk embeddings (s_prot_mean 384 + s_lig_mean 384 +
    # z_interface_mean/max 128+128). Two TabPFN variants retained: core
    # pocket and all protein x ligand pairs. Single OOF MAEs: 0.4861 /
    # 0.4860. Inter-correlation 0.96-0.99, so they only add diversity via
    # Caruana-style weight spreading.
    # 2026-04-21 PM drop: lgbm_pooled_boltz_umap (MAE 0.5116, wt 0.011)
    # swapped out for tabpfn_gatedgcn_pretrain_embed_umap_default in the
    # h512 GatedGCN PR. LGBM variant was largely redundant with the TabPFN
    # variant on the same feature; bakeoff showed swap is the lowest-
    # regression way to seat GatedGCN embed (MAE 0.4251, Δ +0.0009, within
    # fold std 0.027 = noise). Framework stays in the DB and run_train.py.
    "tabpfn_pooled_boltz_umap_default",
    "tabpfn_pooled_boltz_allpairs_umap_default",
    # 2026-04-20 PM drop: peft_molformer_xl_lora_r32a64_umap_default (added
    # in PR #95) regressed LB rank 9 -> 10 (RAE 0.5545 -> 0.5565, MAE 0.4414
    # -> 0.4430) despite OOF -0.0087. Classic OOF/LB disconnect. Framework
    # + experiment id=581 stay in DB; just dropped from the allow-list.
    # 2026-04-20 PM drop: chemprop_relative_aux_umap_default (added in PR
    # #97) single-model OOF regressed on ALL metrics vs chemprop_optuna
    # baseline (MAE 0.5208 -> 0.5457, Spearman 0.7252 -> 0.7118). Caruana
    # confirmed rejection: wt=0, pool MAE unchanged. FMGCL aux on PXR is
    # falsified; do not revisit this loss family without new evidence.
    # Framework (losses.py + run_chemprop_relative_aux.py) stays on main.
    # --- MoLFormer-c3 pretrain embed + TabPFN (1) ---
    # Transformer-encoder-family analog of tabpfn_chemprop_pretrain_embed.
    # Same recipe (Buterez 2024 strategy-3: pretrain on log2_fc ->
    # frozen embedding -> TabPFN) but different backbone family
    # (MoLFormer-c3 transformer vs chemprop D-MPNN). Single-model OOF
    # MAE 0.4753 (chemprop version 0.4373, but decorrelation hypothesis
    # justifies pool addition). Name lacks "_default" suffix because
    # run_train.py used its 20-trial Optuna default, not --trials 0.
    # PR #98.
    "tabpfn_molformer_c3_pretrain_embed_umap",
    # --- KERMT/GROVER pretrain embed + TabPFN ---
    # Graph-transformer family (continued-pretrain of GROVER_base on
    # single_concentration log2_fc, frozen, then TabPFN with
    # ignore_pretraining_limits=True for 3200d). Added as a
    # decorrelating 11th pool member. Previous pretrain-embed backbones:
    # chemprop (GNN) and MoLFormer-c3 (transformer).
    # Single-model OOF MAE 0.4485, Pearson r vs others: chemprop 0.945,
    # molformer_c3 0.918, 2d_full_boltz 0.943. PR <TBD>.
    "tabpfn_kermt_pretrain_embed_umap_default",
    # --- AttentiveFP pretrain embed + TabPFN ---
    # Graph-attention family (PyG AttentiveFP pretrained on
    # single_concentration log2_fc via PR #79 checkpoint, then 512d
    # readout frozen into TabPFN). Added as decorrelating 12th pool
    # member. Complements chemprop (D-MPNN), molformer_c3 (transformer),
    # kermt (graph-transformer).
    # Single-model OOF MAE 0.4844, Pearson r vs existing:
    # chemprop 0.937, molformer_c3 0.922, 2d_full_boltz 0.941, kermt 0.920.
    # 12-pool caruana_bag20 OOF MAE 0.4242 (-0.0026 vs 11-pool 0.4268),
    # attentivefp caruana weight 0.0223. PR #104.
    "tabpfn_attentivefp_pretrain_embed_umap_default",
    # --- GatedGCN pretrain embed + TabPFN (h512, added via swap) ---
    # Gated edge-conditioned message-passing family (PyG ResGatedGraphConv
    # pretrained on single_concentration log2_fc, hidden_dim=512,
    # batch=64, 74 epochs until early stop, val_loss 0.7478). 512d
    # global_mean_pool frozen into TabPFN. Single-model OOF MAE 0.4740
    # (h128 was 0.4902; re-pretrain with 4x hidden bought -0.016).
    # Pearson r vs existing: chemprop 0.941, molformer_c3 0.912
    # (decorrelation champion), 2d_full_boltz 0.941, kermt 0.941,
    # attentivefp 0.934. Caruana weight 0.036 in the swap config.
    # Added via swap with lgbm_pooled_boltz_umap (12->12 members).
    # 12-pool MAE 0.4251 (+0.0009 vs 0.4242 = noise-level within fold
    # std 0.027).
    "tabpfn_gatedgcn_pretrain_embed_umap_default",
    # --- top500 LGBM-selected TabPFN (2026-04-22 evening) ---
    # Same feature base as cheme_2d_full_boltz_log2fc_pred (2103d) but
    # reduced to top-500 features per-fold via LGBM gain. Per-fold
    # selection = leak-free. Single-model OOF MAE 0.4181 vs production
    # baseline 0.4212 (Δ -0.0031) -- strict supersession on all metrics.
    # Bakeoff (16_pool_bakeoff_top500.py, add_top500_drop_both_boltz_strategy3):
    # 9-pool ADD top500 + DROP c_concat + DROP raw_plus_pretrain_concat
    # delivered caruana OOF MAE 0.4150 (-0.0031 vs 10-pool baseline 0.4181),
    # top500 caruana weight 0.2427 (pool-top). The Boltz-strategy-3 pair
    # we added this afternoon was net +0.0004 weak-signal; top500 fully
    # displaces them.
    # Mechanism: TabPFN v7 is pretrained on d<=500, and manual LGBM-gain
    # feature selection outperforms TabPFN's internal random subsampling
    # when fed large-d inputs.
    # 2026-04-25 SWAP: top500 variant rebuilt on the seedNens parquet
    # (15_tabpfn_topk_proper_cv.py with --feature
    # cheme_2d_full_boltz_log2fc_pred_seed{N}ens). Per-fold LGBM-gain
    # selection from the seed-averaged 2103d feature matrix.
    #
    # 5-seed: single-model OOF MAE 0.4181 -> 0.3988 (Δ -0.0193, first
    # sub-0.40). 10-seed: 0.3988 -> 0.3968 (Δ -0.0020, additional gain)
    # AND Spearman 0.843 -> 0.846 (Δ +0.003, matches LB Sp).
    #
    # Both swaps together (default + top500 -> 10-seed): 9-pool
    # caruana MAE 0.4034 -> 0.4019 (Δ -0.0015) AND Sp 0.8403 -> 0.8424
    # (Δ +0.0021).  See 29_seed10ens_swap.py for the bakeoff.
    "tabpfn_cheme_2d_full_boltz_log2fc_pred_seed10ens_top500_umap",
    # 2026-04-23 dropped: tabpfn_mixed_pool_top500_umap.
    # Two LB submissions regressed (+0.0051, +0.0091) despite OOF 0.4113 best.
    # Per-fold LGBM-gain on 8375d mega-concat overfits fold structure.
    # See memory:feedback_mixed_top500_oof_lb_disconnect and issue #100 log.
    # Framework (script 18_mixed_compression_top500.py) stays for future
    # research under a proper nested-CV selection protocol.
    #
    # 2026-04-23 also tested: tabpfn_oe_cheme_2d_full_boltz_log2fc_pred_top500_umap
    # LEAK version (ROCS self-match enabled). Single OOF 0.4071 / Ensemble
    # 0.4103 / LB 0.4243 -- confirmed the OOF improvement does NOT transfer
    # and actually hurts LB. Clean (self-skipped) ROCS contributed only
    # Δ -0.0007 OOF = noise, not pool-worthy. See memory:
    # feedback_self_match_leak_in_similarity_features.
    # 2026-04-24 dropped: tabpfn_boltz2_tabular_tier0_umap_default (added
    # in PR #117). 17 scalars (6 affinity + 9 confidence + 2 geometry)
    # passed BOTH gates at OOF level: 9-pool 0.4150 -> 10-pool 0.4130
    # (Δ -0.0020) AND Codex new-family min residual r ≤ 0.85 (actual
    # r 0.71-0.82 to all 9 existing members). But LB A/B (id=29, PR #117
    # submission) regressed: LB MAE 0.4149 -> 0.4189 (Δ +0.0040, 2x
    # reverse amplification of the OOF gain). rank 3 held but gap to
    # rank 2 widened 0.006 -> 0.010.
    # Follows memory feedback_no_revert_prefer_ensemble_drop: drop from
    # allow-list, keep single-model experiment (id=1332) + feature
    # handler in run_train.py + Phase A/B/C bakeoff scripts. mordred3d
    # counterpart (id=1353) also stays in DB.
    # 2026-05-04 Phase 3 (Codex priority #1) Uni-Mol v2 log2fc seed5ens:
    # 5-seed pretrain ensemble (seeds 42-46), gate 1 single OOF MAE 0.4844
    # (Δ -0.004 vs seed=42 0.4885), gate 2 FAIL (min r 0.8994 to pool;
    # threshold 0.85), gate 3 ADD test caruana weight=0.0000 (not picked).
    # OOF Δ -0.005 attributed to bag noise (memory
    # feedback_multi_seed_weight_threshold: multi-seed needs wt >= 0.2 to
    # propagate). Family share 0.90 in LB regress zone. NOT added to pool.
    # Single-experiment id=2250 + feature handler in run_train.py kept.
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_models(
    y_train: np.ndarray, n_test: int
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Load OOF + test predictions for every model in ``ENSEMBLE_MODELS``.

    Fails loudly if anything is missing: the allow list is authoritative.
    Weight optimization assumes at least two candidates (a single-model
    "ensemble" is just the model itself and all six strategies collapse
    to the same weights), so a pool of < 2 raises.
    """
    if len(ENSEMBLE_MODELS) < 2:
        raise RuntimeError(
            f"ENSEMBLE_MODELS has {len(ENSEMBLE_MODELS)} entries; need >= 2 "
            "for weight optimization to be meaningful."
        )

    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    cur.execute(
        "SELECT id, name, submission_path FROM experiments WHERE name = ANY(%s)",
        (list(ENSEMBLE_MODELS),),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    rows_by_name = {r[1]: r for r in rows}
    missing = [n for n in ENSEMBLE_MODELS if n not in rows_by_name]
    if missing:
        raise RuntimeError(
            f"ENSEMBLE_MODELS not found in experiments table: {missing}. "
            "Fix the allow list or the DB before re-running."
        )

    names: list[str] = []
    oofs: list[np.ndarray] = []
    tests: list[np.ndarray] = []

    for name in ENSEMBLE_MODELS:
        exp_id, _, sub_path = rows_by_name[name]

        # OOF
        oof = load_oof_predictions(exp_id)
        if oof is None:
            raise RuntimeError(f"{name}: no OOF predictions in DB")
        if len(oof) != len(y_train):
            raise RuntimeError(
                f"{name}: OOF length {len(oof)} != train length {len(y_train)}"
            )

        # Test — validate CSV has the expected column and length
        if sub_path is None:
            raise RuntimeError(f"{name}: experiments.submission_path is NULL")
        csv_path = REPO_ROOT.joinpath(sub_path)
        if not csv_path.exists():
            raise RuntimeError(f"{name}: submission CSV not found at {csv_path}")
        test_df = pd.read_csv(csv_path)
        if "pEC50" not in test_df.columns:
            raise RuntimeError(
                f"{name}: submission CSV {csv_path} has no 'pEC50' column "
                f"(got {list(test_df.columns)})"
            )
        if len(test_df) != n_test:
            raise RuntimeError(
                f"{name}: submission CSV has {len(test_df)} rows, "
                f"expected {n_test} (test_activity length)"
            )
        test_pred = test_df["pEC50"].to_numpy()

        names.append(name)
        oofs.append(oof)
        tests.append(test_pred)

    oof_matrix = np.column_stack(oofs)
    test_matrix = np.column_stack(tests)
    return names, oof_matrix, test_matrix


# ---------------------------------------------------------------------------
# Weight optimization strategies
# ---------------------------------------------------------------------------


def normalize_weights(w: np.ndarray) -> np.ndarray:
    """L1-normalize the absolute weights so they sum to 1.

    Raises if the optimizer collapsed to all-zero weights — that outcome
    is pathological and silently turning it into uniform weights would
    hide the failure behind an ensemble that looks like a legitimate
    weighted blend. ``ens_simple_avg`` already covers the "everything
    uniform" strategy as a distinct, labeled submission.
    """
    w_abs = np.abs(w)
    total = w_abs.sum()
    if total < 1e-12:
        raise RuntimeError(
            "normalize_weights: optimizer produced all-zero weights "
            "(|sum| < 1e-12). This is pathological — refusing to fall "
            "back to uniform weights silently. Investigate the objective."
        )
    return w_abs / total


def _check(result, label: str) -> None:
    """Raise if scipy.optimize.minimize did not converge.

    The previous implementation only printed a warning, which meant
    non-converged weights silently flowed into the submission CSV and
    DB record. For a 20-dim Nelder-Mead at maxiter=50000 a non-success
    return is rare but plausible, and its weights are untrustworthy —
    stop the run and surface the failure loudly instead.
    """
    if not result.success:
        raise RuntimeError(
            f"{label}: optimizer did not converge ({result.message}). "
            "Refusing to use the last iterate as production weights."
        )


def _rae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(
        np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true - y_true.mean()))
    )


def optimize_vanilla(oof: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = oof.shape[1]

    def obj(w):
        return _rae(y, oof @ normalize_weights(w))

    result = minimize(
        obj,
        np.ones(n) / n,
        method="Nelder-Mead",
        options={"maxiter": 50000, "xatol": 1e-8, "fatol": 1e-8},
    )
    _check(result, "vanilla")
    return normalize_weights(result.x)


def optimize_l2(oof: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    n = oof.shape[1]
    equal = np.ones(n) / n

    def obj(w):
        wn = normalize_weights(w)
        return _rae(y, oof @ wn) + alpha * np.sum((wn - equal) ** 2)

    result = minimize(
        obj,
        equal.copy(),
        method="Nelder-Mead",
        options={"maxiter": 50000, "xatol": 1e-8, "fatol": 1e-8},
    )
    _check(result, f"l2(alpha={alpha})")
    return normalize_weights(result.x)


def optimize_caruana(
    oof: np.ndarray,
    y: np.ndarray,
    *,
    n_iter: int = 100,
    init_top_n: int = 3,
    bag_frac: float = 0.5,
    n_bags: int = 20,
    seed: int = 42,
    sample_weight: np.ndarray | None = None,
) -> np.ndarray:
    """Caruana 2004 forward stepwise selection with replacement,
    sorted initialization, and bagged selection.

    Motivation: continuous weight optimization (vanilla / L2) concentrates
    weight onto whichever member has the best OOF, regardless of how
    correlated it is with other strong members. When that member's OOF
    advantage does not fully reflect LB behaviour, the optimiser
    overpays and the blend regresses. Caruana selection uses discrete
    counts and bagging over library subsets, which naturally spreads
    weight across correlated-but-strong members.

    When ``sample_weight`` is provided, MAE is replaced by importance-
    weighted MAE (sum(w * |err|) / sum(w)) at every selection step.
    This biases caruana to pick blends that perform well on test-like
    train compounds, addressing the test-distribution shift on PXR.

    Returns a simplex-normalized weight vector over all input columns.
    Models not chosen in any bag receive weight 0.
    """
    rng = np.random.RandomState(seed)
    N, M = oof.shape
    counts = np.zeros(M, dtype=np.int64)

    if sample_weight is None:
        sw = np.ones(N, dtype=np.float64)
    else:
        sw = np.asarray(sample_weight, dtype=np.float64)
        if sw.shape != (N,):
            raise ValueError(
                f"sample_weight shape {sw.shape} != ({N},) (must match OOF rows)"
            )
    sw_total = sw.sum()
    sw_col = sw[:, None]

    bag_size = max(init_top_n + 1, int(M * bag_frac))
    for _ in range(n_bags):
        members = rng.choice(M, size=bag_size, replace=False)
        bag_arr = oof[:, members]  # (N, bag_size)
        bag_maes = np.sum(sw_col * np.abs(bag_arr - y[:, None]), axis=0) / sw_total
        sort_idx = np.argsort(bag_maes)
        top = sort_idx[:init_top_n]

        bag_counts = np.zeros(bag_size, dtype=np.int64)
        if init_top_n > 0:
            bag_counts[top] = 1
            current_sum = bag_arr[:, top].sum(axis=1)
        else:
            current_sum = np.zeros(N)
        current_count = int(bag_counts.sum())

        for _ in range(n_iter):
            cand_pred = (current_sum[:, None] + bag_arr) / (current_count + 1)
            cand_maes = (
                np.sum(sw_col * np.abs(cand_pred - y[:, None]), axis=0) / sw_total
            )
            best = int(np.argmin(cand_maes))
            bag_counts[best] += 1
            current_sum = current_sum + bag_arr[:, best]
            current_count += 1

        for idx_in_bag, global_idx in enumerate(members):
            counts[global_idx] += int(bag_counts[idx_in_bag])

    total = int(counts.sum())
    if total == 0:
        raise RuntimeError("Caruana selection produced zero total count")
    return counts.astype(np.float64) / total


def optimize_fold(
    oof: np.ndarray, y: np.ndarray, splits: list, alpha: float = 0.0
) -> np.ndarray:
    n = oof.shape[1]
    equal = np.ones(n) / n
    per_fold = []

    for i, (_, val_idx) in enumerate(splits):
        ov = oof[val_idx]
        yv = y[val_idx]

        def obj(w):
            wn = normalize_weights(w)
            r = _rae(yv, ov @ wn)
            if alpha > 0:
                r += alpha * np.sum((wn - equal) ** 2)
            return r

        result = minimize(
            obj,
            equal.copy(),
            method="Nelder-Mead",
            options={"maxiter": 50000, "xatol": 1e-8, "fatol": 1e-8},
        )
        _check(result, f"fold_{i}(alpha={alpha})")
        per_fold.append(normalize_weights(result.x))

    # Average the per-fold weight vectors, then re-normalize. Each
    # per_fold entry is already simplex-normalized (sums to 1), so the
    # mean also sums to 1 in exact arithmetic; the final division
    # exists only to scrub floating-point drift.
    avg = np.mean(per_fold, axis=0)
    return avg / avg.sum()


# ---------------------------------------------------------------------------
# Evaluation and submission
# ---------------------------------------------------------------------------


def evaluate_and_record(
    name: str,
    weights: np.ndarray,
    oof_matrix: np.ndarray,
    y_train: np.ndarray,
    test_matrix: np.ndarray,
    test_df: pd.DataFrame,
    model_names: list[str],
) -> dict:
    blended_oof = oof_matrix @ weights
    metrics = compute_metrics(y_train, blended_oof)

    print(f"\n  {name}:")
    print(
        f"    OOF RAE={metrics['RAE']:.4f}  MAE={metrics['MAE']:.4f}  "
        f"R2={metrics['R2']:.4f}  Spearman={metrics['Spearman_R']:.4f}"
    )

    # Display threshold: 0.02 keeps the "significant" list focused on
    # models with >= 2% weight. Below that, contributions are within
    # normalization noise for a 20-model pool (1/20 = 5% uniform
    # baseline, so 0.02 is under half of uniform).
    significant = sorted(
        [(n, w) for n, w in zip(model_names, weights) if w > 0.02],
        key=lambda x: -x[1],
    )
    print(f"    Weights ({len(significant)} significant):")
    for n, w in significant:
        print(f"      {n:<45} {w:.4f}")

    blended_test = test_matrix @ weights
    sub = pd.DataFrame(
        {
            "SMILES": test_df["smiles"],
            "Molecule Name": test_df["molecule_name"],
            "pEC50": blended_test,
        }
    )
    sub_filename = f"{name}.csv"
    sub.to_csv(SUBMISSION_DIR.joinpath(sub_filename), index=False)

    # Ensemble names (``ens_*``) are stable across runs, so we pass
    # ``on_conflict_replace=True`` to replace any prior row + its cascaded
    # cv_results / oof_predictions. Regular experiment names still error on
    # duplicate insert. The CSV is written regardless so a hard DB failure
    # is recoverable from disk.
    weight_dict = {n: float(w) for n, w in zip(model_names, weights)}
    try:
        record_experiment(
            name=name,
            description=f"Canonical ensemble ({name})",
            model_type="ensemble",
            feature_set="weighted_blend",
            hyperparameters={
                "weights": weight_dict,
                "strategy": name.replace("ens_", ""),
                "pool_size": len(model_names),
                "pool_models": list(model_names),
            },
            fold_metrics=[metrics],
            submission_path=f"track1_activity/submissions/{sub_filename}",
            notes=f"OOF RAE={metrics['RAE']:.4f}, canonical (UMAP-only)",
            on_conflict_replace=True,
        )
    except Exception as exc:  # noqa: BLE001 — DB errors are diverse
        print(
            f"    [ERROR] {name}: record_experiment failed ({type(exc).__name__}: "
            f"{exc}). CSV still on disk at submissions/{sub_filename}; continuing."
        )

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("Loading data...")
    train_df = load_train_smiles_target()
    test_df = load_test_smiles()
    y_train = train_df["pec50"].to_numpy()

    print(f"Train rows: {len(y_train)}, test rows: {len(test_df)}")
    print(f"Allow list: {len(ENSEMBLE_MODELS)} models")

    model_names, oof_matrix, test_matrix = load_models(y_train, n_test=len(test_df))
    print(f"\nLoaded {len(model_names)} models with aligned OOF + test.")

    # Print per-model OOF RAE as a sanity check
    for i, name in enumerate(model_names):
        rae_i = _rae(y_train, oof_matrix[:, i])
        print(f"  {name:<48} OOF RAE = {rae_i:.4f}")

    # UMAP folds for fold-based weight optimization (matches candidates' split)
    print("\nBuilding UMAP folds for fold-based weight optimization...")
    umap_splits = umap_split_indices(train_df["smiles"].tolist(), n_splits=5, seed=42)

    n = len(model_names)
    equal_w = np.ones(n) / n

    print("\n" + "=" * 70)
    print("  STRATEGY COMPARISON")
    print("=" * 70)

    results: dict[str, dict] = {}

    # 1. Simple average
    results["simple_avg"] = evaluate_and_record(
        "ens_simple_avg",
        equal_w,
        oof_matrix,
        y_train,
        test_matrix,
        test_df,
        model_names,
    )

    # 2. Vanilla unconstrained
    w_vanilla = optimize_vanilla(oof_matrix, y_train)
    results["vanilla"] = evaluate_and_record(
        "ens_vanilla",
        w_vanilla,
        oof_matrix,
        y_train,
        test_matrix,
        test_df,
        model_names,
    )

    # 3. L2-regularized
    for alpha in (0.05, 0.1, 0.3, 0.5):
        w = optimize_l2(oof_matrix, y_train, alpha=alpha)
        results[f"l2_a{alpha}"] = evaluate_and_record(
            f"ens_l2_a{alpha}",
            w,
            oof_matrix,
            y_train,
            test_matrix,
            test_df,
            model_names,
        )

    # 4. Fold-based (UMAP folds)
    w_fold = optimize_fold(oof_matrix, y_train, umap_splits, alpha=0.0)
    results["fold"] = evaluate_and_record(
        "ens_fold",
        w_fold,
        oof_matrix,
        y_train,
        test_matrix,
        test_df,
        model_names,
    )

    # 5. Fold-based + L2 (UMAP folds)
    for alpha in (0.1, 0.3):
        w = optimize_fold(oof_matrix, y_train, umap_splits, alpha=alpha)
        results[f"fold_l2_a{alpha}"] = evaluate_and_record(
            f"ens_fold_l2_a{alpha}",
            w,
            oof_matrix,
            y_train,
            test_matrix,
            test_df,
            model_names,
        )

    # 6. Caruana 2004 forward stepwise selection with bagging
    # Structural regularization: discrete count-based weights + bagged
    # library subsets prevent any single model from capturing > ~0.3
    # weight, which mitigates the OOF-LB gap from correlated members.
    # See docs/papers/shotgun_raw/shotgun.txt for the paper.
    w_caru = optimize_caruana(
        oof_matrix,
        y_train,
        n_iter=100,
        init_top_n=3,
        bag_frac=0.5,
        n_bags=20,
        seed=42,
    )
    results["caruana_bag20"] = evaluate_and_record(
        "ens_caruana_bag20",
        w_caru,
        oof_matrix,
        y_train,
        test_matrix,
        test_df,
        model_names,
    )

    # Summary table
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(
        f"  {'Strategy':<25}  {'OOF RAE':>8}  {'OOF MAE':>8}  {'OOF R2':>8}  {'Spearman':>9}"
    )
    print("  " + "-" * 66)
    sorted_results = sorted(results.items(), key=lambda kv: kv[1]["RAE"])
    for strategy, m in sorted_results:
        print(
            f"  {strategy:<25}  {m['RAE']:>8.4f}  {m['MAE']:>8.4f}  "
            f"{m['R2']:>8.4f}  {m['Spearman_R']:>9.4f}"
        )

    best_strategy, best_metrics = sorted_results[0]
    print(f"\n  Best (by OOF RAE): {best_strategy} = {best_metrics['RAE']:.4f}")
    print(
        "  LB RAE reference: 0.62 (ens_v7, 2026-04-07 snapshot — "
        "update after next submission)"
    )
    print(
        "\n  Remember: lower OOF RAE may not translate to better LB if "
        "the optimizer overfits weights."
    )


if __name__ == "__main__":
    main()

# Track 1 Current State

Last reviewed: 2026-05-17 JST.

This summary is based on GitHub issue #100, the current production ensemble
script, and the latest local leaderboard snapshots.

## One-Sentence Status

Track 1 Phase 1 is effectively in hold mode: the strongest useful model family
has been found, recent small local gains have not transferred to the public
leaderboard, and the next high-value step is to wait for Phase 2 labels before
fitting more calibration or local-gain variants.

## Latest Public-LB Context

Latest local snapshot checked:
`docs/leaderboards/activity/leaderboard_2026-05-16_2141JST.csv`.

Visible N283T row:

| Rank | MAE | RAE | R2 | Spearman |
|---:|---:|---:|---:|---:|
| 5 | 0.407730 | 0.512323 | 0.678512 | 0.844763 |

Recent internal submission anchors from issue #100:

| id | Submission | Public result | Interpretation |
|---:|---|---|---|
| 55 | `ens_id51_top500_potent46_t40_soft_g35` | MAE 0.407080, rank 3 | Best recent practical anchor. |
| 57 | `ens_id51_top500_potent46_t40_soft_g50` | MAE 0.407389, rank 4 | Slightly worse than id55, still trusted. |
| 58 | `ens_id55_combo_gate_rank1` | MAE 0.407520, rank 5 | Local OOF/preflight gain did not transfer. |
| 59 | `ens_id57_high_activity_lift_rank2` | MAE 0.407730, rank 5 | Conservative calibration-style lift also did not transfer. |

The absolute differences are small, but the repeated id58/id59 pattern is the
important signal: small OOF or pseudo-public improvements are no longer reliable
enough to justify spending Phase 1 submissions.

## Current Production Ensemble Shape

The canonical script is `track1_activity/scripts/run_ensemble.py`.

As of this review, `ENSEMBLE_MODELS` contains nine active members:

1. `tabpfn_cheme_2d_full_boltz_log2fc_pred_optuna_trial10_seed5ens_umap_default`
2. `tabpfn_chemprop_pretrain_embed_umap_default`
3. `tabpfn_pooled_boltz_umap_default`
4. `tabpfn_pooled_boltz_allpairs_umap_default`
5. `tabpfn_molformer_c3_pretrain_embed_umap`
6. `tabpfn_kermt_pretrain_embed_umap_default`
7. `tabpfn_attentivefp_pretrain_embed_umap_default`
8. `tabpfn_gatedgcn_pretrain_embed_umap_default`
9. `tabpfn_cheme_2d_full_boltz_log2fc_pred_seed10ens_top500_umap`

The ensemble recipe is `caruana_bag20`: a bagged forward-selection ensemble
that spreads weight across correlated strong members more conservatively than
continuous weight optimizers.

## Main Lessons

### What Worked

- Low-fidelity `log2_fc` pretraining followed by frozen embedding extraction
  was the strongest repeatable modeling axis.
- TabPFN was very effective once strong molecular representations or selected
  tabular features were prepared for it.
- Multi-seed `log2_fc` prediction features improved the dominant 2D/Boltz
  tabular model family.
- Per-fold top-500 feature selection helped TabPFN handle large feature sets.
- Simple positive-slope affine calibration gave one large early public-LB gain.

### What Became Risky

- Adding highly correlated variants often improved OOF but hurt public LB.
- Local OOF gains below roughly 0.004 became weak evidence near the end of the
  Phase 1 sweep.
- Public-LB movement after id55/id57 suggests the current validation setup no
  longer simulates Analog Set 1 well enough for small calibration moves.

### What Did Not Pay Off Enough

- Direct MoLFormer-XL LoRA fine-tuning on pEC50.
- FMGCL-style auxiliary loss without the full pretraining setup.
- Repeated small Boltz trunk/pose-feature swaps after the main trunk features
  were already represented.
- Combined gates based on `log2_fc`, ring count, feature-family gaps, and
  top500 deltas.
- Conservative high-activity lifts before Phase 2 labels.

## Phase 2 Carry-Forward

Keep id55/id57/id58/id59 prediction deltas as diagnostic anchors. Once Analog
Set 1 labels are released, revisit:

- very small affine or monotone calibration layers;
- validation split construction that better simulates the released analog set;
- whether high R2 from id59 indicates useful broad variance structure despite
  worse MAE/RAE.

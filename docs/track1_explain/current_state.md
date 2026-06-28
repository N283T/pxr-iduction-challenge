# Track 1 Current State

Last reviewed: 2026-06-28 JST.

This summary is based on GitHub issue #208, local Phase 2 answer checks, the
`lb_submissions` table, and the latest local leaderboard snapshot. Detailed
experiment chronology remains in issue #208.

## One-Sentence Status

Track 1 Phase 2 is in final-candidate mode: the submitted candidate is local
`lb_submissions.id = 63`, a conservative AS1-label-filled update that nudges
the id62 anchor from `alpha=0.40` to `alpha=0.45` and adds only two new
high-confidence AS2 composite-gate lifts.

## Submitted Phase 2 Candidate

Submitted candidate:

```text
phase2_as1_aug_top500_id55blend_a0p45_pairrankchembl_q95_g0p15_plus_combo_new_h0p15_l0p15_labels_as1
```

Recipe:

- AS1 rows are filled with released labels.
- AS2 rows start from the id55/id60 Phase 1 anchor.
- AS2 blends `0.45` toward the AS1-augmented top500 TabPFN v3 model.
- The old ChEMBL/public-PXR pairrank q95 high gate is retained at `+0.15`.
- The new composite pairrank/ChemProp gate adds only AS2 rows not already
  lifted by the old gate, also at `+0.15`.
- No low-side shift is applied in the submitted candidate.

The two new composite-only AS2 additions are:

- `OADMET-0006488`
- `OADMET-0006142`

Submission was accepted by the Hugging Face API on 2026-06-28 JST and recorded
locally as `lb_submissions.id = 63`. Leaderboard metrics are still pending in
the local snapshot.

## Immediate Preflight Read

The selected `alpha=0.45` candidate was compared against the current id62
anchor on AS2:

| metric | value |
|---|---:|
| AS2 mean absolute shift | 0.00729 |
| AS2 p90 absolute shift | 0.01263 |
| AS2 max absolute shift | 0.14696 |
| AS2 rows with abs shift > 0.05 | 2 |
| AS2 rows with abs shift > 0.10 | 2 |
| preflight verdict vs id62 | PASS |

This is the "strict-small-move" point in the alpha ladder. Larger alpha values
improved the AS1-aug model proxy but started to move many more AS2 rows:

| alpha | AS1 model proxy MAE | AS2 mean abs shift vs id62 | AS2 p90 shift | rows > 0.05 |
|---:|---:|---:|---:|---:|
| 0.40 | 0.28033 | 0.00115 | 0.00000 | 2 |
| 0.45 | 0.26456 | 0.00729 | 0.01263 | 2 |
| 0.50 | 0.24881 | 0.01343 | 0.02527 | 5 |
| 0.55 | 0.23308 | 0.01957 | 0.03790 | 19 |
| 0.60 | 0.21735 | 0.02571 | 0.05054 | 28 |

## Current Modeling Read

The id55/id60 anchor remains the most trusted Phase 1 base, but it does not
train on released AS1 labels. The AS1-augmented top500 model therefore carries
real continuation signal; the risk is that broad movement toward it repeats the
Phase 1 pattern where small local gains failed to transfer.

The current submission compromise is:

- trust the AS1-augmented model enough to move from `alpha=0.40` to `0.45`;
- keep AS2 movement very small;
- keep the proven ChEMBL/public-PXR pairrank high gate;
- use the new Boltz-style ChemProp/pairrank composite only as a sparse extra
  high-tail confirmation signal;
- avoid low-tail gates because the safe low threshold flags zero AS2 rows and
  looser thresholds looked too broad.

## Useful New Research Artifacts

- `docs/track1_explain/chembl_pairwise_deep.md` records the ActFound/Boltz-style
  ChEMBL same-assay pairwise ChemProp line and the id63 decision.
- `docs/track1_explain/twinbooster_zero_shot.md` records a negative zero-shot
  assay-text probe.
- `track1_activity/scripts/phase2_apply_composite_gate.py` reproduces sparse
  composite-gate submission adjustments.
- `track1_activity/scripts/prepare_chembl_pairwise_deep.py`,
  `run_chemprop_pairwise_pretrain.py`, and
  `score_chemprop_pairwise_pretrain.py` reproduce the pairwise ChemProp
  pretraining and scoring pipeline.

## Next Watch Items

- Fetch the activity leaderboard again later and back-fill id61/id62/id63 local
  rows once the public metrics appear.
- If id63 is positive, revisit a slightly more aggressive `alpha=0.50` variant.
- If id63 is negative, keep id62 as the safer Phase 2 anchor and treat the
  composite gate as research-only.

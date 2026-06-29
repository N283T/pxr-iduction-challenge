# Phase 2 final submission decision

Date: 2026-06-29 JST.

This note records the final Track 1 Phase 2 submission review after re-checking
the id62/id63 lineage, the top500 v3 versus v2.6 question, seed10/seed15
variants, and one small consensus hedge. Phase 2 Activity has no live public
leaderboard feedback, so this decision deliberately does not depend on fetching
or replaying a new LB row after submission.

## Primary decision

Keep local submission `id63` as the primary final candidate:

```text
phase2_as1_aug_top500_id55blend_a0p45_pairrankchembl_q95_g0p15_plus_combo_new_h0p15_l0p15_labels_as1.csv
```

Recipe:

```text
AS1 rows = released labels
AS2 rows = id55 + 0.45 * (AS1-aug top500 - id55)
         + ChEMBL/public-PXR pairrank q95 high gate, +0.15
         + composite assay-rank/ChemProp high rows not already lifted, +0.15
```

The composite-only AS2 rows are:

- `OADMET-0006488`
- `OADMET-0006142`

The low side remains inactive. The safe low gate has no AS2 flags, and looser
low gates were treated as too broad for final use.

## Why id63 over id62

`id62` is the safer candidate:

```text
phase2_as1_aug_top500_id55blend_a0p4_pairrankchembl_q95_g0p15_labels_as1.csv
```

It uses the same AS1 label fill and the same old pairrank high gate, but stops
at `alpha=0.40` and does not add the two composite-high rows. The final review
keeps `id63` because the extra move is small relative to id62 while addressing a
real concern: the id55/id60 Phase 1 anchor never trained on released AS1 labels.

AS2 shift of id63 relative to id62:

| metric | value |
|---|---:|
| mean absolute shift | 0.00729 |
| p90 absolute shift | 0.01263 |
| rows with abs shift > 0.05 | 2 |
| rows with abs shift > 0.10 | 2 |

That is the largest alpha step that still reads as a strict-small-move update.
`alpha=0.50` is a reasonable attack candidate, but it starts to move more AS2
rows and is not justified without live Phase 2 feedback.

## Top500 variant review

The top500 target inside id62/id63 is the AS1-augmented final-fit TabPFN v3
model:

```text
phase2_as1_aug_cheme_2d_full_boltz_log2fc_pred_seed10ens_top500_tabpfnv3_ne8_t0p7_model_only.csv
```

The final review re-ran the main alternatives as id63-style candidates:

- seed10, TabPFN v2.6, `softmax_temperature=0.9`
- seed10, TabPFN v2.6, `softmax_temperature=0.7`
- seed15, TabPFN v2.6, `softmax_temperature=0.9`
- seed15, TabPFN v3, `softmax_temperature=0.7`

The v2.6 and seed15 candidates were useful uncertainty checks but did not
justify replacing id63. The main reason is evidence ordering:

- Phase 1 production mostly favored TabPFN v2.6, so the concern was legitimate.
- The best pre-AS1 AS1 replay among the top500 family was nevertheless the
  seed10 v3 `temp0.7` row.
- Switching back to v2.6 mostly backs away from the AS1-augmented signal rather
  than adding a new independent signal.
- seed15 v3 is the only plausible challenger, but its row movement versus id63
  is too large for the available evidence.

Conclusion: keep seed10 TabPFN v3 `temp0.7` as the top500 target for id63.

## Hedge candidate

The only new candidate with a real case is an id63-preserving version of the
small consensus Boltz/top500 correction:

```text
phase2_id63_plus_consensus_boltz_top500_b0p2_delta_labels_as1.csv
```

This applies the existing `b0.2` consensus delta on top of id63 rather than
using the older id62-anchored file directly. That matters because the older file
partly undoes id63's two composite-high additions.

Preflight versus id63:

| metric | value |
|---|---:|
| verdict | PASS |
| mean shift | -0.00581 |
| mean absolute shift | 0.00809 |
| p90 absolute shift | 0.03549 |
| max absolute shift | 0.08542 |
| rows with abs shift > 0.10 | 0 |

This is a sensible diversification or backup submission if multiple final files
can be retained and selected. It is not strong enough to replace id63 as the
single/latest final candidate because it mainly nudges high predictions
downward and has no blind AS2 feedback.

## Final portfolio read

If only one/latest Activity submission matters, leave id63 as final.

If multiple submissions can be retained or selected, the clean portfolio is:

1. `id63` as primary.
2. `phase2_id63_plus_consensus_boltz_top500_b0p2_delta_labels_as1.csv` as the
   controlled consensus hedge.
3. `id62` only as a conservative fallback, not as the default final.

Avoid for final replacement:

- v2.6 top500 swaps,
- seed15 top500 swaps,
- id55-shape conservative reset,
- broad low-side gates,
- `alpha >= 0.50` unless intentionally making an attack submission.

Generated audit outputs are under
`track1_activity/analysis/phase2_final_decision/` and are intentionally kept out
of git as analysis artifacts.

# ChEMBL External Judge Probe

Date: 2026-05-16

Goal: test ChEMBL PXR activation data as an external candidate judge without
using it as training data. Exact challenge train/test overlaps are excluded
before nearest-neighbor scoring.

## Setup

Script:

- `track1_activity/analysis/chembl_pxr_probe/chembl_external_judge.py`

Outputs:

- `track1_activity/analysis/chembl_pxr_probe/outputs/external_judge/external_judge_report.md`
- `track1_activity/analysis/chembl_pxr_probe/outputs/external_judge/external_judge_coverage.csv`
- `track1_activity/analysis/chembl_pxr_probe/outputs/external_judge/train_external_signal_summary.csv`
- `track1_activity/analysis/chembl_pxr_probe/outputs/external_judge/candidate_external_judge_summary.csv`
- `track1_activity/analysis/chembl_pxr_probe/outputs/external_judge/candidate_external_judge_largest_shifts.csv`

External reference:

- Start from filtered ChEMBL PXR activation EC50 rows in
  `chembl_pxr_activation_probe.py`.
- Remove any ChEMBL molecule whose InChIKey exactly matches a challenge train or
  test compound.
- Use only nearest-neighbor and top-k ChEMBL pChEMBL signals for diagnostics.

## Coverage

| item | value |
|---|---:|
| raw filtered ChEMBL PXR activation molecules | 267 |
| exact challenge overlaps removed | 12 |
| external molecules after exclusion | 255 |
| test compounds with NN Tanimoto >= 0.25 | 121 |
| test compounds with NN Tanimoto >= 0.30 | 27 |
| test compounds with NN Tanimoto >= 0.35 | 4 |
| test compounds with NN Tanimoto >= 0.40 | 0 |
| test NN max / median | 0.3939 / 0.2222 |

## Train Signal

After exact challenge-overlap exclusion, ChEMBL PXR nearest-neighbor pChEMBL is
only weakly aligned with challenge train pEC50.

| threshold | n train | Spearman pEC50 vs external | Pearson pEC50 vs external |
|---:|---:|---:|---:|
| 0.25 | 790 | 0.1559 | 0.0705 |
| 0.30 | 161 | 0.2391 | 0.1215 |
| 0.35 | 55 | 0.1400 | -0.0204 |
| 0.40 | 36 | 0.0758 | -0.2102 |

This is too weak and sparse to use as a hard submission gate.

## Candidate Judge

At test NN Tanimoto >= 0.30 there are only 27 compounds. The external judge does
not catch the id58 regression. It mildly favors the same upward moves, because
the covered ChEMBL neighbors have much higher pChEMBL than our predictions.

| candidate | direct MAE delta vs id57 | centered MAE delta vs id57 | mean shift vs id57 |
|---|---:|---:|---:|
| id55 soft g35 | -0.00083 | -0.00150 | +0.00083 |
| id58 combo rank1 | -0.00330 | +0.00010 | +0.00430 |
| log2fc gate q60 g50 | -0.00360 | +0.00034 | +0.00360 |
| num_rings positive g50 | -0.00331 | +0.00008 | +0.00473 |

## Interpretation

ChEMBL PXR activation data are useful for qualitative auditing and future
Phase 2 interpretation, but not for current Track 1 candidate selection:

- test coverage is sparse and below 0.40 Tanimoto after exact-overlap removal;
- train correlation is weak after removing challenge exact overlaps;
- id58 still looks slightly better by this judge, so it would not have prevented
  the LB-negative submission.

Next step should be split construction rather than a ChEMBL gate: build
pseudo-public validation folds using test-likeness, high pEC50/log2fc regions,
potent-neighborhood structure, and optional ChEMBL coverage as a weak stratum.

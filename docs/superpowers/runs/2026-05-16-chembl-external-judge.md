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

## Current Model vs ChEMBL

Exact external inference for the full submitted ensemble on arbitrary ChEMBL
SMILES is not available from the stored artifacts. Most pool members are stored
as train OOF plus challenge-test submissions, not reusable arbitrary-SMILES
model checkpoints. The measurable proxy is therefore:

- exact challenge-train overlaps with ChEMBL PXR activation;
- challenge-train compounds compared to their exact-overlap-excluded ChEMBL
  nearest-neighbor pChEMBL.

For the 12 exact challenge-train overlaps, challenge pEC50 itself is closer to
ChEMBL than the current ensemble OOF. The current model underpredicts ChEMBL on
average.

| set | prediction | n | MAE vs ChEMBL | bias pred - ChEMBL | Pearson | Spearman |
|---|---|---:|---:|---:|---:|---:|
| exact train overlap | challenge pEC50 label | 12 | 0.6476 | -0.1793 | 0.7780 | 0.9702 |
| exact train overlap | raw current OOF | 12 | 0.8046 | -0.4463 | 0.4086 | 0.6480 |
| exact train overlap | calibrated-best OOF | 12 | 0.8280 | -0.4425 | 0.4080 | 0.6480 |

For non-exact ChEMBL-nearest train regions, the signal is much weaker and the
model is roughly one log unit below ChEMBL pChEMBL on average:

| train subset | prediction | n | MAE vs external NN | bias pred - ChEMBL NN | Spearman |
|---|---|---:|---:|---:|---:|
| NN >= 0.25 | raw current OOF | 790 | 1.1496 | -1.0322 | 0.1979 |
| NN >= 0.30 | raw current OOF | 161 | 1.2897 | -1.2105 | 0.2638 |
| NN >= 0.35 | raw current OOF | 55 | 1.5155 | -1.4140 | 0.1091 |

This reinforces the main conclusion: ChEMBL is not currently a calibrated
target-compatible judge for our challenge predictions. It mostly says "these
nearby public PXR compounds are active in ChEMBL assays", but that does not
translate cleanly to Octant challenge pEC50 or to our model scale.

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

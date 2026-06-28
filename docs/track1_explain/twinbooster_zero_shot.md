# TwinBooster Zero-Shot Assay-Text Probe

This experiment tested TwinBooster as an assay-conditioned external prior for
PXR induction. TwinBooster predicts active probability for a `(SMILES, assay
description)` pair, so it was evaluated as a ranking/gating signal rather than a
pEC50 regressor.

## Setup

- External clone: `/home/nagaet/twinbooster-pxr`
- Local patch: removed an eager `LSA` import from the clone `__init__.py` because
  it triggered a circular import unrelated to prediction.
- Pretrained models were downloaded from the official TwinBooster links. The
  Barlow Twins archive needed an `aria2c` retry after a partial download.
- Main script:
  `track1_activity/analysis/phase2_classifier_gate/run_twinbooster_zero_shot_probe.py`
- Outputs:
  `track1_activity/analysis/phase2_classifier_gate/outputs/twinbooster_zero_shot/`

## Results

The generic PXR prompts were weak on AS1. The best AS1 high-tail average
precision was only about `0.059`, from the exact PubChem AID 720659-style
description. AS1 Spearman was near zero or negative for most prompts.

The exact PubChem prompt produced a small sparse id55 gate improvement in an
answer-check scan:

| prompt | mode | q | gamma | flags | AS1 MAE delta vs id55 |
|---|---:|---:|---:|---:|---:|
| pubchem_aid720659_exact | high_lift | 0.95 | 0.20 | 13 | -0.00146 |

This is much weaker than the ChEMBL/public-PXR assay-rank gate and is not enough
evidence for a submission direction. The useful takeaway is mostly negative:
assay-text zero-shot activity probabilities do not recover the PXR pEC50
landscape here, even when prompted with a known PubChem PXR activation assay.

## Interpretation

TwinBooster can remain a diagnostic prior for exact-assay active-looking
compounds, but it should not be concatenated into the top500 feature set without
a stronger readout. If revisited, the next step should be a separate stack/gate
or a public-PXR PubChem/ChEMBL assay-label calibration, not raw feature fusion.

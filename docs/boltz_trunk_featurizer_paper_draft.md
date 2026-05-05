# Boltz Trunk as a Target-Conditioned Molecular Featurizer

Draft research note, 2026-05-05.

## Working Title

Using Structure Prediction Trunks Without Structures: Target-Conditioned Molecular Featurization from Boltz-2 Pair Representations

## One-Sentence Pitch

Protein-ligand structure prediction models can be reused as target-conditioned molecular featurizers by pooling their internal ligand-protein trunk representations, even when the predicted pose, confidence head, and affinity head are deliberately ignored.

## Core Idea

Boltz-2 is normally used to generate a protein-ligand structure and, optionally, affinity-related outputs. In the PXR activity task, we instead treat the Boltz-2 trunk as a frozen interaction encoder:

1. Run the receptor-ligand system through the Boltz-2 trunk.
2. Extract raw `s` token embeddings and `z` pair embeddings from the output NPZ.
3. Ignore the final structure, pose confidence, PoseBusters checks, and affinity head outputs.
4. Pool protein-ligand `z` representations by receptor region and ligand atoms.
5. Use the pooled representation as a side feature for downstream QSAR/activity prediction.

This reframes a structure predictor as a protein-conditioned representation model.

## Why This Is Interesting

The standard assumption is that a structure prediction model helps activity prediction only if the predicted binding pose or affinity estimate is accurate enough. This experiment suggests a weaker and more useful claim: the trunk representation may contain activity-relevant protein-ligand interaction information before any final pose is trusted.

This is especially attractive for small QSAR settings:

- It avoids relying on one noisy predicted pose.
- It avoids using a potentially miscalibrated affinity head.
- It supports cheap low-recycle inference, because the representation can still be extracted before high-quality structure refinement.
- It creates target-conditioned ligand features without training a new protein-ligand model.

## PXR Pilot Result

Dataset/context:

- Target: Pregnane X receptor, fixed receptor.
- Downstream task: pEC50 regression in the OpenADMET PXR Blind Challenge Track 1.
- Available trunk rows: 13,134 compounds in `compound_boltz2_trunk_fast`.
- Recycling split:
  - 4,652 rows from full Boltz rcycle=3 runs.
  - 8,482 rows from cheap trunk-only rcycle=1 runs.
- Missing rows: 01657 and 08624.

Feature construction:

- Source: raw Boltz NPZ `s` and `z` tensors.
- Excluded: pose geometry, confidence, affinity, PoseBusters, and contact features.
- `s` pooling:
  - ligand token mean/std
  - core-pocket residue mean
- `z` pooling:
  - protein-region x ligand-atom cross-pairs
  - per-region mean/std/q10/q90
  - regions: `nterm`, `lbd_entrance`, `lbd_body`, `h11_h12`, `core_pocket`
- Output: 3,713 feature columns.
- Artifact: `data/boltz_affhead/repooled_trunk_region_zstats.parquet`.
- Training feature name: `repooled_trunk_region_zstats`.

OOF result:

| Model | Feature | UMAP OOF MAE | Note |
| --- | --- | ---: | --- |
| TabPFN | existing core pooled trunk | 0.4860 | existing Boltz trunk baseline |
| TabPFN | existing allpairs pooled trunk | 0.4859 | existing Boltz trunk baseline |
| TabPFN | re-pooled region zstats trunk | 0.4744 | new trunk featurizer |

Ensemble behavior:

- Baseline 9-pool Caruana OOF MAE: 0.3971.
- Replacing the weaker core-pooled Boltz member with the re-pooled trunk member: OOF MAE 0.3945.
- Importance-calibrated version: OOF MAE 0.3937.
- Correlation with existing trunk members is high:
  - vs allpairs: r = 0.986
  - vs core pooled: r = 0.982

Interpretation:

- The re-pooling improves the Boltz-trunk representation.
- It is not a new independent ensemble axis in this PXR setup.
- A residual-head diagnostic found no meaningful ability to explain the remaining ensemble residual:
  - ridge residual correction: MAE delta -0.000025
  - LGBM residual correction: MAE delta -0.000112, with Spearman loss

## Hypothesis

The Boltz trunk learns a useful ligand-target compatibility representation in its pair embedding `z`. For downstream activity prediction, region-wise summaries of protein-ligand pair embeddings can outperform naive whole-protein pooling, even if final structure coordinates are not used.

More specifically:

1. `z` contains richer target-conditioned signal than ligand-only embeddings.
2. Region-aware pooling preserves receptor-local interaction information lost by global allpairs mean/max pooling.
3. Low-recycle trunk-only inference may be sufficient for QSAR features, because precise final pose convergence is not required.

## What Needs To Be Shown For A Paper

The PXR result is a strong pilot but not enough by itself. A credible preprint needs multi-target evidence.

Minimum benchmark:

- Several fixed-target QSAR tasks from ChEMBL or public activity datasets.
- For each target:
  - train/test split with scaffold or temporal separation
  - ligand-only baselines: Morgan, RDKit descriptors, ChemBERTa/MoLFormer, TabPFN or LightGBM
  - structure-model baselines: Boltz affinity scalar, pose-derived descriptors if available
  - trunk-featurizer variants: global pooling, allpairs pooling, region/pocket pooling
- Metrics:
  - regression: MAE/RMSE/Spearman
  - classification if available: AUROC/AUPRC/EF
  - compute cost per ligand

Key ablations:

- trunk-only vs final pose features
- `s` only vs `z` only vs `s+z`
- allpairs global pooling vs region-aware pooling
- rcycle=1 vs rcycle=3
- target-conditioned trunk features vs ligand-only foundation embeddings
- frozen trunk features vs affinity head outputs

Potential target panel:

- PXR as the motivating case.
- Nuclear receptors with flexible ligand-binding domains.
- Kinases or GPCRs where many activity labels and receptor structures exist.
- A matched set where target identity is fixed within each task, to test fixed-target QSAR utility before attempting cross-target generalization.

## Possible Paper Claims

Conservative:

- Internal pair representations from structure prediction models provide useful target-conditioned side features for small QSAR models.
- Region-wise pooling of trunk pair embeddings improves over naive global pooling.
- Useful signal can be extracted without using predicted structures or affinity heads.

Stronger, if multi-target results support it:

- Low-recycle structure-model inference is sufficient for activity-relevant featurization.
- Structure prediction trunks can serve as general-purpose protein-conditioned molecular encoders.
- Final coordinate accuracy is not necessary for all downstream ligand-ranking tasks.

Avoid overclaiming:

- Do not claim pose prediction is irrelevant.
- Do not claim Boltz trunk features beat all ligand-only methods by themselves.
- Do not claim affinity prediction is solved.
- Do not claim generality from PXR alone.

## Suggested Figures

1. Concept diagram:
   - conventional Boltz use: input -> trunk -> structure/affinity -> score
   - proposed use: input -> trunk -> `s/z` pooling -> QSAR model
2. Pooling schematic:
   - receptor regions x ligand atoms from `z`
   - mean/std/quantile pooling
3. PXR pilot bar chart:
   - existing core pooled, existing allpairs, new region zstats
4. Multi-target benchmark table:
   - ligand-only baselines vs trunk-featurizer variants
5. Compute-quality tradeoff:
   - rcycle=1 vs rcycle=3 vs downstream performance

## Risks And Confounders

- Boltz training data may contain related activity or structural data for some targets. Any paper needs careful leakage discussion.
- Better trunk features may still be highly correlated with existing ligand-only features in many tasks.
- PXR has a large flexible pocket; effects may be weaker for rigid targets or narrow SAR series.
- Region definitions are target-specific. A generic method needs either sequence/structure-derived automatic regions or pocket-token selection.
- If the final benchmark relies on public ChEMBL, chemical series leakage and split design will dominate the interpretation.

## Near-Term Next Steps

1. Freeze the PXR pilot as a motivating case.
2. Build a small benchmark harness for 5-10 ChEMBL targets.
3. Standardize three trunk poolers:
   - global allpairs
   - pocket/core residue pooling
   - region-wise zstats
4. Run rcycle=1 features first to test the cheap-inference claim.
5. Add rcycle=3 on a smaller subset only if rcycle=1 shows signal.
6. Write the first preprint around the representation hypothesis, not leaderboard performance.

## Useful Internal Artifacts

- `docs/papers/boltz2_affinity_notes.md`
- `docs/track1_boltz2_features.md`
- `docs/superpowers/specs/2026-05-05-boltz-allpairs-compact-variants-design.md`
- `docs/superpowers/plans/2026-05-05-boltz-trunk-fast-inventory.md`
- `track1_activity/scripts/boltz_affhead/37_trunk_fast_inventory.py`
- `track1_activity/scripts/boltz_affhead/38_repool_trunk_npz.py`
- `track1_activity/scripts/boltz_affhead/39_repool_bakeoff.py`
- `track1_activity/scripts/boltz_affhead/40_trunk_residual_head.py`
- issue #100 entries from 2026-05-05

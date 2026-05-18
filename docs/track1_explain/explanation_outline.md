# Explanation Outline

Use this as a speaking outline for explaining Track 1 to someone who knows
machine learning but has not followed the full experiment log.

## 1. Start With The Task

- Goal: predict pEC50 for 513 blinded compounds.
- Primary metric: MAE.
- Constraint: public leaderboard feedback is sparse and can be noisy relative
  to local OOF improvements.

Key framing:

> This became less about finding one magic model and more about building several
> PXR-relevant molecular views, then learning which local improvements actually
> transferred to the hidden analog set.

## 2. Explain The Data Signal

The decisive extra signal was the auxiliary single-concentration activity
(`log2_fc`). It is not the final target, but it gives a broad low-fidelity
activity axis.

The winning recipe repeatedly reused this axis:

1. pretrain an encoder on `log2_fc`;
2. freeze the encoder;
3. extract compound embeddings;
4. train pEC50 predictors on top.

## 3. Explain The Model Families

Introduce the final ensemble as a mix of four kinds of evidence:

| Evidence type | Examples | Role |
|---|---|---|
| Classical tabular chemistry | RDKit/Mordred/Jazzy-style descriptors and fingerprints | Stable baseline chemical representation. |
| Low-fidelity activity features | Predicted `log2_fc`, ChemProp embeddings, GNN/transformer embeddings | Main PXR-specific signal. |
| Structural/protein-ligand features | Boltz trunk and pooled interaction representations | Adds a different physical view. |
| Feature-compressed TabPFN models | top500 variants | Lets TabPFN focus on the most useful dimensions. |

## 4. Explain Why TabPFN Appears Everywhere

TabPFN was not used because it knows chemistry by itself. It worked because the
pipeline gave it strong molecular representations first. In this project,
TabPFN was the downstream learner that converted prepared descriptors,
fingerprints, and embeddings into pEC50 predictions.

## 5. Explain The Ensemble

The ensemble is not a simple average of every experiment. It is an explicit
allow-list in `track1_activity/scripts/run_ensemble.py`, then a bagged Caruana
selection.

The key idea:

- Keep strong models.
- Keep some weaker models only if they are genuinely different.
- Drop models that are merely correlated copies, even when they look good in
  local OOF.

## 6. Explain The OOF/LB Mismatch

This is one of the most important lessons.

OOF gains transferred early, but near the end many small local improvements did
not transfer to public LB. Some even moved in the wrong direction. This led to a
more conservative policy:

- Treat tiny OOF gains as weak evidence.
- Check prediction movement against known LB-positive and LB-negative anchors.
- Stop spending submissions when recent local-gain variants are flat or
  negative on public LB.

## 7. End With The Current Decision

The current stance is to pause Phase 1 Track 1 submissions.

Reason:

- The core ensemble is strong.
- The latest small calibration and gated variants did not improve public MAE.
- Phase 2 labels will be much more valuable for diagnosing whether the remaining
  issue is calibration, analog-set mismatch, or true model error.

## 8. Natural Follow-Up For Phase 2

Once labels are available:

- compare id55/id57/id58/id59 deltas compound-by-compound;
- fit small calibration layers using released analog labels;
- redesign validation splits to mimic the analog-set selection better;
- preserve broad ranking/variance structure unless labels prove it is overfit.

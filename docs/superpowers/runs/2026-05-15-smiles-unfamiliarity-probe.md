# SMILES Unfamiliarity Probe

Date: 2026-05-15

Context: quick check of van Tilborg et al. 2026, "Molecular deep learning at
the edge of chemical space", for Track 1 PXR. The paper is primarily a
classification/OOD reliability method, so this was treated as a diagnostic
probe rather than a direct leaderboard candidate.

Implementation:

- Added `track1_activity/scripts/run_smiles_unfamiliarity_probe.py`.
- Fold-safe canonical UMAP CV.
- Lightweight joint model: SMILES CNN encoder, conditioned GRU decoder, and
  pEC50 regression head.
- Per-molecule unfamiliarity is `log(reconstruction_loss)`.
- Upstream reference was inspected at `molML/JointMolecularModel`.

Command:

```bash
pixi run python track1_activity/scripts/run_smiles_unfamiliarity_probe.py \
  --run-name jmm_lite_e30_g0p1 \
  --max-epochs 30 \
  --patience 5 \
  --batch-size 256 \
  --hidden-dim 128 \
  --z-dim 64 \
  --emb-dim 64 \
  --gamma 0.1
```

Result:

```text
JMM pEC50 head OOF MAE: 0.590046
JMM pEC50 head OOF RAE: 0.648452
JMM pEC50 head OOF Spearman: 0.632424

OOF unfamiliarity mean/std:  -0.099229 / 0.214689
Test unfamiliarity mean/std: -0.180048 / 0.191688
Test minus OOF mean:         -0.080819

Spearman(unfamiliarity, y):                          0.011644
Spearman(unfamiliarity, JMM abs error):              0.073386
Spearman(unfamiliarity, current ensemble abs error): 0.064572
Spearman(unfamiliarity, current ensemble residual): -0.020003
```

Additional read:

- Test compounds are not higher-unfamiliarity than train OOF under this model.
- Only about 1.2% of test compounds exceed the OOF q95 unfamiliarity threshold;
  none exceed OOF q99.
- The signal is essentially uncorrelated with the current ensemble's OOF
  absolute error.

Decision: do not use this as a submission axis or gate in its current form. It
is useful as a negative diagnostic: the paper's classification/OOD reliability
setup does not transfer cleanly to the current PXR regression leaderboard
problem without a stronger hypothesis.

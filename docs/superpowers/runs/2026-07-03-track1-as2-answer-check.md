# Track 1 AS2 Answer-Check Notes

Post-competition answer-check log for OpenADMET PXR Track 1 after the official
AS2 labels were released.

Detailed running notes live in GitHub issue #222. This file keeps the durable
summary in the repository and intentionally does not add generated replay CSVs.

## Scope

- Replay submitted and unsubmitted Track 1 candidates on official AS2 labels.
- Check whether the best behavior was manual CSV algebra or mechanically
  reproducible.
- Revisit whether the production ensemble added value over single models.
- Inspect member leave-one-out effects and hindsight reweighting.

## Main Results

Key AS2 MAE values from the replay:

```text
anchor_residual rerun        0.404879
hindsight id55shape          0.405580
id55 / id60 anchor           0.407489
importance calibrated        0.407242
id62 phase2 final update     0.411463
id61 phase2 base             0.412133
id63 actual final            0.412345
raw caruana_bag20            0.418851
best pre single full model   0.426669
pre top500 single            0.439571
post-AS1 top500 single       0.431475
```

The strongest checked mechanical candidate was a rerun of
`ens_caruana_bag20_anchor_residual`. It slightly beat the hindsight
`id55shape` CSV candidate and did not depend on AS2 labels, AS1 labels, or
manual algebra between submitted CSVs.

## Anchor Residual

The anchor residual recipe was:

```text
raw caruana_bag20
-> global importance affine calibration
-> linear residual model using potent-46 anchor proximity features
```

The residual model used four features:

```text
nn_tanimoto        nearest-neighbor Tanimoto to potent-46 train anchors
anchor_pec50       measured pEC50 of that nearest potent train anchor
base_pred          calibrated ensemble prediction
pred_minus_anchor  base_pred - anchor_pec50
```

The rerun learned a mostly proximity-driven correction:

```text
nn_tanimoto        +0.3413
anchor_pec50       -0.0186
base_pred          -0.0302
pred_minus_anchor  -0.0116
intercept          +0.1172
```

Interpretation: the calibrated ensemble tended to underpredict compounds close
to potent, selective train anchors. A small learned correction fixed part of
that bias without changing the prediction shape aggressively.

## Ensemble Value

The ensemble and calibration both mattered:

```text
best pre single full model   0.426669
raw caruana_bag20            0.418851
importance calibrated        0.407242
anchor_residual rerun        0.404879
```

Stepwise deltas:

```text
single -> raw ensemble       -0.007819
raw ensemble -> calibration  -0.011609
calibration -> anchor resid  -0.002362
```

The original `caruana_bag20` weights were:

```text
top500 CheMeleon/TabPFN      0.309223
full CheMeleon/TabPFN        0.287864
chemprop embed               0.151456
KermT                        0.110680
pooled Boltz                 0.045631
MoLFormer                    0.040291
Boltz allpairs               0.034951
GatedGCN                     0.017476
AttentiveFP                  0.002427
```

After anchor residual, AS2 leave-one-out suggested that weak single models were
not simply useless. Several diversity members worsened AS2 when removed. The
main hindsight issue was that the top500 member was overweighted in this
mixture.

## Reweighting Checks

Top500-excluded and top500-included variants were checked around the anchor
residual pipeline.

```text
AS2 oracle full                  0.399156
AS2 oracle no top500             0.400542
old weights, top500 removed      0.402326
old weights full                 0.404879
OOF opt no top500                0.408050
OOF opt full                     0.421790
```

The OOF reoptimizers overfit badly. The best simple non-oracle change was to set
the top500 member weight to zero and renormalize the old Caruana weights.

AS2 oracle full weights were:

```text
top500 CheMeleon/TabPFN      0.3318
pooled Boltz                 0.1950
MoLFormer                    0.1681
full CheMeleon/TabPFN        0.1361
KermT                        0.1180
chemprop embed               0.0510
AttentiveFP                  0.0000
GatedGCN                     0.0000
Boltz allpairs               0.0000
```

This means top500 was not conceptually dead. In the AS2 oracle it still received
the largest weight. The likely failure mode was combining top500 with too much
same-axis full/chemprop signal and too little orthogonal Boltz/MoLFormer
diversity.

## Lessons

- Phase 2 AS1 augmentation did not help AS2 reliably.
- The final high-gate updates were plausible but too blunt for dense 4-6
  compounds.
- The strongest clean mechanical recipe was already present in Phase 1:
  ensemble, importance calibration, and potent-anchor residual.
- OOF was useful for model building, but late-stage OOF-only weight optimization
  was dangerous.
- Top500 carried real signal, but its OOF-derived role in the production mixture
  was not well calibrated for AS2.

If this were still live, the primary mechanical fallback would be
`old_no_top500_renorm_anchor_residual`; the higher-risk follow-up would be
constrained diversity-aware reweighting with caps per correlated model family.

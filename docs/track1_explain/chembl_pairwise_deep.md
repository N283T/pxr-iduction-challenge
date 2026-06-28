# ChEMBL Pairwise Deep Pretraining

This note records ChemProp implementations of the ActFound/Boltz-style
same-assay pairwise idea on local ChEMBL36. "Boltz-style" here means the
affinity-training objective design, not using Boltz structures or pose features:
same-assay deltas carry the main signal, and a small absolute-value auxiliary
term keeps the scalar head weakly calibrated.

## Pipeline

Source scripts:

- `track1_activity/scripts/prepare_chembl_pairwise_deep.py`
- `track1_activity/scripts/run_chemprop_pairwise_pretrain.py`
- `track1_activity/scripts/score_chemprop_pairwise_pretrain.py`

External ChEMBL SMILES are standardized with the ChEMBL structure pipeline
(`standardize_mol` + `get_parent_mol`) before exact challenge InChIKey exclusion.
Generated parquet, CSV, and checkpoint artifacts stay under ignored
`data/chembl/` and `track1_activity/checkpoints/` paths.

The first model is a Siamese ChemProp MPNN trained on same-assay deltas:

```text
pred_delta = f(smiles_a) - f(smiles_b)
loss = weighted SmoothL1(pred_delta, pChEMBL_a - pChEMBL_b)
```

This cancels assay-level offsets and keeps the objective close to ActFound and
the Boltz affinity pairwise training recipe.

The Boltz-affinity variant adds two switches:

```text
loss = diff_weight * SmoothL1(f(a) - f(b), pChEMBL_a - pChEMBL_b)
     + abs_weight  * 0.5 * (SmoothL1(f(a), pChEMBL_a) + SmoothL1(f(b), pChEMBL_b))
```

The default Boltz-inspired setting tested here is `diff_weight=0.9`,
`abs_weight=0.1`. A second switch samples assays proportional to their activity
IQR before drawing pairs. This was intended to mimic Boltz-2's dynamic-range
sampling, but it did not transfer cleanly in the first PXR tests.

## Pilot Results

AS1 metrics below use the scalar `f(smiles)` score directly.

| data | train | AS1 Spearman | AS1 gte6 AUC | AS1 gte6 AP | AS1 lt3 AP |
|---|---:|---:|---:|---:|---:|
| broad random 250k, B/F, IC50/Ki/Kd/EC50/AC50 | 100k pairs, 5 epochs | 0.476 | 0.799 | 0.225 | 0.118 |
| binding random 250k, B, IC50/Ki/Kd, seed 42 | 100k pairs, 5 epochs | 0.516 | 0.826 | 0.359 | 0.169 |
| binding random 250k, B, IC50/Ki/Kd, seed 7 | 100k pairs, 5 epochs | 0.481 | 0.799 | 0.127 | 0.197 |
| binding random 250k, B, IC50/Ki/Kd, seed 123 | 100k pairs, 5 epochs | 0.504 | 0.760 | 0.252 | 0.220 |
| binding 3-seed z-score mean | 100k pairs, 5 epochs | 0.531 | 0.812 | 0.247 | 0.201 |
| binding random 250k, B, IC50/Ki/Kd | 200k pairs, 5 epochs | 0.509 | 0.800 | 0.195 | 0.198 |
| functional random 150k, F, EC50/AC50 | 100k pairs, 5 epochs | 0.298 | 0.567 | 0.064 | 0.120 |
| binding, IQR sampler, 0.9 diff + 0.1 abs, seed 42 | 100k pairs, 5 epochs | 0.421 | n/a | 0.070 | 0.186 |
| binding, IQR sampler, diff-only, seed 42 | 100k pairs, 5 epochs | 0.479 | 0.703 | 0.107 | 0.242 |
| binding, uniform sampler, 0.95 diff + 0.05 abs, seed 42 | 100k pairs, 5 epochs | 0.516 | 0.827 | 0.337 | 0.182 |
| binding, uniform sampler, 0.9 diff + 0.1 abs, seed 42 | 100k pairs, 5 epochs | 0.476 | 0.816 | 0.424 | 0.150 |
| binding, uniform sampler, 0.8 diff + 0.2 abs, seed 42 | 100k pairs, 5 epochs | 0.459 | 0.828 | 0.313 | 0.159 |
| binding, uniform sampler, 0.9 diff + 0.1 abs, seed 42 | 200k pairs, 5 epochs | 0.455 | 0.832 | 0.219 | 0.188 |
| binding, uniform sampler, 0.9 diff + 0.1 abs, seed 7 | 100k pairs, 5 epochs | 0.482 | n/a | 0.111 | 0.198 |
| binding, uniform sampler, 0.9 diff + 0.1 abs, seed 123 | 100k pairs, 5 epochs | 0.466 | n/a | 0.339 | 0.178 |
| binding, uniform + abs 3-seed z-score mean | 100k pairs, 5 epochs | 0.501 | 0.815 | 0.383 | 0.173 |

The binding-only ChEMBL subset transfers much better to PXR AS1 than broad or
functional-only rows, matching the earlier observation that ActFound's BindingDB
checkpoint was stronger than its ChEMBL checkpoint for the PXR high tail.

## Read

The deep pairwise model is promising as a PXR ranking representation, especially
from binding-like assay rows. More ChEMBL-pair training does not monotonically
improve the PXR high tail: 200k pairs and better ChEMBL validation performance
reduced AS1 high-tail AP relative to the 100k-pair seed-42 run.

Simple id55 gating remains mixed. The strongest local AS1 move in the original
diff-only scan came from a very sparse low-tail drop on the binding seed-42 score
(bottom 3%, -0.50 shift: AS1 MAE 0.3976 vs id55 0.4066). High-tail lifts were
smaller, with the best tested high lift around -0.0018 AS1 MAE.

The Boltz-affinity-style auxiliary changed the picture. Uniform sampling with
`0.9 diff + 0.1 abs` made the seed-42 scalar much stronger for the high tail:
AS1 `gte6` AP increased from 0.359 to 0.424. Standardized id55 gate scans were
written under
`track1_activity/analysis/phase2_classifier_gate/outputs/scalar_id55_gate/`.

| scalar | best id55 gate | AS1 MAE | delta vs id55 | read |
|---|---|---:|---:|---|
| uniform+abs0.05 seed42 | low-drop q=0.95, -0.20 | 0.40528 | -0.00129 | best overall rank, weak gate |
| uniform+abs seed42 | low-drop q=0.95, -0.30 | 0.40325 | -0.00332 | best practical sparse gate so far from this family |
| uniform+abs seed42 | high-lift q=0.90, +0.10 | 0.40444 | -0.00213 | useful high-tail signal, but not pure |
| uniform+abs0.2 seed42 | high-lift q=0.97, +0.20 | 0.40337 | -0.00319 | sharper top-8 high flag despite lower AP |
| uniform+abs 200k seed42 | high-lift q=0.85, +0.05 | 0.40614 | -0.00043 | more pairs diluted the actionable tail |
| uniform+abs 3-seed mean | low-drop q=0.97, -0.30 | 0.40396 | -0.00260 | more stable rank, weaker high gate |
| IQR diff-only seed42 | low-drop q=0.85, -0.05 | 0.40647 | -0.00010 | low-tail AP is higher, but it does not move id55 safely |

The current read is:

- The transferable Boltz idea is the objective weighting: pairwise differences
  remain primary, while a small absolute auxiliary improves scalar calibration
  enough to help sparse gates.
- The auxiliary weight controls the score personality. `abs_weight=0.05` gives
  the best AS1 Spearman, `0.1` gives the best high-tail AP, and `0.2` gives a
  sharper very-sparse high-lift gate.
- Boltz's IQR/dynamic-range sampler is not automatically helpful for this PXR
  use case. It improved low-tail AP in isolation but hurt high-tail transfer and
  did not produce a useful id55 adjustment.
- More pairs are not automatically better: the 200k-pair uniform+abs run kept
  high-tail AUC high but collapsed top-rank AP and gate utility.
- Seed behavior is large. Seed42 is the best high-tail scalar; the 3-seed mean
  improves overall rank but dilutes the seed42 high-tail gate.

Next useful steps:

- Extract frozen pairwise ChemProp embeddings and test TabPFN/LGBM on PXR train.
- Add a light PXR support adapter instead of using the ChEMBL scalar directly.
- Re-run binding-only uniform+abs with longer training or more pairs to test
  whether the high-tail seed42 result is stable.
- Build a small gate lab that combines uniform+abs seed42 for high-tail flags
  with 3-seed/IQR signals for low-tail sanity checks.
- Re-run binding-only with assay/target family stratification and ensemble seeds.
- Compare against ActFound BDB top flags compound-by-compound.

## Frozen Embedding Readouts

A first follow-up extracted 256d embeddings from the best binding seed-42
pairwise ChemProp checkpoint and trained PXR-side readouts on the train+AS1
folds.

| readout | AS1 MAE | AS1 Spearman | AS1 gte6 AP | AS1 lt3 AP |
|---|---:|---:|---:|---:|
| LGBM regression on embedding | 0.606 | 0.667 | 0.329 | 0.263 |
| TabPFN regression on embedding | 0.587 | 0.711 | 0.279 | 0.245 |
| TabPFN 5-class probability for `gte6` | n/a | 0.666 | 0.305 | 0.185 |
| LGBM 5-class probability for `gte6` | n/a | 0.311 | 0.106 | 0.123 |

The embedding readout is useful as a rank/tail diagnostic, but it is not a
standalone pEC50 regressor: MAE remains much worse than the id55 anchor. TabPFN
is the better embedding readout overall. The 5-class classifier is conceptually
well matched to the desired high/normal/low gate, but with current features and
labels the class probabilities did not beat the regression readout for AS1
high-tail ranking.

An id55 gate scan on these readouts found only small AS1 improvements. The best
row was a very sparse LGBM-regression high lift (top 3%, +0.30), moving AS1 MAE
from 0.4066 to 0.4047.

## Existing Top500 Fusion

The next diagnostic appended the 256d pairwise ChemProp embedding to the
existing `cheme_2d_full_boltz_log2fc_pred_seed10ens` feature family, then reused
the production-style per-fold LGBM-gain top-K selection before TabPFN.

| run | all MAE | AS1 MAE | lt3 MAE | gte6 MAE | read |
|---|---:|---:|---:|---:|---|
| existing top500 baseline | 0.3961 | 0.4242 | 0.6229 | 0.9142 | reference |
| global top500 with pairchem candidates | 0.3967 | 0.4313 | 0.6250 | 0.9134 | pairchem selected, but AS1 worse |
| global top600 with pairchem candidates | 0.3982 | 0.4314 | 0.6351 | 0.9445 | more dimensions diluted TabPFN |
| base top500 + pairchem top25 | 0.3979 | 0.4293 | 0.6399 | 0.9139 | small add also worse |

Pairwise ChemProp dimensions were strongly selected by the LGBM ranker
(`~78-106` selected dimensions in global top-K, `~4-5%` total gain share), but
they did not improve the existing top500 TabPFN member. The likely read is that
the representation carries real activity information, but it is too correlated
or too miscalibrated to be mixed into the high-weight top500 member by simple
feature concatenation. It is better treated as a separate stacking/gate signal
or adapted through a residual/support head.

## Composite Pairrank + ChemProp Gate

A follow-up combined the public-PXR pairrank scalars with Boltz-style ChemProp
scalars after z-scoring each signal on the labeled train pool:

- `combo_high_chembl_cp01 = z(pairrank_chembl) + z(cp_abs01)`
- `combo_high_chembl_cp02 = z(pairrank_chembl) + z(cp_abs02)`
- `combo_high_htchem_cp02 = z(pairrank_htchem) + z(cp_abs02)`

Generated files:

- `track1_activity/analysis/phase2_classifier_gate/build_pairrank_chemprop_composite_signal.py`
- `track1_activity/analysis/phase2_classifier_gate/outputs/composite_pairrank_chemprop/pool_composite_scores.csv`
- `track1_activity/analysis/phase2_classifier_gate/outputs/composite_pairrank_chemprop/test_composite_scores.csv`
- `track1_activity/analysis/phase2_classifier_gate/outputs/composite_pairrank_chemprop/as1_gate_scan.csv`
- `track1_activity/analysis/phase2_classifier_gate/outputs/composite_pairrank_chemprop/test_top_flags.csv`

AS1 scalar metrics:

| score | Spearman | gte6 AUC | gte6 AP | lt3 AP |
|---|---:|---:|---:|---:|
| combo_high_htchem_cp02 | 0.636 | 0.846 | 0.593 | 0.236 |
| combo_high_chembl_cp01 | 0.515 | 0.895 | 0.504 | 0.130 |
| combo_high_chembl_cp02 | 0.512 | 0.900 | 0.501 | 0.130 |
| cp_abs01 | 0.476 | 0.816 | 0.424 | 0.150 |
| pairrank_chembl | 0.348 | 0.817 | 0.314 | 0.143 |
| pairrank_htchem | 0.501 | 0.791 | 0.309 | 0.274 |

Best AS1 gate rows:

| score | gate | AS1 MAE | delta vs id55 | flags | true high | true low |
|---|---|---:|---:|---:|---:|---:|
| combo_high_htchem_cp02 | high-lift q=0.97, +0.30 | 0.40064 | -0.00592 | 8 | 5 | 0 |
| pairrank_chembl | high-lift q=0.95, +0.30 | 0.40142 | -0.00515 | 13 | 5 | 0 |
| cp_abs01 | low-drop q=0.95, -0.30 | 0.40325 | -0.00332 | 13 | 0 | 2 |

This is the cleanest result from the Boltz-style line so far. The important
part is not that the ChemProp scalar beats all existing gates alone; it does
not. The useful behavior appears when it is crossed with an independent assay
rank signal. `combo_high_htchem_cp02` keeps the true high count from the
pairrank family while removing low-tail contamination in the top AS1 flags.

## Phase 2 OOF Check

The AS1-only gate scan is intentionally optimistic, so the next check reused
the frozen Phase 2 `train + AS1` folds and the existing Phase 2 top500 OOF base.
For each held-out fold, scalar z-scoring and quantile thresholds were fit only
on the other folds.

Generated files:

- `track1_activity/analysis/phase2_classifier_gate/evaluate_composite_gate_oof.py`
- `track1_activity/analysis/phase2_classifier_gate/run_composite_residual_probe.py`
- `track1_activity/analysis/phase2_classifier_gate/outputs/composite_pairrank_chemprop_oof/`
- `track1_activity/analysis/phase2_classifier_gate/outputs/composite_pairrank_chemprop_residual/`

Fold-wise score ranking still supports the composite as a tail signal:

| score | Phase2 OOF Spearman | gte6 AUC | gte6 AP | lt3 AP |
|---|---:|---:|---:|---:|
| combo_high_htchem_cp02 | 0.510 | 0.745 | 0.068 | 0.612 |
| pairrank_htchem | 0.498 | 0.672 | 0.063 | 0.477 |
| combo_high_chembl_cp01 | 0.477 | 0.719 | 0.054 | 0.544 |

But the fixed AS1 high gates do **not** transfer as broad OOF moves:

| fixed gate | all OOF delta MAE | source AS1 delta | true gte6 delta | read |
|---|---:|---:|---:|---|
| combo_high_htchem_cp02 q97 +0.30 | +0.00173 | -0.00987 | -0.04675 | helps AS1/gte6 but hurts train-source too much |
| pairrank_chembl q95 +0.30 | +0.00354 | -0.00906 | -0.04675 | AS1 replay overstates generality |
| cp_abs01 q95 -0.30 | +0.00219 | -0.00259 | +0.00000 | low-drop too broad in Phase2 OOF |
| pairrank_htchem q85 -0.10 | -0.00109 | +0.00196 | +0.00909 | broad low-tail correction, not high-tail |

The better use is a small residual adapter on scalar signals rather than a
manual fixed gate. A fold-wise LightGBM residual model, capped to small shifts,
kept a weak but consistent improvement:

| cap | model | all delta MAE | AS1 delta | lt3 delta | gte6 delta |
|---:|---|---:|---:|---:|---:|
| 0.05 | residual LGBM | -0.00095 | -0.00587 | -0.00187 | -0.02015 |
| 0.10 | residual LGBM | -0.00105 | -0.00711 | -0.00377 | -0.02551 |
| 0.15 | residual LGBM | -0.00097 | -0.00763 | -0.00450 | -0.02793 |
| 0.20 | residual LGBM | -0.00093 | -0.00802 | -0.00479 | -0.02930 |
| 0.10 | tail-class gate | -0.00051 | -0.00160 | -0.02965 | -0.01315 |
| 0.20 | tail-class gate | +0.00102 | -0.00173 | -0.05013 | -0.02630 |

Current interpretation:

- The composite score is real, but its AS1 top-flag behavior should be treated
  as a hypothesis, not a fixed shift rule.
- A low-cap residual adapter is the safer next modeling object. It improves
  AS1 and both tails in Phase2 OOF while keeping global movement small.
- The tail classifier version aggressively fixes low tail but pays too much in
  mid-range calibration once the cap is increased.

## Test-Set Residual Adapter Diagnostic

For a non-submission diagnostic, the cap-0.10 residual adapter was fit on the
full labeled pool and applied to the id55 anchor as a proxy base. This is not a
clean AS1 estimate because AS1 labels are included in the residual fit; it is
useful for inspecting AS2 movement and preflight risk.

Generated files:

- `track1_activity/analysis/phase2_classifier_gate/outputs/composite_pairrank_chemprop_residual_cap010/test_residual_candidate.csv`
- `track1_activity/analysis/phase2_classifier_gate/outputs/composite_pairrank_chemprop_residual_cap010/test_residual_shift.csv`
- `track1_activity/analysis/phase2_candidate_scorer/outputs/composite_pairrank_chemprop_residual_cap010/`
- `track1_activity/analysis/submission_preflight/outputs/composite_pairrank_chemprop_residual_cap010_vs_id55/`

Diagnostics:

| check | value |
|---|---:|
| AS1 replay MAE vs id55 | 0.39635 vs 0.40657 |
| AS1 delta MAE vs id55 | -0.01021 |
| AS1 gte6 delta MAE | -0.06863 |
| AS2 mean abs shift | 0.04382 |
| AS2 p90 abs shift | 0.09857 |
| max abs shift | 0.10000 |
| id56-minus-id55 projection | -0.02713 |
| preflight verdict | PASS (`small_anchor_shift`) |

This diagnostic is directionally encouraging but should not be mistaken for an
honest submission estimate. The relevant honest evidence is still the Phase2 OOF
residual result above: about `-0.0010` global MAE, with larger improvements on
AS1 and high-tail slices.

## TabPFN Scalar Readouts

The same scalar panel was also tested with TabPFN residual and 3-class tail
classifier readouts. This was motivated by the original desire to model
`high / normal / low` directly.

Generated files:

- `track1_activity/analysis/phase2_classifier_gate/run_composite_tabpfn_probe.py`
- `track1_activity/analysis/phase2_classifier_gate/outputs/composite_pairrank_chemprop_tabpfn/`
- `track1_activity/analysis/phase2_classifier_gate/outputs/composite_pairrank_chemprop_tabpfn_ne32/`

With cap `0.10`, `n_estimators=32`:

| model | all delta MAE | AS1 delta | lt3 delta | gte6 delta | read |
|---|---:|---:|---:|---:|---|
| TabPFN residual | +0.00027 | -0.00466 | -0.02803 | -0.03546 | fixes tails but hurts mid-range/global |
| TabPFN tail classifier | -0.00017 | -0.00039 | -0.02423 | -0.00529 | stable but weaker than LGBM residual |
| LGBM residual cap0.10 | -0.00105 | -0.00711 | -0.00377 | -0.02551 | current best scalar-panel adapter |

The direct classification framing is not wrong, but with the current scalar
panel it is not the strongest global adapter. The best current recipe remains:
Boltz-style ChemProp scalar + assay pairrank composite, then a low-cap residual
model rather than a broad fixed gate or standalone TabPFN classifier.

## Train-Only Transfer Check

To separate AS1 adaptation from real transfer, a residual adapter was trained
only on source-train rows and transferred to AS1/test using the id55 anchor as
the base. This is a stricter AS1 check than the full `train + AS1` residual
diagnostic.

Generated files:

- `track1_activity/analysis/phase2_classifier_gate/run_composite_train_residual_transfer.py`
- `track1_activity/analysis/phase2_classifier_gate/outputs/composite_pairrank_chemprop_train_transfer_cap005/`
- `track1_activity/analysis/phase2_classifier_gate/outputs/composite_pairrank_chemprop_train_transfer_cap010/`
- `track1_activity/analysis/phase2_classifier_gate/outputs/composite_pairrank_chemprop_train_transfer_cap015/`

Train-only residual transfer improved tails but not total AS1 MAE:

| cap | AS1 total delta | lt3 delta | gte6 delta | read |
|---:|---:|---:|---:|---|
| 0.05 | +0.00018 | -0.00703 | -0.02898 | nearly neutral total, tail-positive |
| 0.10 | +0.00173 | -0.01342 | -0.04290 | tail-positive, mid-range hurt |
| 0.15 | +0.00190 | -0.01895 | -0.04494 | more aggressive, still total-negative |

This argues against using the scalar panel as a broad residual correction when
AS1 labels are not included. The stronger transfer behavior is sparse tail
gating with thresholds fixed from the train distribution.

Train-quantile gate scan:

| gate | AS1 delta MAE | AS1 flags | AS1 high | AS1 low | AS2 flags | preflight |
|---|---:|---:|---:|---:|---:|---|
| pairrank_chembl train q95, +0.30 | -0.00550 | 12 | 5 | 0 | 32 | HOLD |
| pairrank_chembl train q95, +0.20 | -0.00480 | 12 | 5 | 0 | 32 | HOLD |
| combo_high_htchem_cp02 train q98, +0.30 | -0.00474 | 9 | 5 | 0 | 7 | PASS |
| combo_high_htchem_cp02 train q98, +0.20 | -0.00396 | 9 | 5 | 0 | 7 | PASS |

Generated files:

- `track1_activity/analysis/phase2_classifier_gate/outputs/composite_pairrank_chemprop_train_quantile_gates/`
- `track1_activity/analysis/phase2_candidate_scorer/outputs/combo_high_htchem_cp02_trainq98_gates/`
- `track1_activity/analysis/submission_preflight/outputs/combo_high_htchem_cp02_trainq98_lift020_vs_id55/`
- `track1_activity/analysis/submission_preflight/outputs/combo_high_htchem_cp02_trainq98_lift030_vs_id55/`

The key practical read changed after the train-only check: the best use of the
Boltz-style ChemProp scalar is not a broad adapter, but a **sparse intersection
gate**. Pairrank alone finds a slightly stronger AS1 lift, but it flags many
more AS2 compounds and trips preflight. The composite gate keeps nearly the same
AS1 high-tail benefit while reducing AS2 movement to seven compounds and passing
preflight.

### Sparse Two-Sided Gate

A final small scan combined the train-quantile high gate with a much sparser
low-drop gate:

```text
high: combo_high_htchem_cp02 >= train q98, shift +0.20
low:  -cp_abs01 >= train q98, shift -0.20
```

This produced the best current conservative sparse diagnostic:

| metric | value |
|---|---:|
| AS1 MAE | 0.40182 |
| AS1 delta vs id55 | -0.00475 |
| AS1 shifted rows | 10 |
| AS1 true high among high flags | 5 / 9 |
| AS1 true low among low flags | 0 / 1 |
| AS2 shifted rows | 7 |
| AS2 low flags | 0 |
| mean abs test shift | 0.00663 |
| max abs shift | 0.20 |
| id56-minus-id55 projection | 0.00049 |
| preflight verdict | PASS (`small_anchor_shift`) |

Generated files:

- `track1_activity/analysis/phase2_classifier_gate/build_composite_train_quantile_gates.py`
- `track1_activity/analysis/phase2_classifier_gate/outputs/composite_pairrank_chemprop_train_quantile_gates/combo_high_low_fine_scan.csv`
- `track1_activity/analysis/phase2_classifier_gate/outputs/composite_pairrank_chemprop_train_quantile_gates/combo_q98_lift02__cp_abs01_lowq98_drop02.csv`
- `track1_activity/analysis/phase2_candidate_scorer/outputs/combo_high_low_train_quantile_gates_sparse/`
- `track1_activity/analysis/submission_preflight/outputs/combo_q98_lift02_cpabs01_lowq98_drop02_vs_id55/`

AS1 flags:

- High-lift flags include 5 true high compounds and 4 high-5.x compounds.
- The only low-drop AS1 flag is `OADMET-0006600` (`pec50=3.99`, id55 overpredicts
  at `4.38`), so the low side is not acting as a broad low-tail classifier here.
- AS2 flags are seven high-lift rows only:
  `OADMET-0006492`, `0006488`, `0006466`, `0006373`, `0006142`,
  `0006118`, `0006097`.

This is currently the cleanest practical artifact from the Boltz-style line:
it preserves most of the AS1 high-tail gain, adds only one low-side correction,
keeps AS2 movement tiny, and avoids the known bad axis.

## Phase 2 id63 Submission Decision

The final practical use of this line was not the sparse replacement gate above.
The submitted candidate kept the id62 ChEMBL pairrank gate and used the new
composite gate only to add high-confidence AS2 rows that id62 had not already
lifted.

Submitted local row:

- `lb_submissions.id = 63`
- `phase2_as1_aug_top500_id55blend_a0p45_pairrankchembl_q95_g0p15_plus_combo_new_h0p15_l0p15_labels_as1.csv`
- Submitted on 2026-06-28 JST; LB metrics pending.

Recipe:

```text
AS1 rows = released labels
AS2 rows = id55 + 0.45 * (AS1-aug top500 - id55)
         + old ChEMBL/public-PXR pairrank q95 high gate, +0.15
         + new composite high rows not already lifted by old gate, +0.15
```

The new composite additions are only:

- `OADMET-0006488`
- `OADMET-0006142`

The low-side setting is present in the generation script but is a no-op for the
submitted candidate: the safe `cp_abs01` train-q98 low gate has zero AS2 flags.
Looser low thresholds improved AS1 replay but moved too many rows and were not
used for this cooldown spend.

Alpha ladder against the current id62 anchor, AS2 only:

| alpha | AS1 model proxy MAE | mean abs shift | p90 abs shift | n > 0.05 | n > 0.10 | read |
|---:|---:|---:|---:|---:|---:|---|
| 0.40 | 0.28033 | 0.00115 | 0.00000 | 2 | 2 | safest composite-only update |
| 0.45 | 0.26456 | 0.00729 | 0.01263 | 2 | 2 | selected strict-small-move update |
| 0.50 | 0.24881 | 0.01343 | 0.02527 | 5 | 2 | balanced half-trust in AS1-aug model |
| 0.55 | 0.23308 | 0.01957 | 0.03790 | 19 | 2 | starts broad AS2 movement |
| 0.60 | 0.21735 | 0.02571 | 0.05054 | 28 | 4 | attack candidate |

The decision was to move beyond id62's `alpha=0.40` because the user-facing
concern was credible: id55 never trained on AS1, so anchoring too hard to id55
could leave AS1-continuation signal unused. The candidate stops at `alpha=0.45`
because previous LB history repeatedly punished broad top500-like moves despite
local improvements, while `alpha=0.45` still changes only two AS2 rows by more
than `0.05` relative to id62 and has negative projection onto the known id56
bad axis in preflight.

# Boltz affinity rank/delta probe

Date: 2026-06-30 JST.

This is the research-facing log for the Boltz affinity follow-up. It separates
the protein-aware rank/delta signal from the submission-risk checks in
`boltz_affinity_embedding_probe.md`.

## Question

The first Boltz trunk experiments used pooled trunk representations as PXR-aware
features:

- `pooled_boltz`: core-pocket pooling from trunk `s` / `z`
- `pooled_boltz_allpairs`: all protein-ligand pair pooling

The follow-up question was whether the Boltz-2 affinity path is more naturally
useful as a relative activity signal than as a calibrated pEC50 model. This was
tested in two ways:

- affinity-module embeddings: `compound_boltz2_affinity_reuse.affinity_g1/g2`
- raw Boltz affinity scalars: predicted affinity values and binary binding
  probabilities from `compound_boltz2_affinity_reuse`

## Affinity Path

In the patched Boltz checkout used for this repo, the affinity path is roughly:

```text
trunk z
 + s_inputs_i / s_inputs_j
 + predicted-pose distogram conditioning
 -> affinity PairFormer stack
 -> mean pool protein-ligand + intra-ligand pairs
 -> affinity_out_mlp
 -> 384d g
 -> affinity value / binary likelihood heads
```

The saved `affinity_g1/g2` vectors are the 384d outputs of
`affinity_out_mlp`, after affinity PairFormer processing and masked mean
pooling, but before the scalar heads. The scalar columns are the final Boltz
affinity head outputs.

## Absolute pEC50 Readout

Phase 1 style AS1 replay:

- Train on original `train_activity` only.
- Hold out released AS1 labels as a blind Phase 1 style test set.
- Do not fit on AS1 labels.
- Use TabPFN v2.6 for the main comparison because the existing pooled Boltz
  members were v2.6-era models.

AS1 results:

| feature | model | AS1 MAE | Spearman | note |
|---|---|---:|---:|---|
| `pooled_boltz_allpairs` | TabPFN v2.6 fresh replay | 0.4918 | 0.7735 | best fresh replay |
| `pooled_boltz` | TabPFN v2.6 fresh replay | 0.4946 | 0.7641 | close |
| `boltz_affinity_g1g2` | TabPFN v2.6 | 0.5284 | 0.7499 | signal, but weaker |
| `boltz_affinity_gmean` | TabPFN v2.6 | 0.5289 | 0.7345 | similar |
| `boltz_affinity_g1g2_scalars` | TabPFN v2.6 | 0.5333 | 0.7486 | scalars did not help |

Raw scalar affinity also has a clear monotone relationship to AS1 pEC50, but it
is not a calibrated pEC50 model:

| scalar | AS1 Pearson | AS1 Spearman | interpretation |
|---|---:|---:|---|
| `affinity_pred_value` | -0.5523 | -0.6235 | lower value means higher PXR activity |
| `affinity_probability_binary` | 0.5377 | 0.6012 | higher probability means higher PXR activity |
| `affinity_pred_value_1` | -0.4975 | -0.5760 | same orientation as ensemble mean |
| `affinity_pred_value_2` | -0.5006 | -0.5249 | same orientation as ensemble mean |
| `affinity_probability_binary_1` | 0.4757 | 0.5363 | weaker member |
| `affinity_probability_binary_2` | 0.5083 | 0.5519 | stronger member |

The sign of `affinity_pred_value` is energy-like: better binders have lower
predicted values.

## Pairwise Embedding Readout

The embedding probe used a Boltz-style pair objective:

- sample training pairs from original `train_activity`
- input `X_i - X_j`
- classify whether `pEC50_i > pEC50_j`
- regress `pEC50_i - pEC50_j`
- evaluate all AS1 pairs, with the main readout restricted to
  `|delta pEC50| >= 0.5`

Pairwise readout, 150k train pairs, no train-pair delta filter:

| feature | AS1 pair AUC | class accuracy | delta Spearman |
|---|---:|---:|---:|
| `pooled_boltz_allpairs` | 0.9433 | 0.8709 | 0.8106 |
| `pooled_boltz` | 0.9393 | 0.8610 | 0.8113 |
| `boltz_affinity_gmean` | 0.9237 | 0.8437 | 0.7567 |
| `boltz_affinity_g1g2` | 0.9237 | 0.8453 | 0.7606 |
| `boltz_affinity_g1g2_scalars` | 0.9225 | 0.8447 | 0.7614 |

With train pairs restricted to `|delta pEC50| >= 0.5`, affinity `g1g2`
improved slightly to pair AUC 0.9260, but `pooled_boltz_allpairs` also improved
to 0.9448.

## Pairwise Scalar Readout

The raw scalar affinity outputs are also strong relative-rank features. On AS1
pairs:

| scalar | pair filter | best-orientation AUC | best sign accuracy | delta Spearman |
|---|---|---:|---:|---:|
| `affinity_pred_value` | all pairs | 0.7889 | 0.7193 | -0.5794 |
| `affinity_probability_binary` | all pairs | 0.7800 | 0.7075 | 0.5623 |
| `affinity_pred_value` | `|delta pEC50| >= 0.5` | 0.8647 | 0.7890 | -0.6372 |
| `affinity_probability_binary` | `|delta pEC50| >= 0.5` | 0.8579 | 0.7767 | 0.6272 |
| `affinity_pred_value` | `|delta pEC50| >= 1.0` | 0.9105 | 0.8333 | -0.6573 |
| `affinity_probability_binary` | `|delta pEC50| >= 1.0` | 0.9053 | 0.8250 | 0.6533 |

The scalar outputs are therefore much more convincing as binary/ranking
features than as absolute pEC50 estimates. The signal gets stronger when the
true pEC50 gap is large.

## Interpretation

- Boltz affinity outputs preserve a real PXR activity-ranking signal.
- The scalar affinity value has the expected energy-like orientation: lower is
  stronger.
- The binary binding probability is easier to read directly and behaves as a
  useful rank score.
- Affinity embeddings and raw trunk pooling both improve under a pairwise
  framing, which matches the Boltz-2 affinity training philosophy.
- Even in the pairwise setting, raw-ish trunk pooling is stronger than the
  compressed affinity-head embedding for this PXR EC50 task.
- The likely failure mode is not absence of signal. It is mismatch between
  generic binding affinity / binding likelihood and PXR transcriptional EC50,
  plus poor calibration of the scalar output onto the pEC50 scale.

Practical read: Boltz affinity is worth treating as a protein-aware
rank/comparison axis. It should not be interpreted as a direct EC50 predictor
without a learned calibration layer and careful validation against assay-local
exceptions.

## Files

Scripts:

- `track1_activity/analysis/boltz_affinity_embedding_probe/run_probe.py`
- `track1_activity/analysis/boltz_affinity_embedding_probe/run_pairwise_probe.py`
- `track1_activity/analysis/boltz_affinity_embedding_probe/run_scalar_delta_probe.py`

Ignored generated outputs:

- `track1_activity/analysis/boltz_affinity_embedding_probe/outputs/`
- `track1_activity/analysis/boltz_affinity_embedding_probe/outputs_pairwise/`
- `track1_activity/analysis/boltz_affinity_embedding_probe/outputs_scalar_delta/`

Related background:

- `docs/track1_explain/models/tabpfn_pooled_boltz_trunk_umap.md`
- `db/boltz2_affinity_reuse_schema.sql`
- `track1_activity/boltz2/scripts/boltz2_affinity_reuse_postprocess.py`

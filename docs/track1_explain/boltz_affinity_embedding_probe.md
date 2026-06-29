# Boltz affinity embedding probe

Date: 2026-06-30 JST.

This note records an experiment-only follow-up to the Boltz trunk pooling work.
It does not change the Track 1 final submission. The goal was to test whether
the Boltz-2 affinity module's internal embedding is a better PXR activity
feature than the existing pooled trunk representations.

## Question

Earlier Track 1 work used Boltz trunk tensors as PXR-aware features:

- `pooled_boltz`: 1024d core-pocket pooling from trunk `s` / `z`
- `pooled_boltz_allpairs`: 1024d all protein-ligand pair pooling

The new question was different: Boltz-2's affinity head also creates an
internal pooled representation before the affinity value and binary heads. In
this repo that representation is stored as:

```text
compound_boltz2_affinity_reuse.affinity_g1  # 384d
compound_boltz2_affinity_reuse.affinity_g2  # 384d
```

The two vectors come from the two affinity ensemble members. The probe tested
`g1+g2` (768d), `mean(g1,g2)` (384d), and `g1+g2` plus affinity scalars.

## What the affinity embedding is

Boltz-2's affinity module is not just a scalar postprocess on the final pose.
In the patched official Boltz checkout, the affinity path roughly does:

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

So `affinity_g1/g2` are the 384d outputs of `affinity_out_mlp`, after the
affinity PairFormer and masked mean pooling, but before the final scalar heads.

This is more compressed and more task-shaped than raw trunk pooling. It is
trained for generic binding affinity / binding likelihood, not specifically
PXR transcriptional EC50.

## Phase 1 style AS1 replay

Setup:

- Train on original `train_activity` only.
- Hold out released AS1 labels as a Phase 1 style test set.
- No AS1 labels are used for model fitting.
- Main comparison uses TabPFN v2.6, because the existing pooled Boltz members
  were originally v2.6-era models.

Results on AS1:

| feature | model | AS1 MAE | Spearman | note |
|---|---|---:|---:|---|
| `pooled_boltz_allpairs` | TabPFN v2.6 fresh replay | 0.4918 | 0.7735 | best fresh replay |
| `pooled_boltz` | TabPFN v2.6 fresh replay | 0.4946 | 0.7641 | close |
| `boltz_affinity_g1g2` | TabPFN v2.6 | 0.5284 | 0.7499 | signal, but weaker |
| `boltz_affinity_gmean` | TabPFN v2.6 | 0.5289 | 0.7345 | similar |
| `boltz_affinity_g1g2_scalars` | TabPFN v2.6 | 0.5333 | 0.7486 | scalars did not help |

Existing production CSV AS1 replay for context:

| existing CSV | AS1 MAE | Spearman |
|---|---:|---:|
| `tabpfn_pooled_boltz_umap_default.csv` | 0.4879 | 0.7667 |
| `tabpfn_pooled_boltz_allpairs_umap_default.csv` | 0.4905 | 0.7731 |

TabPFN v3 showed the same ordering. `pooled_boltz_allpairs` remained best, and
the affinity embeddings stayed weaker.

## Pairwise objective probe

Because the Boltz-2 affinity paper emphasizes assay-local pairwise differences,
we also tested pairwise PXR readouts:

- Sample train pairs from original `train_activity`.
- Input is `X_i - X_j`.
- Fit LightGBM binary classifier for `pEC50_i > pEC50_j`.
- Fit LightGBM regressor for `pEC50_i - pEC50_j`.
- Evaluate all AS1 compound pairs, with a main readout on pairs where
  `|delta pEC50| >= 0.5`.

Pairwise readout, 150k train pairs, no train-pair delta filter:

| feature | AS1 pair AUC | class accuracy | delta Spearman |
|---|---:|---:|---:|
| `pooled_boltz_allpairs` | 0.9433 | 0.8709 | 0.8106 |
| `pooled_boltz` | 0.9393 | 0.8610 | 0.8113 |
| `boltz_affinity_gmean` | 0.9237 | 0.8437 | 0.7567 |
| `boltz_affinity_g1g2` | 0.9237 | 0.8453 | 0.7606 |
| `boltz_affinity_g1g2_scalars` | 0.9225 | 0.8447 | 0.7614 |

With train pairs restricted to `|delta pEC50| >= 0.5`, affinity `g1g2` improved
slightly to pair AUC 0.9260, but `pooled_boltz_allpairs` also improved to
0.9448.

Interpretation:

- Affinity embeddings look much better as ranking / difference features than as
  absolute pEC50 features.
- That matches the Boltz-2 affinity training philosophy.
- The high pair AUC should not be read as 21k independent test observations:
  each AS1 compound appears in many pairs, and the evaluation excludes small
  deltas in the main readout.
- Even under the pairwise framing, raw-ish trunk pooling is stronger than the
  affinity-head embedding for PXR.

## Gate probe

The pairwise models were also turned into transductive test-set scores:

```text
score(compound) = mean probability that compound beats the other Track 1 test compounds
```

These scores were scanned as sparse low/high gates on the id55 anchor and then
as tiny AS2-only overlays on id63. This was only a risk audit, not a submission
flow.

Best id55 AS1 replay rows:

| score | gate | AS1 MAE | delta vs id55 | AS1 flags | AS2 flags | AS1 true low/high flags |
|---|---|---:|---:|---:|---:|---|
| `pooled_boltz_allpairs` pairwise score | low q20, -0.10 | 0.4056 | -0.0010 | 57 | 46 | low 19 / high 0 |
| `pooled_boltz_allpairs` pairwise score | high q95, +0.05 | 0.4069 | +0.0003 | 8 | 18 | low 0 / high 3 |
| `boltz_affinity_g1g2` pairwise score | low q25, -0.05 | 0.4067 | +0.0001 | 68 | 61 | low 18 / high 0 |
| `boltz_affinity_g1g2` pairwise score | high q95, +0.05 | 0.4068 | +0.0003 | 10 | 16 | low 0 / high 3 |

The broad `pooled_boltz_allpairs` low gate had real directionality on AS1, but
the gain was very small. To be useful on the id55 low-tail failure it would
need a larger negative shift, which would move many AS2 compounds. Very small
AS2-only overlays on id63 passed preflight, but were too weak to justify
changing the final candidate.

Example AS2-only overlays versus id63:

| overlay | AS2 flags | shift | mean abs shift | preflight |
|---|---:|---:|---:|---|
| consensus low q05 | 5 | -0.03 | 0.00029 | PASS |
| pooled low q05 | 15 | -0.03 | 0.00088 | PASS |
| pooled low q20 | 46 | -0.03 | 0.00269 | PASS |
| pooled low q20 | 46 | -0.10 | 0.00897 | PASS |

Final read: this is a real low-activity signal, but not enough to replace the
Phase 2 final decision. It is more useful as a research direction for
protein-aware activity ranking than as a last-minute submission edit.

## Files

Scripts:

- `track1_activity/analysis/boltz_affinity_embedding_probe/run_probe.py`
- `track1_activity/analysis/boltz_affinity_embedding_probe/run_pairwise_probe.py`
- `track1_activity/analysis/boltz_affinity_embedding_probe/run_pairwise_gate_scan.py`

Generated outputs are intentionally ignored under:

- `track1_activity/analysis/boltz_affinity_embedding_probe/outputs/`
- `track1_activity/analysis/boltz_affinity_embedding_probe/outputs_pairwise/`
- `track1_activity/analysis/boltz_affinity_embedding_probe/outputs_pairwise_gate/`

Related background:

- `docs/papers/boltz2_affinity_notes.md`
- `docs/track1_explain/models/tabpfn_pooled_boltz_trunk_umap.md`
- `db/boltz2_affinity_reuse_schema.sql`
- `track1_activity/boltz2/scripts/boltz2_affinity_reuse_postprocess.py`

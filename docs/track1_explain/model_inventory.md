# Track 1 Model And Method Inventory

This is a practical taxonomy for explaining the Track 1 solution. It groups the
work by modeling idea rather than by chronological PR.

## Production Members

| Family | Representative model(s) | Plain-English explanation | Status |
|---|---|---|---|
| 2D + Boltz + `log2_fc` TabPFN | [`tabpfn_cheme_2d_full_boltz_log2fc_pred_optuna_trial10_seed5ens_umap_default`](models/tabpfn_cheme_2d_full_boltz_log2fc_pred_optuna_trial10_seed5ens_umap_default.md) | A large tabular feature set combining 2D descriptors, Boltz-derived features, CheMeleon fingerprints, and predicted low-fidelity activity. | Core production member. |
| Top-500 selected TabPFN | `tabpfn_cheme_2d_full_boltz_log2fc_pred_seed10ens_top500_umap` | The same broad feature universe, reduced to the 500 most useful features per fold before TabPFN. Related diagnostic doc: [`optuna_trial10_seed5ens_top500`](models/tabpfn_cheme_2d_full_boltz_log2fc_pred_optuna_trial10_seed5ens_top500_umap.md). | Core production member and strong anchor family. |
| ChemProp pretrain embedding | [`tabpfn_chemprop_pretrain_embed_umap_default`](models/tabpfn_chemprop_pretrain_embed_umap_default.md) | Train a ChemProp encoder on auxiliary `log2_fc`, freeze it, extract embeddings, then train TabPFN on pEC50. | Strong repeatable axis. |
| MoLFormer-c3 pretrain embedding | [`tabpfn_molformer_c3_pretrain_embed_umap`](models/tabpfn_molformer_c3_pretrain_embed_umap.md) | Transformer-family version of the same low-fidelity pretrain then frozen-embedding recipe. | Kept for diversity. |
| KERMT/GROVER pretrain embedding | [`tabpfn_kermt_pretrain_embed_umap_default`](models/tabpfn_kermt_pretrain_embed_umap_default.md) | Graph-transformer embedding trained on auxiliary activity, then reused as TabPFN input. | Kept for diversity and strong OOF contribution. |
| AttentiveFP pretrain embedding | [`tabpfn_attentivefp_pretrain_embed_umap_default`](models/tabpfn_gnn_pretrain_embed_aux_members.md) | Graph-attention encoder pretrained on `log2_fc`, frozen, then used as a TabPFN feature source. | Kept, lower weight but useful diversity. |
| GatedGCN pretrain embedding | [`tabpfn_gatedgcn_pretrain_embed_umap_default`](models/tabpfn_gnn_pretrain_embed_aux_members.md) | Edge-conditioned graph neural net embedding, hidden size 512, pretrained on `log2_fc`. | Kept, especially for decorrelation. |
| Boltz trunk embeddings | [`tabpfn_pooled_boltz_umap_default`, `tabpfn_pooled_boltz_allpairs_umap_default`](models/tabpfn_pooled_boltz_trunk_umap.md) | Protein-ligand trunk representations pooled into tabular vectors. | Kept as auxiliary structural signal. |

## Main Method Patterns

### Low-Fidelity Pretrain Then Frozen Embedding

This became the central recipe.

1. Use auxiliary single-concentration `log2_fc` data to train an encoder.
2. Freeze the encoder.
3. Extract one fixed vector per compound.
4. Train a downstream pEC50 model, usually TabPFN.

Why it worked: the auxiliary assay has broad chemical coverage and teaches the
encoder a PXR-relevant activity axis before the blinded pEC50 task.

### Large Tabular Feature Set Then TabPFN

The strongest tabular members combine:

- classical 2D descriptors/fingerprints;
- Boltz-derived pose/trunk signals;
- CheMeleon/foundation-model fingerprints;
- predicted `log2_fc` scalars.

TabPFN then acts as the downstream learner. Feature selection to top 500 is
important because TabPFN behaves better when the huge feature matrix is compressed
in a leak-free per-fold way.

### Caruana Bagged Ensemble

The production ensemble uses bagged Caruana forward selection. This is a
discrete, count-based ensemble strategy that is less eager than continuous
optimizers to put all weight on one locally strong but correlated member.

Why it mattered: continuous optimizers repeatedly found impressive OOF gains
that moved public-LB predictions in bad directions.

### Calibration And Gates

Calibration started as a win and later became a risk.

- Early positive-slope affine calibration improved the public leaderboard.
- Later local calibration/gating variants around id55/id57 were flat or
  slightly negative on public LB.
- The current conclusion is to keep calibration tooling for Phase 2, when labels
  can anchor it.

## Negative Or Historical Families

| Family | What was tried | Outcome |
|---|---|---|
| Direct pEC50 neural fine-tuning | MoLFormer-XL LoRA and direct GNN-style models. | Often improved local metrics or built useful infrastructure, but did not survive the production allow-list. |
| Direct graph neural nets | GIN, GraphGPS, direct AttentiveFP/GatedGCN, and later KA-GNN probes. | Generally too weak as direct pEC50 predictors; useful only when a strong pretrain-embedding recipe rescued the backbone. |
| FMGCL-style auxiliary loss | Relative-distance auxiliary loss added to ChemProp-style training. | Regressed on OOF; framework kept, member dropped. |
| Extra correlated ChemProp/log2_fc variants | Optuna and multi-seed additions beyond the main swap. | OOF-positive but LB-negative due to over-concentration in one family. |
| Boltz small feature swaps | Additional confidence, pose, or trunk-derived variants. | Useful diagnostics, but repeated small variants did not transfer reliably. |
| Public-LB proxy gates | `log2_fc`, ring-count, family-gap, high-activity, and potent46 gates. | Good for diagnostics; not reliable enough for more Phase 1 submissions. |
| External ChEMBL judge | Nearest-neighbor comparison to filtered ChEMBL PXR activation records. | Coverage and assay transfer were too weak for submission gating. |

## Suggested Mental Model

Explain the final Track 1 system as:

1. Build several different views of the molecule.
2. Make the strongest views PXR-aware using auxiliary low-fidelity activity.
3. Convert those views into TabPFN predictions.
4. Blend only the members that add either strength or useful diversity.
5. Avoid overreacting to tiny OOF gains because the public analog subset does
   not exactly match the local validation split.

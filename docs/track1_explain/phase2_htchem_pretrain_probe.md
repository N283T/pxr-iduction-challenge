# Phase 2 HTChem ChemProp Pretrain Probe

Purpose: test whether HTChem helps as an encoder pretraining signal rather than as direct Track 1 training rows or a standalone `pred_htchem` scalar.

## Setup

- Pretrain model: ChemProp MPNN, same default architecture as `run_chemprop_pretrain.py`.
- Heads:
  - `log2fc_8p25`, 10,752 labels
  - `log2fc_33`, 9,527 labels
  - `htchem_corrected_pec50`, 441 labels
- Task weights: `1.0 / 0.5 / 0.25`.
- Training stopped at epoch 18 by early stopping.
- Output embedding: `data/chemprop_log2fc_htchem_pretrain_embed.parquet`.

## Single-Member OOF

| model | OOF MAE | OOF RAE | OOF Spearman |
|:--|--:|--:|--:|
| `tabpfn_chemprop_pretrain_embed_umap_default` | 0.4371 | 0.4854 | 0.8073 |
| `tabpfn_chemprop_log2fc_htchem_pretrain_embed_umap_default` | 0.4506 | 0.5005 | 0.7885 |

## AS1 Replay

| model | AS1 MAE | bias | Spearman |
|:--|--:|--:|--:|
| `id55_anchor` | 0.4066 | 0.0516 | 0.8488 |
| `tabpfn_chemprop_log2fc_htchem_pretrain_embed_umap_default` | 0.4351 | 0.0607 | 0.8316 |
| `tabpfn_chemprop_pretrain_embed_umap_default` | 0.4441 | -0.0243 | 0.8307 |

## Ensemble Probe

| variant | OOF MAE | delta MAE | Spearman | delta Spearman | old embed weight | new embed weight |
|:--|--:|--:|--:|--:|--:|--:|
| baseline | 0.3958 | 0.0000 | 0.8467 | 0.0000 | 0.1117 | 0.0000 |
| swap old ChemProp embed to HTChem pretrain embed | 0.3967 | +0.0010 | 0.8455 | -0.0011 | 0.0000 | 0.0840 |
| add HTChem pretrain embed | 0.3937 | -0.0020 | 0.8484 | +0.0018 | 0.0723 | 0.0199 |

## Read

The HTChem pretrain encoder is AS1-positive versus the old ChemProp embed, but OOF-negative and SWAP-negative. The ADD result is mildly OOF-positive with only a tiny new-member weight, so it is not strong enough to promote. This supports the earlier conclusion: HTChem is useful as a local diagnostic/annotation signal, but its 441 labels are not enough to improve the broad log2fc-pretrained encoder.

# KA-GNN Pretrain Embedding Report

Date: 2026-05-09

## Setup

- Pretrain model: Fourier KA-GNN, hidden=128, layers=3, grid=3, mean aggregation/pooling.
- Auxiliary targets: `single_concentration.log2_fc_estimate` at 8.25 uM and 33 uM.
- Pretrain data: 13,136 compounds; random 90/10 validation split.
- Checkpoint: `track1_activity/checkpoints/ka_gnn_pretrain/pretrain.pt` (generated, not git-tracked).
- Frozen embedding: `data/ka_gnn_pretrain_embed.parquet` with shape `(4653, 128)` (generated, not git-tracked).

## Pretrain Fit

- Best validation weighted MSE: `0.381689`.
- Epochs run: `51` (early stopped; best epoch 39).
- Embedding extraction stats: `mean_abs=0.543603`, `std=0.672291`.

## Track 1 Frozen-Embedding Downstream Results

| Experiment | Downstream | MAE | RAE | Spearman | Ensemble ADD weight |
|---|---|---:|---:|---:|---:|
| `tabpfn_ka_gnn_pretrain_embed_umap_default` | TabPFN | `0.502228` | `0.551942` | `0.741151` | `0.0000` |
| `kan_ka_gnn_pretrain_embed_ka_gnn_pretrain_full_h32_lr1e3_umap` | pykan MLP replacement | `0.537695` | `0.590919` | `0.720650` | `0.0000` |

## Encoder-Finetune Check

Also tested pEC50 fine-tuning from the pretrained KA-GNN encoder with a fresh KAN readout:

| Experiment | MAE | RAE | Spearman | Ensemble ADD weight |
|---|---:|---:|---:|---:|
| `ka_gnn_pretrain_finetune_fourier_h128_l3_g3_meanaggr_lr1e3_umap` | `0.545506` | `0.599503` | `0.681830` | `0.0000` |

For reference, the previous direct KA-GNN from scratch was `0.569189` MAE, so auxiliary pretraining helps the raw graph model a little but remains far below the pool gate.

## Correlations

Prediction correlations against key existing axes:

| Candidate | ChemProp pretrain TabPFN | GatedGCN pretrain TabPFN | KERMT pretrain TabPFN | Direct KA-GNN | ChemProp-embed KAN | Current Caruana ensemble |
|---|---:|---:|---:|---:|---:|---:|
| TabPFN on KA-GNN pretrain embed | `0.9192` | `0.9373` | `0.9260` | `0.8917` | `0.8991` | `0.9269` |
| KAN on KA-GNN pretrain embed | `0.8889` | `0.9074` | `0.8926` | `0.8717` | `0.8722` | `0.8964` |
| Pretrained KA-GNN fine-tune | `0.8584` | `0.8842` | `0.8691` | `0.8979` | `0.8359` | `0.8711` |

## Decision

Do not add to the material pool and do not submit.  The frozen KA-GNN pretrain embedding is weaker than ChemProp/KERMT/GatedGCN embeddings, and the ensemble optimizer assigns zero weight to all tested KA-GNN-pretrain variants.

Interpretation: low-fidelity KA-GNN pretraining does learn a sensible activity-related representation (better than direct KA-GNN from scratch after fine-tuning), but without a strong pretrained molecular backbone it does not create a useful new leaderboard axis.

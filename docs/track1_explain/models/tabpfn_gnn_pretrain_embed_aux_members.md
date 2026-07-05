# GNN pretrain-embed auxiliary members

確認日: 2026-05-18 JST

対象モデル:

- `tabpfn_attentivefp_pretrain_embed_umap_default`
- `tabpfn_gatedgcn_pretrain_embed_umap_default`

## 位置づけ

この2つは、ChemProp / KERMT / MoLFormer と同じ
「low-fidelity pretrain -> frozen embedding -> TabPFN」レシピを、
別のGNN backboneで試した補助メンバー。

一言でいうと、
「単体最強ではないが、強い `cheme/top500/log2fc` 系にensemble weightが
寄りすぎるのを防ぐ diversity reserve」。

最終的なweightは小さく見えるが、drop試験ではこれらを消すと
Caruanaが強い同系統モデルへweightを再配分し、public LBで悪化した。
したがって、現時点では「weightが小さいから削る」対象ではない。

## 何をしているか

共通の流れ:

1. `compounds.std_smiles` から PyTorch Geometric の分子graphを作る。
2. 8.25 uM と 33 uM の single-concentration `log2_fc` を2-head targetにする。
3. GNNを `log2_fc` でpretrainする。
4. 最終の予測headを外し、分子レベルのhidden vectorを固定embeddingとして抽出する。
5. そのembeddingを TabPFN に入れて、最終の pEC50 を予測する。

つまり、AttentiveFP / GatedGCN が直接 pEC50 を出しているわけではない。
PXRに関係する弱いactivity signalをGNN表現に染み込ませ、
TabPFNがその表現をdownstreamで使う構成。

## AttentiveFP版

AttentiveFP は graph attention 系のGNN。
原子・結合特徴からmessage passingを行い、GRU/attentionベースのreadoutで
molecule-level vectorを作る。

今回の設定:

| item | value |
|---|---:|
| checkpoint | `track1_activity/checkpoints/attentivefp_pretrain/pretrain.pt` |
| embedding file | `data/attentivefp_pretrain_embed.parquet` |
| embedding dim | 512 |
| hidden channels | 512 |
| layers | 4 |
| timesteps | 3 |
| dropout | 0.1 |
| learning rate | 0.00033232775773741117 |
| batch size | 128 |
| best val loss | 0.7011593688618053 |

抽出時は `model.lin2 = nn.Identity()` とし、
2-head `log2_fc` projectionの手前、512次元のmolecular readoutを保存する。

再現用の主な入口:

```bash
pixi run python track1_activity/scripts/run_attentivefp_embed_extract.py
pixi run python track1_activity/scripts/run_train.py \
  --model tabpfn \
  --feature attentivefp_pretrain_embed \
  --split umap \
  --trials 0
```

## GatedGCN版

GatedGCN は edge-conditioned / gated message passing 系のGNN。
この実装では PyG の `ResGatedGraphConv` stack を使い、
最終的に `global_mean_pool` で分子レベルのvectorを作る。

初期の h128 checkpoint は単体性能が弱かったため、
h512でre-pretrainした版が採用された。

今回の設定:

| item | value |
|---|---:|
| checkpoint | `track1_activity/checkpoints/gatedgcn_pretrain/pretrain.pt` |
| old h128 checkpoint | `track1_activity/checkpoints/gatedgcn_pretrain/pretrain_h128.pt` |
| embedding file | `data/gatedgcn_pretrain_embed.parquet` |
| embedding dim | 512 |
| hidden dim | 512 |
| layers | 4 |
| dropout | 0.05 |
| learning rate | 0.0006655177988301952 |
| batch size | 64 |
| best val loss | 0.7477596132528215 |

抽出時は `model.ffn = nn.Identity()` とし、
FFN headの手前、512次元の `global_mean_pool` outputを保存する。

再現用の主な入口:

```bash
pixi run python track1_activity/scripts/run_gatedgcn_embed_extract.py
pixi run python track1_activity/scripts/run_train.py \
  --model tabpfn \
  --feature gatedgcn_pretrain_embed \
  --split umap \
  --trials 0
```

## 単体性能

直接GNNや frozen-head finetune は弱めだったが、
embeddingを抜いてTabPFNに渡すとかなり改善した。
この流れは ChemProp で最初に効いた recipe を、
別backboneへ横展開したもの。

| model | feature / recipe | OOF MAE | Spearman | note |
|---|---|---:|---:|---|
| `attentivefp_optuna_umap` | direct AttentiveFP | 0.5280 | 0.6984 | 不採用 |
| `attentivefp_pretrain_finetune_frozen_umap` | pretrained GNN + head | 0.5222 | 0.7321 | 不採用 |
| `tabpfn_attentivefp_pretrain_embed_umap_default` | 512d embed + TabPFN | 0.4843 | 0.7598 | 採用 |
| `tabpfn_attentivefp_pretrain_embed_umap_default_v3` | TabPFN v3側の再評価 | 0.4824 | 0.7600 | 診断 |
| `gatedgcn_optuna_umap` | direct GatedGCN | 0.5463 | 0.6965 | 不採用 |
| `gatedgcn_pretrain_finetune_frozen_umap` | pretrained GNN + head | 0.5076 | 0.7424 | 不採用 |
| `tabpfn_gatedgcn_pretrain_embed_umap_default` | h512 512d embed + TabPFN | 0.4739 | 0.7642 | 採用 |
| `tabpfn_gatedgcn_pretrain_embed_umap_default_v3` | TabPFN v3側の再評価 | 0.4753 | 0.7635 | 診断 |

GatedGCNは h128 embed の単体OOF MAEが約0.4902で、
h512 re-pretrainにより 0.4739 まで改善した。
このため、現在の採用版は h512。

## 試したが採用しなかった他のGNN

他のGNN系もいくつか試している。
結論として、PXRでは「raw molecular graphを直接pEC50に当てる」方向は弱く、
強い外部/補助signalでpretrainしたbackboneからembeddingを抜く方が有効だった。

| model | recipe | OOF MAE | Spearman | decision |
|---|---|---:|---:|---|
| `gin_optuna_umap` | GIN/GINEConv direct pEC50 | 0.5712 | 0.6810 | 不採用 |
| `graphgps_optuna_umap` | GraphGPS direct pEC50 | 0.5714 | 0.6642 | 不採用 |
| `attentivefp_optuna_umap` | AttentiveFP direct pEC50 | 0.5280 | 0.6984 | pretrain-embedに置換 |
| `gatedgcn_optuna_umap` | GatedGCN direct pEC50 | 0.5463 | 0.6965 | pretrain-embedに置換 |
| `ka_gnn_best_fourier_h128_l3_g3_meanaggr_lr1e3_umap` | Fourier KA-GNN direct pEC50 | 0.5691 | 0.6605 | ADD weight 0 |
| `ka_gnn_pretrain_finetune_fourier_h128_l3_g3_meanaggr_lr1e3_umap` | KA-GNN `log2_fc` pretrain -> finetune | 0.5454 | 0.6811 | ADD weight 0 |
| `tabpfn_ka_gnn_pretrain_embed_umap_default` | KA-GNN `log2_fc` embed + TabPFN | 0.5021 | 0.7399 | ADD weight 0 |

`run_ensemble.py` の初期pruneコメントにも、
GIN / GraphGPS は「near-zero weight, OOF-weakest graph nets」として
落とした記録が残っている。
issue #100でも、GIN / GraphGPS のpretrain from scratchは
phase-1 costに対して、既存baselineが弱くROIが微妙として後回しになった。

KA-GNNは2026-05-09に別途深掘りした。
Fourier KAN message passing のPyG移植と、
`log2_fc` pretrain -> frozen embedding -> downstream の両方を試したが、
best direct modelでも OOF MAEは約0.569、
TabPFN-on-KA-GNN-pretrain-embedでも約0.502に留まった。
decorrelation自体は一部あり、たとえば recorded direct KA-GNN は
ensembleとの residual correlation が低めだったが、単体精度が足りず
Caruana ADD weight は 0.0000 だった。

この結果から見ると、GNN一般が全く駄目というより、
「PXRの4140 trainだけで直接graph modelを育てる」のが難しい。
ChemProp / KERMT / AttentiveFP / GatedGCN のうち残ったものは、
単にGNNだから残ったのではなく、`log2_fc` pretrainを通じて
PXR-awareな固定表現を作れたもの、という理解がよい。

## なぜ残っているか

単体性能だけで見ると、ChemProp / KERMT / 2D top500 系より弱い。
それでも残っている理由は、同じ `log2_fc` signalでもbackboneが違うため、
エラーの出方が完全には一致しないから。

追加直後のissue #100記録では、AttentiveFP版は
12-pool Caruana OOF MAEを 0.4268 から 0.4242 に改善し、
weight 0.0223 を持った。

GatedGCN h512版は単体 OOF MAE 0.4740 で、
swap構成の Caruana weight は 0.036。
OOF上のensemble改善は noise-level だったが、
当時のLBでは MAE 0.4318 から 0.4244 へ改善した。
issue #100では、MoLFormer-c3との Pearson r が 0.912 と低めで、
decorrelation が効いた可能性として整理されている。

後期の9-poolではweightがかなり小さくなった。

| member | approximate later weight | interpretation |
|---|---:|---|
| `tabpfn_gatedgcn_pretrain_embed_umap_default` | 0.018 | small but nonzero diversity reserve |
| `tabpfn_attentivefp_pretrain_embed_umap_default` | 0.002 | almost zero locally, but not safely droppable |

ここで重要なのは、Caruana weightが小さいことと、
blind testで役に立たないことは同じではなかった、という点。

## Drop試験と結論

drop試験は2種類ある。

1つ目は、古い direct GNN 系を落とす試験。
`chemprop_optuna`、`chemprop_chemeleon`、
`attentivefp_optuna`、`gatedgcn_pretrain_finetune_frozen` のような
直接予測系は、pretrain-embed版に置き換わったのでdropされた。
これはこの2つのpretrain-embed memberを落とした話ではない。

2つ目が、今回の論点である low-weight pretrain-embed member のdrop試験。

Phase 4後の `32_drop_lowweight_members.py` では、
AttentiveFP / GatedGCNを落とすとOOFは少し良く見えた。
ただし、その改善幅は Caruana bagging のrun間varianceである
約±0.003の範囲内だった。

| variant | n | OOF MAE | Spearman | Δ MAE | Δ Sp |
|---|---:|---:|---:|---:|---:|
| baseline_10pool | 10 | 0.3953 | 0.8462 | - | - |
| drop_attentivefp | 9 | 0.3944 | 0.8476 | -0.0009 | +0.0014 |
| drop_gatedgcn | 9 | 0.3944 | 0.8477 | -0.0009 | +0.0015 |
| drop_both | 8 | 0.3924 | 0.8496 | -0.0029 | +0.0034 |

その後の `34_drop_lowweight.py` では、
AttentiveFP、GatedGCN、`pooled_boltz_allpairs` を順に落とすと、
OOFは単調に改善した。

| variant | dropped members | OOF Δ MAE |
|---|---|---:|
| drop_att | AttentiveFP | -0.0042 |
| drop_att_gate | AttentiveFP + GatedGCN | -0.0049 |
| drop_att_gate_pb | AttentiveFP + GatedGCN + pooled_boltz_allpairs | -0.0059 |

しかし、この方向はpublic LBで崩れた。
id41 `drop_att_gate_pb` は OOF MAE -0.0059 とかなり良く見えたが、
public LBでは MAE +0.0057、Spearman -0.0096 と悪化した。

原因としてissue #100では、weight集中が整理されている。

| configuration | chemprop / cheme family share | LB effect |
|---|---:|---|
| baseline / reference pools | about 0.61-0.76 | stable to marginal zone |
| Phase 4 trial10+11 ADD id38 | about 0.85 | LB MAE +0.0031 |
| drop_att_gate_pb id41 | about 0.94 | LB MAE +0.0057 |

id41では、6-pool weights が
`cheme_t10_seed5ens` 0.576 + `cheme_top500` 0.362 に集中し、
chemprop/cheme family shareが約0.94まで上がった。
つまり、low-weight memberをdropしたことで、
Caruanaが「削ったweight分だけ」ではなく、より大きく
top500/log2fc系へ再配分してしまった。

さらに、`35_drop_att_gate_submit.py` で予定されていた
AttentiveFP + GatedGCN だけを落とす7-pool variantも、
実測 family share が 0.864 まで上がったため取りやめになった。
issue #100では、
`att/gate 0.020 drop` が `cheme +0.107` に増幅された教訓として記録されている。

反対に、id42 の family-meta実験では、
強い chemprop/cheme family を1つのmeta memberに畳むと、
AttentiveFP / GatedGCN のweightが 0.076 / 0.067 まで自然に戻った。
これは、この2つが完全なノイズではなく、
強い同系統memberの過集中が緩んだときに使われる
別方向の補助signalだったことを示している。

最終的な結論:

- OOFだけを見ると、AttentiveFP / GatedGCN drop は良さそうに見えることがある。
- しかし、それは多くの場合 `cheme/top500/log2fc` 系へのweight集中を伴う。
- このweight集中はpublic LBで悪化しやすい。
- よって、AttentiveFP / GatedGCN pretrain-embed は
  「小さいが残す」補助メンバーとして扱う。

## 再現性メモ

現在確認できるartifact:

| artifact | status |
|---|---|
| `data/attentivefp_pretrain_embed.parquet` | exists, 13,136 rows x 512 dims, NaNなし |
| `data/gatedgcn_pretrain_embed.parquet` | exists, 13,136 rows x 512 dims, NaNなし |
| `track1_activity/checkpoints/attentivefp_pretrain/pretrain.pt` | exists |
| `track1_activity/checkpoints/gatedgcn_pretrain/pretrain.pt` | exists |
| `track1_activity/checkpoints/gatedgcn_pretrain/pretrain_h128.pt` | exists, older h128 |
| `track1_activity/submissions/tabpfn_attentivefp_pretrain_embed_umap_default.csv` | exists |
| `track1_activity/submissions/tabpfn_gatedgcn_pretrain_embed_umap_default.csv` | exists |

feature parquetとcheckpointは残っているので、
既存artifactからの再学習・再評価は可能。

注意点:

- checkpointのpretrain splitはrandom 90/10で、UMAP CVとは別。
- full pretrainから完全再現する場合はGPU環境と乱数seedの影響を受ける。
- 実務上は、既存の `*_pretrain_embed.parquet` を固定入力として使えば、
  downstream TabPFN側の再現性はかなり高い。
- drop可否は単体OOFではなく、family share とLB-negative anchorへの近さで判断する。

## 参照した主な記録

- GitHub issue #100:
  - PR #104 AttentiveFP pretrain-embed
  - PR #105 GatedGCN h512 pretrain-embed
  - PR #106 direct GNN drop
  - 2026-04-27 drop_lowweight / family concentration notes
  - 2026-04-28 family-meta notes
- `track1_activity/scripts/run_attentivefp_embed_extract.py`
- `track1_activity/scripts/run_gatedgcn_embed_extract.py`
- `track1_activity/scripts/boltz_affhead/32_drop_lowweight_members.py`
- `track1_activity/scripts/boltz_affhead/34_drop_lowweight.py`
- `track1_activity/scripts/boltz_affhead/35_drop_att_gate_submit.py`
- `track1_activity/scripts/run_gin_optuna.py`
- `track1_activity/scripts/run_graphgps_optuna.py`
- `track1_activity/src/ka_gnn.py`
- `track1_activity/scripts/run_ka_gnn.py`
- `track1_activity/scripts/run_ka_gnn_pretrain.py`
- `track1_activity/scripts/run_ka_gnn_embed_extract.py`

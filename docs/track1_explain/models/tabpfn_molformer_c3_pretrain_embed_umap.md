# tabpfn_molformer_c3_pretrain_embed_umap

確認日: 2026-05-18 JST

## 位置づけ

これは MoLFormer-c3 を single-concentration `log2_fc` でpretrainし、
凍結した 768次元 `[CLS]` embedding を TabPFN に入れたモデル。

ChemProp pretrain embed と同じ
「low-fidelity pretrain -> frozen embedding -> TabPFN」レシピだが、
backbone は graph GNN ではなく SMILES transformer。

一言でいうと、これは
「`log2_fc` signal を SMILES language-model 系の表現として ensemble に入れる」
ためのモデル。

## 何をしているか

学習の流れ:

1. `DeepChem/MoLFormer-c3-1.1B` を backbone として使う。
2. `compounds.std_smiles` の 13,136 compounds について、
   8.25 uM と 33 uM の `log2_fc` を2-head targetとして作る。
3. MoLFormer-c3 に LoRA を入れて、NaN-masked MSEで `log2_fc` をpretrainする。
4. pretrain後、backboneを固定し、train/test 4653 compoundsについて
   `[CLS]` の 768次元 embedding を抽出する。
5. その 768次元 embedding を TabPFN に入れて pEC50 を予測する。

このモデルは、pEC50をMoLFormerで直接fine-tuneしているわけではない。
MoLFormerはあくまで `log2_fc` でPXR-awareな表現を作るために使い、
最終のpEC50回帰は TabPFN に任せている。

## MoLFormer-c3 の中身

MoLFormer は SMILES を token列として読む transformer encoder。
KERMT/GROVER のように明示的な molecular graph を扱うのではなく、
SMILES language model として分子表現を学習する。

今回使ったのは [DeepChem/MoLFormer-c3-1.1B](https://huggingface.co/DeepChem/MoLFormer-c3-1.1B)。
設計ログ上では
hidden size 768、12 layers、最大長 202 の encoder として扱っている。
名前の `1.1B` は実装上のparameter数ではなく、pretraining側の規模を指す名前として扱う。
Hugging Face の model card でも model size は 46.8M params と表示されている。
実際の downstream feature は、各compoundにつき `[CLS]` 768次元だけ。

[ChemBERTa3 repository](https://github.com/deepforestsci/chemberta3) のREADMEでは、
MoLFormer-c3 系は次のように整理されている。

| model | pretraining data |
|---|---|
| c3-MoLFormer-1.1B | 100% ZINC20 + 100% PubChem |
| c3-MoLFormer-550M | 50% ZINC20 + 50% PubChem |
| c3-MoLFormer-100M | 10% ZINC20 |

したがって、今回のモデルは **550Mではなく1.1B**。
`track1_activity/src/peft_backbones.py` でも `hf_id` は
`DeepChem/MoLFormer-c3-1.1B` になっており、
pretrain checkpoint metadata も `backbone: molformer_c3_1_1b` を指している。

pretrain時の設定:

| item | value |
|---|---:|
| backbone | `molformer_c3_1_1b` |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.1 |
| LoRA target | `qkvo` |
| head hidden dim | 256 |
| head dropout | 0.1 |
| backbone lr | 0.0002 |
| head lr | 0.001 |
| batch size | 64 |
| max epochs | 50 |
| patience | 10 |
| best val loss | 0.5065 |

## なぜ入っているか

採用理由は、単体最強ではないが、backbone family が違うこと。

`tabpfn_molformer_c3_pretrain_embed_umap` の単体OOFは
MAE 0.4752 / Spearman 0.7620。
ChemPropやKERMTよりは明確に弱い。
それでも、現行の `ens_caruana_bag20` では weight 0.0403 が残っている。

つまり、MoLFormer-c3 は「強い主力」ではなく、
SMILES transformer family の多様性を入れる補助memberとして説明するのがよい。

| model | family | OOF MAE | Spearman | current Caruana weight |
|---|---|---:|---:|---:|
| `tabpfn_chemprop_pretrain_embed_umap_default` | D-MPNN | 0.4371 | 0.8073 | 0.1515 |
| `tabpfn_kermt_pretrain_embed_umap_default` | graph transformer | 0.4484 | 0.7891 | 0.1107 |
| `tabpfn_molformer_c3_pretrain_embed_umap` | SMILES transformer | 0.4752 | 0.7620 | 0.0403 |
| `tabpfn_gatedgcn_pretrain_embed_umap_default` | gated GNN | 0.4739 | 0.7642 | 0.0175 |
| `tabpfn_attentivefp_pretrain_embed_umap_default` | graph attention | 0.4843 | 0.7598 | 0.0024 |

## 直接fine-tuneとの違い

MoLFormer-XL の pEC50 direct PEFT/LoRA も試している。
しかし、`peft_molformer_xl_lora_r32a64_umap_default` は
OOF MAE 0.5290 / Spearman 0.7050 で、production allow-listからは落ちた。
`run_ensemble.py` には、LBでも rank 9 -> 10、MAE 0.4414 -> 0.4430 に
悪化したためdropした、というメモが残っている。

このため、MoLFormer系で効いたのは
「pEC50を直接fine-tuneする」方向ではなく、
`log2_fc` でpretrainして凍結embeddingをTabPFNに渡す方向だった。
これは ChemProp で得た
「encoderを直接pEC50に寄せすぎるより、frozen embeddingをTabPFNに渡す方が強い」
という教訓と同じ。

## 相関と役割

MoLFormer-c3 は SMILES transformer なので、ChemProp/KERMTとは別familyだが、
同じ `log2_fc` pretrain を使っているため、予測はかなり相関する。

| comparison target | prediction r vs MoLFormer-c3 | residual r vs MoLFormer-c3 |
|---|---:|---:|
| `tabpfn_cheme_2d_full_boltz_log2fc_pred_optuna_trial10_seed5ens_umap_default` | 0.9330 | 0.8562 |
| `tabpfn_chemprop_pretrain_embed_umap_default` | 0.9282 | 0.8466 |
| `tabpfn_attentivefp_pretrain_embed_umap_default` | 0.9218 | 0.8500 |
| `tabpfn_kermt_pretrain_embed_umap_default` | 0.9177 | 0.8288 |
| `tabpfn_gatedgcn_pretrain_embed_umap_default` | 0.9124 | 0.8302 |
| `tabpfn_pooled_boltz_umap_default` | 0.8997 | 0.8114 |

低相関の独立軸というより、
「主力 `log2_fc` familyに近いが、SMILES transformer由来で少しズレる」
くらいの位置づけ。

## 周辺実験

MoLFormer family では、いくつかの派生形を試している。

| model | 何をしたか | OOF MAE | Spearman | 判断 |
|---|---|---:|---:|---|
| `tabpfn_molformer_c3_pretrain_embed_umap` | `log2_fc` pretrain 768d embed + TabPFN | 0.4752 | 0.7620 | 採用 |
| `tabpfn_molformer_c3_pretrain_embed_umap_default_v3` | TabPFN v3側で再評価 | 0.4800 | 0.7632 | MAEは少し悪化 |
| `tabpfn_molformer_c3_mtr_embed_umap` | 217 RDKit descriptors MTR pretrain | 0.5320 | 0.6943 | 弱く、不採用 |
| `peft_molformer_xl_lora_r32a64_umap_default` | pEC50 direct LoRA fine-tune | 0.5290 | 0.7050 | LB悪化でdrop |
| `lgbm_molformer_xl_umap` | old MoLFormer embedding + LGBM | 0.5878 | 0.6648 | 弱い |

MTR domain adaptation は、`log2_fc` を使わずにRDKit descriptor再構成へ寄せる発想だったが、
このPXRでは `log2_fc` pretrain の方が明確に強かった。

ChemBERTa / BERT-SMILES 系も広く試したが、production member には残らなかった。
[ChemBERTa3](https://github.com/deepforestsci/chemberta3) 自体は
ChemBERTa、MoLFormer、GROVER などを比較するtraining/benchmark frameworkだが、
このPXRでは ChemBERTa系の表現は MoLFormer-c3 / ChemProp / KERMT ほど伸びなかった。

代表例:

| model | 何をしたか | OOF MAE | Spearman | 判断 |
|---|---|---:|---:|---|
| `tabpfn_chemberta_5m_mtr_pretrain_embed_umap_default` | ChemBERTa-5M-MTRをさらに `log2_fc` pretrainしてembed化 | 0.4971 | 0.7318 | BERTa系では良いが採用水準未満 |
| `tabpfn_chemberta_5m_mtr_umap_default` | ChemBERTa-5M-MTR raw embedding | 0.5287 | 0.6914 | 弱い |
| `tabpfn_chemberta_10m_mtr_umap_default` | ChemBERTa-10M-MTR raw embedding | 0.5367 | 0.6910 | 弱い |
| `tabpfn_chemberta_77m_mtr_umap_default` | ChemBERTa-77M-MTR raw embedding | 0.5494 | 0.6767 | 弱い |
| `tabpfn_bert_base_smiles_umap_default` | generic BERT-SMILES embedding | 0.6690 | 0.5558 | かなり弱い |

このため、BERTa系は「試したが、PXRでは主力pretrain-embed axisを超えなかった」
くらいに軽く説明すれば十分。

## 再現性

再現に必要な主要成果物は残っている。

| artifact | status |
|---|---|
| `track1_activity/checkpoints/molformer_c3_pretrain/pretrain.pt` | exists |
| `track1_activity/checkpoints/molformer_c3_pretrain/pretrain_meta.json` | exists |
| `data/molformer_c3_pretrain_embed.parquet` | exists, 4,653 rows x 768 dims |
| `data/molformer_c3_pretrain_log2fc_predictions.parquet` | exists |
| `track1_activity/submissions/tabpfn_molformer_c3_pretrain_embed_umap.csv` | exists |
| `experiment_oof_predictions` | 4,140 rows exists |

`data/molformer_c3_pretrain_embed.parquet` は train/test union の 4,653 compounds をcoverしており、
NaNはない。pretrain自体は 13,136 compounds を使うが、downstream pEC50用の
feature parquet は train/test のみに絞られている点に注意。

## DB 記録

DB の `experiments` に残っている。

| key | value |
|---|---|
| experiment id | 634 |
| model_type | `tabpfn` |
| feature_set | `molformer_c3_pretrain_embed` |
| submission_path | `track1_activity/submissions/tabpfn_molformer_c3_pretrain_embed_umap.csv` |
| OOF rows | 4,140 |

この experiment name は `_default` で終わっていない。
当時の `run_train.py` で TabPFN の 20-trial Optuna default を使っており、
DB hyperparameters には `n_estimators=27`,
`softmax_temperature=0.5910503574798124` が残っている。

OOF summary:

| metric | value |
|---|---:|
| MAE | 0.4752 |
| RAE | 0.5278 |
| Spearman | 0.7620 |

Fold metrics:

| fold | MAE | RAE | R2 | Spearman | Kendall |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.503618 | 0.555588 | 0.599762 | 0.739576 | 0.552090 |
| 1 | 0.447903 | 0.450868 | 0.743142 | 0.811608 | 0.614269 |
| 2 | 0.493814 | 0.581428 | 0.578556 | 0.747886 | 0.556344 |
| 3 | 0.433858 | 0.556487 | 0.598872 | 0.721636 | 0.542517 |
| 4 | 0.496560 | 0.494398 | 0.682756 | 0.789387 | 0.590061 |

## 再現コマンド

pretrain checkpoint を作る:

```bash
pixi run python track1_activity/scripts/run_molformer_c3_pretrain.py
```

768次元 embedding parquet を作る:

```bash
pixi run python track1_activity/scripts/run_molformer_c3_embed_extract.py
```

downstream TabPFN を再実行する:

```bash
pixi run python track1_activity/scripts/run_train.py \
  --model tabpfn \
  --feature molformer_c3_pretrain_embed \
  --split umap \
  --trials 20 \
  --tabpfn-version v2_6
```

同名experimentを厳密に再現するには、当時のTabPFN/Optuna探索条件も合わせる必要がある。
実務上は、既存の `data/molformer_c3_pretrain_embed.parquet` を固定入力として使えば、
downstream検証は再実行しやすい。

## まとめ

MoLFormer-c3 は、単体性能だけを見ると強くない。
ただし、ChemPropやKERMTとは違う SMILES transformer backbone で
`log2_fc` signal を表現し直せるため、ensembleの補助memberとして残った。

説明するときは、
「直接MoLFormerでpEC50を当てるのは弱かった」
「`log2_fc` pretrain + frozen `[CLS]` embedding + TabPFN だと採用水準になった」
「現行weightは小さめだが、SMILES transformer familyの多様性として意味がある」
という整理でよい。

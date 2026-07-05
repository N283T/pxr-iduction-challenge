# Track 1 negative results report

確認日: 2026-05-18 JST

このreportは、Track 1で「試したが最終採用しなかったもの」を説明するための軽い整理。
全部を深掘りするより、なぜ捨てたのかを人に説明できる粒度に留める。
主な根拠は、GitHub issue #100、`track1_activity/scripts/run_ensemble.py` のallow-list/dropコメント、
既存の個別model report、各analysis directoryに残った軽量summary。

結論から言うと、negative resultの大半は次のどちらかだった。

1. そもそも単体精度が足りない。
2. 単体やOOFでは良くても、既存主力memberと相関が高く、ensemble/LBで新しい情報にならない。

加えて後半では、
「OOFは良いがpublic LBで悪い方向へ動く」
という第3パターンも重要になった。

## 1. 採用しなかった時の判断軸

このrepoでは、modelを単体OOFだけで採用しなかった。
主に以下を見ていた。

| check | 見ていたこと | negative判定になりやすい例 |
|---|---|---|
| single-model OOF | MAE / Spearman がproduction候補として足りるか | MAEが0.50前後以上で、既存memberより明らかに弱い |
| ensemble weight | Caruanaでweightが付くか | weight 0、または極小 |
| prediction correlation | 既存memberと違う誤差を出すか | Pearson r が高く、family shareだけ増える |
| public LB transfer | OOF改善がblind analog setへ移るか | OOF改善に反してLB MAEが悪化 |
| leak / proxy safety | 自己近傍、外部judge、pseudo-publicが過剰に効いていないか | self-matchや外部assay差で説明できる改善 |

特に終盤は、OOF MAEで `-0.001` から `-0.004` 程度の改善はかなり弱い証拠として扱った。
id55以降のpublic LB挙動を見ると、validation splitがAnalog Set 1を完全には再現していなかったため。

## 2. そもそも精度が出なかった系

最も単純なnegative resultは、単体modelとして明確に弱かったもの。
これは説明しやすく、「PXR-specificなlow-fidelity signalが入らないと足りなかった」と言える。

| family | 代表例 | OOFの目安 | 判断 |
|---|---|---:|---|
| Morgan / RDKit / classical FP | `lgbm_morgan_r2_2048`, `single_rdkit_desc` | MAE 0.56前後 | baselineとしては有用だが主力には弱い |
| Mordred + Morgan | `optuna_mordred+morgan` | MAE 0.5004 | 2D baselineとしては良いがproduction未満 |
| ChemBERTa raw | `tabpfn_chemberta_5m_mtr_umap_default` など | MAE 0.53-0.55 | generic SMILES representationでは弱い |
| BERT-SMILES | `tabpfn_bert_base_smiles_umap_default` | MAE 0.6690 | かなり弱い |
| direct MoLFormer LoRA | `peft_molformer_xl_lora_r32a64_umap_default` | MAE 0.5290 | direct pEC50 FTは不採用 |
| direct GNN | `attentivefp_optuna_umap`, `gatedgcn_optuna_umap`, `graphgps_optuna_umap` | MAE 0.53-0.57 | 直接pEC50を当てるには弱い |

ここから得た一番大きい教訓は、
「foundation modelをそのままpEC50へfine-tuneすれば勝てる」わけではなかったこと。

今回強かったのは、[overall strategy report](overall_strategy_report.md)でまとめたように、
`log2_fc` でPXR向きに寄せた表現をfreezeし、TabPFNで読む方法だった。

## 3. 高相関で新しい情報にならなかった系

次に多かったのが、単体では悪くないが、既存memberとほぼ同じ方向を向いてしまうケース。
これはOOFでは少し良く見えても、Caruanaでweightが付かなかったり、public LBで悪化したりした。

| family | 何が起きたか | 判断 |
|---|---|---|
| ChemProp派生 | optuna embedding追加、seed追加、FP結合などを多数試した | 主力ChemProp/log2fc軸と相関が高く、ADDでは伸びにくい |
| KAN readout | ChemProp embedding上ではMAE 0.467台まで来た | TabPFN readoutと相関が高く、ensemble axisにはならない |
| direct AttentiveFP / GatedGCN | pretrain embedding版は残ったが、direct版はweight 0 | backbone自体よりrecipeが重要だった |
| re-pooled Boltz trunk | 単体OOFは既存Boltz trunkより改善 | 既存Boltz trunkと r=0.98 前後で、simple swap/dropはLB悪化 |
| Uni-Mol v2 log2fc seed5ens | 単体OOFは少し改善 | min r 0.8994、Caruana ADD weight 0で不採用 |

このカテゴリは「失敗」というより、
すでに強い軸を別の形で再発見していた、という見方が近い。

たとえば ChemProp派生はかなり多く試したが、
最終的には `tabpfn_chemprop_pretrain_embed_umap_default` と
`log2_fc` predicted scalar入りtabular coreが情報をかなり吸収していた。
そのため、追加memberとしては相関が高すぎることが多かった。

Boltz trunkも同様で、re-pooled版は研究としては面白いが、
production poolでは古い `pooled_boltz` / `pooled_boltz_allpairs` を単純に置き換えると悪化した。
詳細は [Boltz trunk report](models/tabpfn_pooled_boltz_trunk_umap.md) に残している。

## 4. OOFは良いがLBに移らなかった系

後半で一番重要だったnegative result。
local OOFだけを見ると良いのに、public LBでは悪化する例が複数あった。

| case | local read | public/LB read | 判断 |
|---|---|---|---|
| `tabpfn_mixed_pool_top500_umap` | OOF MAE 0.4113 と強い | 2回のLB提出で悪化 | 8375d mega-concatのfold構造過適合と判断 |
| ROCS self-match入りtop500 | single OOF 0.4071 / ensemble 0.4103 | LB MAE 0.4243で悪化 | self-match leak由来の見かけ改善 |
| Boltz tier-0 tabular | 9-pool 0.4150 -> 10-pool 0.4130 | LB MAE 0.4149 -> 0.4189 | OOF改善が逆増幅 |
| id56 optuna top500 SWAP | OOF上は強いtop500/log2fc軸 | LB MAE 0.413460まで悪化 | top500方向への寄せすぎ |
| low-weight member drop | OOFは大きく改善 | LBは悪化 | diversity reserveを削りすぎた |

このため、終盤は「OOFで良いから提出」ではなく、
prediction shift、known-bad axis、family share、id55/id56方向との関係を見るようになった。
詳しくは [ensemble calibration report](ensemble_calibration_report.md) に整理している。

## 5. 使い方が難しかった外部・proxy系

外部情報やproxy validationも試したが、Phase 1でそのままhard gateにするには弱かった。

| source | 何を試したか | 結論 |
|---|---|---|
| ChEMBL external judge | filtered PXR activation recordとのnearest-neighbor比較 | qualitative auditには有用だが、assay差とcoverageの問題でsubmission gateには弱い |
| pseudo-public splits | test-likenessや高activity領域を模したholdout | stress testとしては有用だが、新しい確実なfeature familyは出なかった |
| LB-proxy metric battery | id55/id56方向、shift量、family shareなど | 単独の合否判定ではなく、危険方向の検出補助 |
| gate / lift variants | potent46, log2fc, high-activity, SHAP近傍など | id55は良かったが、強くするとすぐ悪化しやすい |

このあたりは「アイデアが悪い」というより、
Phase 1 public LBに対して過学習しやすい領域だった。
Phase 2でAnalog Set 1 labelが見えた後に、validation designとして再利用する価値はある。

## 6. 捨てたが価値は残ったもの

negative resultでも、以下は残す価値があった。

| artifact / family | 残した価値 |
|---|---|
| ChemBERTa / BERT-family audit | generic SMILES encoderがPXRでは弱いという境界線 |
| KANO / Uni-Mol / CLAMP frameworks | 新しいfamilyを試すための再利用可能なpipeline |
| ADMET-AI / kNN pool / 3D shape features | 外部・補助特徴の限界確認 |
| ROCS leak-check | similarity featureでself-matchを避ける安全ルール |
| ChEMBL judge | 外部assayを使う時のcoverage/assay-transfer警告 |
| calibrator variants | Phase 2でlabelが見えた後の再検証候補 |

つまり、採用されなかったmodelも、
「何をこれ以上深追いしないか」を決めるためにはかなり重要だった。

## 7. 説明するときの短いまとめ

このTrack 1では、汎用fingerprint、ChemBERTa、direct GNN、direct MoLFormer fine-tuningなど、
よくある手法も一通り試した。
しかし、多くは単体精度が足りないか、既存の `log2_fc` / ChemProp / top500 axis と高相関で、
ensembleに入れても新しい情報にならなかった。

最終的に残ったのは、
`log2_fc` をPXR向きlow-fidelity signalとして使い、
それをpredicted scalar、frozen embedding、selected tabular featureに変換して、
TabPFNとCaruana ensembleで読む方向だった。

negative resultの価値は、
「何が効かなかったか」ではなく、
「なぜこの最終戦略に収束したか」を説明する補助線として使うのがよい。

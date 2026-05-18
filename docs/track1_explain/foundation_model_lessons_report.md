# Foundation model lessons for Track 1

確認日: 2026-05-18 JST

このreportは、Track 1でfoundation model系をいろいろ試した後の、
やや実務寄りの解釈メモ。
厳密なbenchmark reportというより、
「なぜ有名modelを足しても思ったほど効かなかったのか」を説明するための補助線。

## 1. まず結論

このPXR Track 1では、多くの汎用foundation modelは、
そのまま使っても大きな改善にはならなかった。

理由は大きく3つある。

1. このタスクは PXR induction という固定targetのEC50予測である。
2. 多くのfoundation modelは、複数target・複数assay・汎用分子表現を狙って学習されている。
3. 汎用benchmarkで強いことと、このanalog blind testで強いことは同じではない。

つまり、研究上の「汎用性」と、実務上の「このtargetで当たること」の間にずれがあった。
これは、このchallengeでかなりはっきり見えた点だと思う。

## 2. Target-specific な実務と、generic な研究のずれ

情報系・機械学習研究では、汎用性が重要な価値になる。
たとえば複数target、複数dataset、MoleculeNetのような広いbenchmarkで平均的に良いmodelは、
研究としては強い。

一方で、今回の実務的な目的はかなり狭い。

```text
given molecule
  -> human PXR induction assay
  -> pEC50 on this challenge distribution
```

この場合、汎用的に「分子らしい表現」を持っているだけでは足りない。
むしろ、single-concentration `log2_fc` のように、
PXR inductionそのものに近いlow-fidelity signalを使った方が強かった。

最終的に効いたのは、foundation modelをそのまま信じることではなく、
`log2_fc` でPXR向きに寄せてから、
frozen embeddingやpredicted scalarとしてTabPFNに渡す方法だった。

## 3. Affinity と EC50 induction は同じではない

protein-ligand foundation modelやaffinity modelを使うときに重要なのは、
`affinity != EC50` という点。

PXR inductionのpEC50には、少なくとも以下が混ざる。

| component | PXR pEC50への関係 |
|---|---|
| ligand binding | 必要条件に近いが、十分条件ではない |
| receptor activation / agonism | binding後の機能的応答 |
| cell permeability / exposure | cellular assayの見かけ活性に影響 |
| metabolism / solubility / aggregation | assay readoutを動かす可能性 |
| assay protocol | concentration responseやcounter-assayの影響 |

したがって、汎用affinity modelの出力だけでpEC50を当てるのは難しい。
Boltz affinity scalarも、単体ではpEC50との相関が強いとは言えなかった。
一方で、Boltz trunkの内部表現をpoolして使うと一定の価値が残った。

これは、Boltzが単なる汎用embeddingではなく、
protein-ligand interactionをかなり丁寧にmodel化しているためだと思う。
ただし、それでも最終weightは低く、主役ではなく補助的なstructural reserveだった。

## 4. MoleculeNetで良いことと、このタスクで良いことは別

MoleculeNetや一般的なADMET benchmarkで良いmodelでも、
このchallengeでそのまま強いとは限らなかった。

今回のblind testは、PXR活性compound近傍のanalog setであり、
単なるランダムsplitや広いADMET平均性能とは見ているものが違う。

そのため、汎用fingerprint、ChemBERTa、BERT-SMILES、direct MoLFormer fine-tuning、
direct GNN、GatorAffinityのようなprotein-ligand affinity系の追加軸などは、
期待ほど伸びなかった。
多くはChemProp/log2fc系の強い軸に対して、
ノイズレベルの改善か、高相関な焼き直しに近かった。

これは「それらのmodelが悪い」というより、
このPXR assayで勝つための情報が、
汎用分子表現の平均性能とは別の場所にあった、という解釈が近い。

## 5. 比較的よかったfoundation modelの使い方

foundation modelが完全に無駄だったわけではない。
良かったのは、plug-and-play predictorとしてではなく、
target-specific signalで少し調整してから特徴抽出器として使う方法だった。

| use pattern | 結果 | 解釈 |
|---|---|---|
| raw generic embedding | 弱いことが多い | PXR inductionに十分寄らない |
| direct pEC50 fine-tuning | 不安定 / 弱い | pEC50 labelが少なく、過学習しやすい |
| `log2_fc` pretrain -> frozen embedding | 比較的良い | PXR向き表現を作ってから読む |
| predicted `log2_fc` scalar | 非常に強い | low-fidelity assayを直接activity proxy化できた |
| Boltz trunk pooling | weightは低いが残った | affinity scalarより内部表現の方が使える |

この意味で、foundation modelは「そのまま答えを出すmodel」ではなく、
実務側のtarget-specific signalを受け取る器として使うと価値が出やすかった。

## 6. TabPFN, KAN, LGBM の役割

今回のdownstream modelの役割もかなりはっきりした。

| model | このrepoでの役割 | コメント |
|---|---|---|
| TabPFN | 強い小標本readout | frozen embeddingやselected tabular featureを読むMLP的役割として非常に強い |
| KAN | 代替readout候補 | TabPFNには届かなかったが、意外に良い。TabPFNのlicenseや運用制約がある場合は検討価値あり |
| LGBM | selector / diagnostic | readoutとしてはTabPFNに及ばない場面が多いが、feature gainによるtop-k選択では非常に優秀 |

特にLGBMは、最終予測器として最強ではなくても、
「どのfeatureをTabPFNに渡すか」を決めるselectorとして強かった。
top500戦略はその典型例。

KANはproduction memberにはならなかったが、
TabPFNが使いにくい状況では、ChemProp embeddingなどのreadoutとして再検討する価値がある。
ただし、現時点では既存TabPFN予測と相関が高く、
ensemble上の新しい軸にはなりにくかった。

## 7. 説明用の短い言い方

このchallengeでは、foundation modelをそのまま足してもあまり勝てなかった。
PXR inductionという固定targetでは、汎用分子表現よりも、
PXRに近いlow-fidelity assayである `log2_fc` をどう変換して使うかが重要だった。

Boltzのようにprotein-ligand相互作用を丁寧にmodel化したものは一定の価値があったが、
affinityはEC50 inductionそのものではないため、主役にはならなかった。

実務的には、
「汎用foundation modelを信じる」よりも、
「target-specificな補助signalで寄せて、強いreadoutで読む」
という方針がこのPXRでは強かった。

# Track 1 external data report

確認日: 2026-05-18 JST

このreportは、Track 1で外部データをどう扱ったかを説明するためのもの。
結論として、Phase 1では外部データをそのまま教師データやsubmission gateに使うのは避けた。
少し試した範囲では、coverage、assay差、endpoint差、leakage riskの問題が大きく、
current production modelへ直接混ぜるほどの根拠はなかった。

ただし、外部データが無価値という意味ではない。
Phase 2以降に、validation design、target-specific pretrain、assay補正、類似化合物解析として
使う余地はかなりある。

## 1. まず区別するもの

このrepoで強かった `log2_fc` は、外部データではなくchallenge由来の補助assayである。
ここでは、ChEMBLやToxCast/Tox21、関連nuclear receptor dataなど、
challenge配布データの外側にある情報を外部データとして扱う。

| data | 扱い |
|---|---|
| Track 1 pEC50 train | high-fidelity target |
| single-concentration `log2_fc` | challenge-internal low-fidelity assay |
| ChEMBL PXR activity | external assay data |
| CAR/VDR/FXR/PPARなど関連target | external related-target data |
| ToxCast / Tox21 | external panel data候補 |
| public model pretraining corpus | model側のpretraining data。個別model reportで扱う |

重要なのは、`log2_fc` は同じchallenge文脈のPXR induction signalなので、
外部ChEMBL PXR activityよりもassay compatibilityが高かったこと。

## 2. 外部データをそのまま使うのが難しい理由

外部データを単純に追加教師データとして使いにくかった理由は、主に以下。

| issue | 内容 |
|---|---|
| assay差 | reporter construct、cell line、protocol、測定系がchallengeと違う |
| endpoint差 | pChEMBL / EC50 / AC50 / binding affinity / functional response が混ざる |
| target差 | PXR以外のNR familyはbiologyが近くても同じtargetではない |
| scale差 | ChEMBL pChEMBLとchallenge pEC50の絶対値scaleが合わない |
| coverage不足 | exact overlapを除くとtest近傍のcoverageがかなり薄い |
| leakage risk | test InChIKeyやchallenge compoundと重なる外部rowは慎重に除外が必要 |
| validation不一致 | 外部データで良く見えても、Analog Set 1のpublic LBと一致するとは限らない |

特に今回のtargetはPXR inductionであり、binding affinityだけでは足りない。
この点は [foundation model lessons report](foundation_model_lessons_report.md) の
`affinity != EC50` と同じ問題である。

## 3. ChEMBL PXR activationを試した結果

ChEMBL PXR activation EC50 rowをfilterし、challenge train/testとexact InChIKey overlapするものを除外した上で、
nearest-neighbor signalとして使えるかを見た。

主な成果物:

- `track1_activity/analysis/chembl_pxr_probe/chembl_pxr_activation_probe.py`
- `track1_activity/analysis/chembl_pxr_probe/chembl_external_judge.py`
- `track1_activity/analysis/chembl_pxr_probe/outputs/external_judge/external_judge_report.md`

外部judge側のcoverage:

| item | value |
|---|---:|
| raw filtered ChEMBL PXR activation molecules | 267 |
| exact challenge overlaps removed | 12 |
| external molecules after exclusion | 255 |
| test compounds with NN Tanimoto >= 0.25 | 121 |
| test compounds with NN Tanimoto >= 0.30 | 27 |
| test compounds with NN Tanimoto >= 0.35 | 4 |
| test compounds with NN Tanimoto >= 0.40 | 0 |
| test NN max / median | 0.3939 / 0.2222 |

train側でも、external nearest-neighbor pChEMBLとchallenge pEC50の相関は弱かった。

| threshold | n train | Spearman pEC50 vs external | Pearson pEC50 vs external |
|---:|---:|---:|---:|
| 0.25 | 790 | 0.1559 | 0.0705 |
| 0.30 | 161 | 0.2391 | 0.1215 |
| 0.35 | 55 | 0.1400 | -0.0204 |
| 0.40 | 36 | 0.0758 | -0.2102 |

このため、ChEMBL PXR activityは
「近傍に外部PXR active compoundがあるかを見るqualitative audit」には使えるが、
Phase 1のsubmission gateや教師データとしては弱いと判断した。

## 4. ChEMBL judgeがid58を止められなかった

外部judgeを実際のcandidate判定に使えるかも確認した。
しかし、test NN Tanimoto >= 0.30では27 compoundsしかなく、
id58のLB-negativeな動きを止めるsignalにはならなかった。

ChEMBL近傍compoundは外部pChEMBLが高めで、
candidateが少し上方向に動くとむしろ良く見えやすかった。
centeredに見ると差はほとんどなく、hard gateとして使うには弱い。

この結果から、外部judgeは「warning light」にはなるが、
提出判断を置き換えるものではないと整理した。

## 5. 関連target dataは候補だが、Phase 1では深追いしなかった

CAR, VDR, FXR, PPAR, GR, AR, AHRなどの関連targetについても、
ChEMBLからnearest-neighbor signalをざっと確認した。

代表的なscan結果:

| target | n external mols | test NN >= 0.30 | test NN >= 0.40 | simple OOF MAE | Spearman |
|---|---:|---:|---:|---:|---:|
| CAR_NR1I3 | 189 | 26 | 0 | 0.8291 | 0.2155 |
| PXR_NR1I2 | 950 | 110 | 2 | 0.8370 | 0.2307 |
| FXR_NR1H4 | 3187 | 108 | 8 | 0.8541 | 0.1297 |
| VDR_NR1I1 | 562 | 10 | 0 | 0.8738 | 0.0878 |
| PPAR_gamma | 4317 | 146 | 10 | 0.8566 | 0.1216 |

このnearest-neighbor feature自体はかなり弱く、直接使う価値は低かった。
ただし、関連target dataは「類似化合物の値をそのまま入れる」よりも、
target-specific pretrainやmulti-task representation learningとして使う方が可能性がある。

実際、`docs/superpowers/specs/2026-05-01-cross-nr-multitask-plan.md` では、
CAR/VDR/FXRなどを使ったcross-NR multi-task pretrain案を検討していた。
ただし、Phase 1終盤ではOOF/LB mismatchが大きくなっており、
新しい外部データ軸を急いでsubmissionまで持っていくのはriskが高いと判断した。

## 6. pseudo-public validationとしては有望

外部データを教師として使うより、validation designに使う方向はかなり有望だった。

ChEMBL judgeが弱いと分かった後、
test-likeness、potent46近傍、predicted log2fc、ChEMBL coverageなどを使って、
pseudo-public holdoutを作る方向を試した。

代表的なpseudo-public split:

| split | n | 特徴 |
|---|---:|---|
| `public_adv_top513` | 513 | train/test adversarial classifierでtest-likeなtrainを選ぶ |
| `public_hybrid_nolabel_top513` | 513 | test近傍、adversarial、log2fc、potent近傍を混ぜる |
| `public_hybrid_with_y_top513` | 513 | pEC50も使うlabel-aware stress split |
| `public_testnn_top513` | 513 | test NN similarity重視 |
| `public_log2fc_top513` | 513 | high log2fc / high activity領域stress |
| `public_chembl_ext_nn_ge025` | 790 | ChEMBL外部近傍coverageを使うstress split |

pseudo-public retrain batteryでは、新しいfeature familyが見つかったというより、
やはり predicted `log2_fc` axis が強いことを再確認した。
一方で、high-activity/test-like splitではunderprediction biasが見え、
後半のcalibration/gate検討につながった。

したがって、外部データはPhase 1では
「training data」より「validation stress test」として使う方が安全だった。

## 7. Phase 1でやめた理由

Phase 1で外部データをproductionに入れなかった理由は、以下の整理で説明できる。

| path | 判断 |
|---|---|
| 外部PXR activityをそのまま追加教師データにする | assay/scale差が大きく、challenge pEC50と合いにくい |
| 外部nearest-neighbor featureを入れる | OOFが弱く、coverageも薄い |
| 外部judgeでcandidateを選ぶ | id58 regressionを止められず、hard gateには弱い |
| cross-NR multi-task pretrain | 可能性はあるが、Phase 1終盤に急ぐにはriskが高い |
| pseudo-public validation | 有望。ただし提出判断の補助であり、単独metricにはしない |

つまり、外部データは「少し試したがそのまま使うのはだめだった」。
ただし、これは外部データが不要という意味ではなく、
使うなら、assay補正・target-specific抽出・validation設計を丁寧にやる必要がある。

## 8. Phase 2以降で試す価値があること

Phase 2でAnalog Set 1 labelが見えると、外部データの使い方はかなり変わる。
以下は試す価値がある。

| idea | 内容 | Phase 2でやる理由 |
|---|---|---|
| assay補正 | ChEMBL pChEMBLをchallenge pEC50 scaleへ校正する | Analog Set 1 labelで外部scale差を見られる |
| target-specific pretrain | PXR/CAR/VDR/FXRなどでencoderをpretrainし、frozen embedding化 | direct NN featureよりrepresentationの方が期待できる |
| similar compound augmentation | test近傍の外部類似化合物をsoft labelやlocal priorとして使う | 類似化合物追加はleakとbias管理が必要なのでlabel後が安全 |
| pseudo-public validation | 外部coverageをstratumとしてvalidation foldを作る | id55/id56のようなOOF/LB mismatch対策 |
| external qualitative audit | 著名薬、known PXR ligand、外部active近傍のcase study | model説明やerror analysisに使える |

特に、外部データは「教師データとして足す」よりも、
「どの領域のpredictionを信用するか」を判断する補助として使う方が現実的だった。

assay補正の設計としては、Boltz affinity headの学習方法が参考になるかもしれない。
この点はBoltz-2 paperのparse済みメモでも確認した。

参照:

- raw parse: `docs/papers/boltz2_raw/boltz2_full.txt`
- notes: `docs/papers/boltz2_affinity_notes.md`

Boltz-2は、PubChem、ChEMBL、BindingDBなど由来の
Ki, Kd, IC50, AC50, EC50, XC50 を `log10(μM)` に統一しつつ、
低品質assay、低分散assay、PAINS、heavy atom数の大きいligandなどをfilterしている。
さらに、単一protein targetのbiochemical / functional assayに絞り、
hit discovery用のbinary taskと、lead optimization用のcontinuous affinity taskを分けている。

特に重要なのは、continuous affinity valueを単純な絶対値回帰だけで学習していない点。
Boltz-2では同一assay内のpairwise差分lossを強く使い、
assay-specific offsetやCheng-Prusoff補正不能な条件差をある程度打ち消す設計になっている。
論文でも、KiやIC50などは厳密には異なる量だが、
同一assay内の差分を見ることで補正項が相殺される、という考え方が説明されている。
この発想は、外部ChEMBL PXR activityをchallenge pEC50へ使うときにも重要である。

つまり、外部assayをそのまま追加するのではなく、
assay source、endpoint、measurement type、target construct、cellular/bindingの違いを
contextとして持たせたうえで、challenge assayへtransferするmodelを作る方が自然。
Phase 1ではここまで作り込む時間とlabel anchorが足りなかったが、
Phase 2以降なら、Analog Set labelを使って
「外部assayをどうchallenge scaleへ写像するか」を検証できる。

なお、Boltz-2 paper自体も、affinity performanceはassayごとに大きくばらつくと述べている。
したがって、Boltz型の学習設計を参考にする場合でも、
外部assayを混ぜれば自動的に解決するわけではなく、
assayごとの信頼度、差分学習、challenge assayへのcalibrationを明示的に設計する必要がある。

## 9. 説明用の短い言い方

外部データは少し試したが、Phase 1ではそのまま使わなかった。
ChEMBL PXR activityはcoverageが薄く、challenge pEC50との相関も弱く、
id58のようなLB-negative candidateを止めるjudgeにもならなかった。

一方で、外部データはPhase 2以降に価値がある。
単純に類似化合物の値を足すのではなく、
assay補正、関連targetでのpretrain、test-like validation split、外部active近傍のcase analysisとして使う方がよい。

今回のPhase 1では、外部データを急いで混ぜるより、
challenge-internalな `log2_fc` signalと、public LBで確認済みのanchor近傍を重視する判断をした。

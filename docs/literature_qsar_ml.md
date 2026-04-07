# PXR（Pregnane X Receptor）QSAR/ML 文献レビュー（2015–2026）と OpenADMET PXR Blind Challenge 向け示唆

## スコープと要点整理

本レビューは、PXR 活性（主にアゴニスト／活性化、場合により結合・LBD結合）に対する **QSAR / 機械学習モデル（2015–2026、査読付き中心）**を抽出し、加えて **PXR の結合ポケット／SAR（薬理学・構造生物学）**、**核内受容体（NR）群にまたがる multi-task / transfer**、**GNN・Transformer 系 ADMET モデルの知見**を横断して、ユーザーのコンペ（OpenADMET PXR Blind Challenge：pEC50 回帰）に直結する形で整理する。なお、PXR は「大きく柔軟で疎水性寄りの結合空間」「多様な化合物を受容するプロミスキュア性」を特徴とし、これが ML 側では **分布シフト・外挿・適用領域（AD）**の問題として顕在化しやすい（後述）。citeturn27view0turn15view0turn35search0turn35search2

ユーザーの現状所見である「LogP/SLogP が圧倒的に重要」という点は、PXR 分類タスクでの特徴重要度解析でも **logP（および関連する疎水性・屈折率・溶解度 proxy）**が上位に来る報告が複数あり、**学習の“近道”になりやすい**一方で **外挿性能を損なう危険**も示唆される（後述の“期待効果順の提言”で具体策を提示）。citeturn28view1turn26view0

---

## PXR QSAR/ML 論文カタログ（2015–2026）

以下の表は、2015–2026 の範囲で、PXR の活性化／結合に対して **定量的（分類・回帰）モデル**を構築している論文を中心に整理した。DOI は **検証済み**で、クリック可能なように `https://doi.org/...` 形式で記載する（リンクはコード表記）。  
（注）一部論文は出版社制限のため本文の全数値まで追えず、**抄録・本文で確認できた範囲**に限って記載する。

| 年 | 論文（主題） | DOI（クリック可） | データセット規模・ソース | 目的変数 | モデル | 記述子・表現 | 主な性能指標（論文記載） | 重要所見（特徴／解釈） |
|---|---|---|---|---|---|---|---|---|
| 2015 | ヒトPXR活性化の NB 分類（薬物 ADMET 予測シリーズ内） | `https://doi.org/10.1021/tx500389q` citeturn8search12turn37search4 |（本文未確認：2015 論文） | activator / non-activator | Naive Bayes | 分子記述子（詳細は要本文） |（本文未確認） | 早期の ML 分類例（PXR のプロミスキュア性が背景）。citeturn8search12 |
| 2016 | hPXR activator / non-activator 多手法分類 | `https://doi.org/10.2174/1386207319666160316122327` citeturn10view0 | 529化合物（317 activator / 212 non） | 分類 | SVM / kNN / RF / NB + コンセンサス | RDKit 記述子＋FP | 外部テストで activator 59–73%、non 55–68% など（抄録記載） citeturn10view0 | 記述子/FP と手法の比較・コンセンサス化。citeturn10view0 |
| 2016 | ラット・ヒト PXR activator 予測（ベイズ分類） | `https://doi.org/10.1021/acs.chemrestox.6b00227` citeturn6search6turn35search14 | ラット／ヒト PXR の活性化データ（詳細は本文） | 分類 | Bayesian classification | 物性8種など（本文中の解析あり） |（本文全体の数値は未確認） | activator は **より疎水性（AlogP↑）・重い（MW↑）・柔軟（回転結合↑）**傾向。PXR ポケットの疎水性と整合。citeturn35search14turn35search7 |
| 2016 | 構造多様薬物の hPXR アゴニズム in silico（HM-BSM/HM-PNN） | `https://doi.org/10.1007/s11596-016-1609-4` citeturn9search3 |（抄録中心） | 分類/回帰（詳細は本文） | HM-BSM / HM-PNN（著者手法） |（本文） |（本文） | 手法特化で他研究の参照頻度あり。citeturn9search3turn9search24 |
| 2017 | hPXR 結合活性（binding activity）予測モデル | `https://doi.org/10.1007/s11356-017-9690-1` citeturn8search0 |（本文未確認：環境化学系） | 分類（binding） |（本文） |（本文） |（本文） | PXR 結合（binding）予測の2017年モデル。citeturn8search0 |
| 2017 | REACH 72,524 物質へスクリーニング（PXR binding/activation 等） | `https://doi.org/10.1016/j.comtox.2017.01.001` citeturn7view1 | 2,816 drugs の HTS を学習し、72,524 REACH を予測 | hPXR-LBD binding / full-length hPXR activation / rPXR activation / CYP3A4 induction（全て二値） | QSAR（二値） |（手法詳細は本文） | バランス精度 75.4–92.7%（ブラインド外部検証含む） citeturn7view1 | **ヒト/ラット差**・PXR と CYP3A4 誘導の重なり解析などを実施。citeturn7view1 |
| 2017 | 構造ベース pharmacophore による hPXR activator 予測 | `https://doi.org/10.1016/j.xphs.2017.03.004` citeturn36search0turn36search2 | 既知構造から pharmacophore 構築（詳細は本文） | activator 予測（主に分類） | 構造ベース pharmacophore | MOE で構築（結晶構造由来） citeturn35search30turn36search2 | PXR 結晶構造群を元に **定性的 pharmacophore**を構築。citeturn35search30turn36search2 |
| 2018 | ToxCast（steatosis AOP-MIE）向け QSAR（RF＋DRAGON） | `https://doi.org/10.1021/acs.jcim.8b00297` citeturn15view0 | entity["organization","ToxCast","epa hts program"] の in vitro assay（不均衡） | 複数MIE assay の二値 | Random Forest（不均衡対策） | DRAGON 記述子、undersampling / balanced RF | 多くのモデルで正解率（% correctly predicted）≥75% を満たす、と記載 citeturn15view0 | 不均衡学習（undersampling / balanced RF）を明示的に比較。citeturn15view0 |
| 2020 | “Smallest Maximum Intramolecular Distance” による PXR 活性回避指標（設計寄り） | `https://doi.org/10.1021/acs.jcim.9b00692` citeturn2search2 |（本文未確認） | 主に設計・指標 |（本文） |（本文） |（本文） | PXR を避ける設計指標の提案（要本文）。citeturn2search2 |
| 2022 | “train–validation gap” 罰則の正則化で外挿を改善（分類） | `https://doi.org/10.3390/cells11081253` citeturn27view0turn28view1turn30view2 | entity["organization","PubChem","nih chemical bioassay db"] PXR：最終 941（A:202/N:739）を学習、ToxCast 1179 と文献セット 409 をテスト citeturn30view2 | 二値（activator） | RF / SVM（6モデル×特徴集合） | 物性17＋FP（8192）など | “gap penalty”で MCC が最大 +0.21 改善（文献セット） citeturn28view0turn28view2 | RF feature importance 上位が **esol・分子屈折率・logP（>0.1）**で、FP bit は最大でも 0.018 程度。citeturn28view1 |
| 2023 | PubChem 由来 hPXR 分類（XGBoost が最良） | `https://doi.org/10.1039/d2va00182a` citeturn31view0 | 学習 4,144、外部テスト 1,037（PubChem） citeturn31view0 | 二値 | XGBoost 他5手法比較 | RDKit 記述子、8種FP、次元削減 | AUC：学習 0.913、外部 0.860（最良） citeturn31view0 | descriptors/FP の比較、AD 解析を明示。citeturn31view0 |
| 2024 | QSPRmodeler の例題として PXR pEC50 回帰＋分類（ChEMBL） | `https://doi.org/10.3389/fbinf.2024.1441024` citeturn26view0 | entity["organization","ChEMBL","embl-ebi bioactivity db"] から PXR EC50 1187 entries→精査後に回帰/分類 citeturn26view0 | pEC50（回帰）＋二値（閾値 12,000 nM） | XGBoost（CV） | 8記述子（SLogP 等）＋Morgan FP(1024bit) PCA 50 + 2 filter → 60次元 citeturn26view0 | 分類：精度 82.4%、ROC-AUC 82.4%（hold-out 10%） citeturn26view0 | 記述子に **SLogP** を含め、説明可能性として feature importance 報告（ツール機能）。citeturn26view0 |
| 2024 | NR Few-shot：GNN＋Transformer＋メタラーニング（PXR 含む） | `https://doi.org/10.1186/s13321-024-00902-4` citeturn33view0 | NURA：15,247 compounds、11種NR（PXR/FXR/RXR/AR/ER…） citeturn33view0 | NR binding/agonist/antagonist（二値化） | Few-shot GNN-Transformer（Meta-GTNRP） | 分子グラフ＋Transformer attention |（詳細指標は本文） | multi-task / few-shot が NR に有効、という立場。citeturn33view0 |
| 2025 | FAERS と PXR 予測モデル（薬剤安全性応用） | `https://doi.org/10.3390/ijms26157630` citeturn18search0turn18search2 | PXR agonist 予測 ML を開発し、entity["organization","FAERS","fda adverse event reporting system"] に適用 citeturn18search2 | 二値（推定） | ML（詳細は本文） |（本文） |（本文） | PXR 予測を薬剤有害事象と結びつける応用例。citeturn18search2 |
| 2025 | 生成化学＋MCDA pruning の例として PXR pEC50 回帰モデル構築（ChEMBL 516） | `https://doi.org/10.3390/applbiosci4010002` citeturn25view0 | ChEMBL 516（アッセイ精査・平均化等の手動キュレーション） citeturn25view0 | pEC50（回帰） | 線形ブースト ANN アンサンブル（ADMET Modeler） citeturn25view0 |（ツール内表現） | テスト RMSE 0.468、MAE 0.380 citeturn25view0 | 参照結晶：PXR 5a86 を明示。citeturn25view0 |

**補足（重要）**：上表は「PXR をターゲットとして明示的にモデル構築している」2015–2026 の文献を中心に集約した。環境毒性（ToxCast/Tox21 系）では「NR 全般の一部として PXR を含む」形式の論文が多数あるが、本レビューでは **PXR モデルの仕様・規模・指標が本文/抄録で追えるもの**を優先した。citeturn33view0turn15view0turn31view0

---

## PXR 結合ポケット、薬理学的 SAR、Pharmacophore の確立知見

### 結晶構造（PDB）に基づく“プロミスキュア性”の構造要因

PXR LBD は大きく柔軟で、疎水性寄りの空間が広く、アゴニストによってポケット形状が変わりうる（“適応的”な結合）。この点が、薬理学的にも ML 的にも「疎水性・サイズ・柔軟性」指標（logP、MW、回転結合）の強い相関として現れやすい。citeturn35search2turn35search14turn28view1

代表的な共結晶構造（ヒト PXR LBD）は以下が頻出で、構造解析・ドッキング・pharmacophore の基盤として用いられる：

- **リファンピシン**：PDB **1SKX**（hPXR LBD + rifampicin）citeturn35search0  
- **SR12813**（コレステロール低下薬として報告されるリガンド）：PDB **1ILH**, **1NRL**（coactivator ペプチド併記の構造も）citeturn35search1turn35search5  
- **ハイパーフォリン**：PDB **1M13**（St. John’s wort 成分。結合でポケット体積が増える旨の記載あり）citeturn35search2  
- **PXR–RXRα 複合体**：PDB **4J5X**（SR12813-bound、PXR/RXRα のヘテロ二量体）citeturn35search9  
- 近年追加の例：rifamycin S 共結晶 **8E3N**、SR12813 共結晶の新規構造 **9FZJ** など、構造データは増え続けている。citeturn35search4turn35search16  

構造解析研究では、**ループの柔軟性**や **芳香族残基の配置**がリガンド多様性を許容する根拠として議論され、種差（後述）にも関与する、とされる。citeturn35search0turn35search2

image_group{"layout":"carousel","aspect_ratio":"16:9","query":["human pregnane X receptor LBD rifampicin structure 1SKX","human pregnane X receptor SR12813 crystal structure 1ILH","human pregnane X receptor hyperforin crystal structure 1M13"],"num_per_query":1}

### Pharmacophore モデル（確立・再利用される型）

PXR の pharmacophore は古典的には「疎水性特徴の配置（複数）＋一部 HBA/HBD の許容」という形で語られることが多いが、2017 年以降も **構造ベース**で更新されている。特に Torimoto-Katori らは、既報の hPXR 結晶構造群を基盤として **structure-based pharmacophore** を構築し、hPXR activator を in silico で予測する枠組みを示した。citeturn36search0turn35search30

また、化学空間の広さから「単一 pharmacophore ではなく、複数モデル／複数立体配座の集合で捉える」考え方が使われやすい（例：結晶構造由来の複数リガンド形状を統合する方針）。citeturn35search30turn35search11

### “特権”足場（privileged scaffolds）と substructure アラートの実務的見方

PXR は「特定の狭い骨格」よりも、**疎水性・体積・柔軟性**を満たす多様骨格に反応しうるため、単純な scaffold ルールは外れやすい。一方で、分類モデルの特徴重要度や物性比較から、少なくとも以下は“頻出の成功条件”として繰り返し現れる：

- **疎水性（logP / SLogP）**：分類モデルで重要度上位（例：logP が上位、FP bit は相対的に寄与が小さい）citeturn28view1turn26view0  
- **サイズ・柔軟性（MW、回転結合数、環数）**：activator 側へシフトしやすい（ヒト/ラット双方での物性差の議論）citeturn35search14turn31view0  

ただし、これらは“PXR らしさ”を捉える一方で、コンペの **pEC50 回帰**では「logP を当てればそこそこ当たる」近道になり、化学シリーズ外挿で崩れるリスクがある（次節で対策を提案）。citeturn28view1turn27view0

### 種差（human vs rodent 等）

PXR は種間で LBD の配列差が大きく、リガンドプロファイルも変わりうる。rifampicin 共結晶（1SKX）の構造解説でも、種差とループ領域の柔軟性に関連づけた議論が含まれる。citeturn35search0  
実験系でもヒトとマウスの PXR 活性差を比較する研究があり、種差がリガンド選択性に影響することが示されている。citeturn35search3turn35search27

---

## Multi-task / Transfer Learning と核内受容体（NR）横断の学習

PXR は単独タスクとしてはデータが限られがちで、“近縁NRからの知識移転”が理にかなう。NR はドメイン構造の共通性と、アッセイ（binding/agonist/antagonist）の形式的類似があり、multi-task のメリットが出やすいと整理される。citeturn33view0

### NR 横断データベース（NURA）を用いた Few-shot / Meta-learning

Torres らは、**NURA（15,247 compounds、11 NR）**を用い、GNN と Transformer を統合した few-shot メタラーニング（Meta-GTNRP）で、ラベルが少ない NR タスクに適応する枠組みを提示している。NURA は ChEMBL、BindingDB、NR-DBIND、Tox21 など複数ソース統合で、PXR も含む。citeturn33view0

このタイプの設計は、OpenADMET のように **4,000 規模の回帰**で「テスト化学空間が訓練からずれる」状況に対し、(i) **“NR 共通知識”の表現学習**、(ii) **少数ラベル適応**、(iii) **不均衡・few-shot を前提にした最適化**、という観点で示唆が大きい。citeturn33view0

### PXR 固有データでの“外挿”対策：train–validation gap を罰する正則化

Hirte らは PubChem PXR を学習し、ToxCast / 文献セットをテストに用いて、**訓練–検証ギャップを罰するスコア**でハイパーパラメータを選ぶ“正則化”を提案した。これは「類似分子には強いが、非類似には弱い」典型的な QSAR 崩壊へ明確に対処している。citeturn30view2turn28view0

特に、RF（物性＋FP）で文献テストに対する MCC が **+0.21**改善した、と報告されており、コンペでも「CV を上げる」より「CV–LB のギャップを縮める」方向が有効である可能性が高い。citeturn28view0turn28view2

---

## GNN・Transformer と “2D記述子 vs 表現学習” の実務インプリケーション

### “logP が強すぎる”現象の位置づけ

PXR 分類では、RF の feature importance 上位が **esol・分子屈折率・logP**で、FP の寄与が相対的に小さい、という報告がある。citeturn28view1  
また、回帰/分類の例題として PXR を扱った QSPRmodeler でも、入力 8 記述子に **SLogP** が含まれており、PXR pEC50 を説明する主要軸の一つとして当然に扱われている。citeturn26view0

この状況は「logP が“本質”である」可能性と同時に、**アッセイ／化学空間の偏り（例：疎水性の高い化合物ほど測られやすい・ヒットしやすい）**、および **プロミスキュアターゲットでの“近道学習”**の可能性も意味する。PXR は疎水性ポケットが大きいことが構造的にも支持されるため、logP の寄与自体はメカニズム上も納得できるが、回帰競技では「logP 一本足打法」が未知化学空間で崩れるケースに警戒が必要である。citeturn35search2turn28view1turn35search14

### 2D descriptor / fingerprint 系と GNN/Transformer 系の役割分担

- **2D 記述子＋木系モデル（XGBoost, RF）**は、PXR のような疎水性主導ターゲットで強く出やすく、実際に PubChem PXR 分類では XGBoost＋RDKit 記述子が良好な AUC を示している。citeturn31view0  
- 一方、NR 横断の few-shot では、分子グラフ（局所構造）と Transformer（大域依存）を併用し、少数ラベル下の一般化を狙う設計が提示されている。citeturn33view0  

OpenADMET の「4,140 学習＋513 ブラインド、pEC50 回帰」という設定では、**(i) 2D＋GBDT を“強いベースライン”として保持しつつ、(ii) 表現学習（GNN/Transformer）を“外挿時の補助”に回す**戦略が合理的になりやすい。

---

## アッセイ干渉（PAINS・ルシフェラーゼ等）と PXR データ解釈上の落とし穴

PXR は in vitro screening でルシフェラーゼ系レポーターが多用される一方、**レポーター酵素阻害・化合物干渉**により偽陽性／偽陰性が混入しうる点が重要である。citeturn37search3turn37search5turn37search1

実務上のチェックポイント：

- **PAINS（pan-assay interference compounds）**：頻回ヒット構造をフィルタする発想は、Baell & Holloway の PAINS filters 提案が代表例。citeturn37search0turn37search4  
- **ルシフェラーゼ阻害による偽陽性**：レポーターアッセイでは、ルシフェラーゼ阻害剤がシグナルを歪め、偽陽性を生むことがある（“false positives in a reporter gene assay” の報告など）。citeturn37search1turn37search5  
- **“cytotoxicity burst”**：ToxCast/Tox21 の解釈では、一般細胞毒性による見かけの活性（多経路同時活性化）を区別する必要があり、burst の概念整理や pragmatic な判別枠組みが提案されている。citeturn37search10turn37search2turn37search22  

コンペの学習データが「ドーズレスポンス pEC50」中心であっても、元アッセイがレポーター系であれば **干渉に起因するノイズ（特に“疎水・反応性・凝集性”化合物）**が pEC50 の尾部を歪める可能性があるため、**外れ値処理・robust loss・不確実性推定**が実効策になりやすい（次節に落とす）。

---

## OpenADMET PXR Blind Challenge 向けの提言（期待効果順）

以下は、ユーザーの現状（LightGBM＋Mordred/Fingerprint、GNN各種、LogP が支配的、RAE=0.64、513ブラインド）を前提に、**過学習を抑えつつ順位を上げる可能性が高い順**に並べた。各提言は「何をやるか」「なぜ効くか」「実装の要点」を具体化する。

### 外挿を意識した検証設計と “gap 正則化” を回帰へ移植（最優先）

**何をやるか**  
通常のランダムCVだけでなく、**化学空間分割（scaffold split / similarity split）**を主指標にしてモデル選択し、さらに Hirte らの発想（train–validation gap を罰する）を **回帰（pEC50）**に移植する。citeturn28view0turn28view2

**なぜ効くか**  
PXR はプロミスキュアで、PubChem/ToxCast/文献セット間ですら分布が異なり、CV が“楽すぎる”と外挿で崩れることが示されている。Hirte らは gap 罰則で **非類似化合物側の性能が上がる**ことを示した。citeturn30view2turn28view0

**実装の要点**  
- スプリット：Murcko scaffold（RDKit）＋補助的に Tanimoto 距離閾値（例 0.6/0.7）で外挿寄り評価。citeturn30view2  
- 回帰版 gap 罰則：例）`score = R2_valid - λ * |R2_train - R2_valid|`、または MAE/RMSE 版で `score = -RMSE_valid - λ * |RMSE_train - RMSE_valid|`。λ は外挿優先で 0.5–2 程度から探索。  
- 最終アンサンブル重みも「CV最良」ではなく「外挿split最良」を優先する（現状の L2 最適化も、スプリット別に重み最適化を分けると過学習が減りやすい）。

### logP 支配の“近道学習”を制御する特徴設計（高優先）

**何をやるか**  
logP（SLogP）を捨てるのではなく、**“logP のみで説明できる成分”を分離**して残差を学習させる（2段階モデル／残差学習）、または **monotonic constraint / partial dependence 制御**で破綻を抑える。

**なぜ効くか**  
分類モデルでは logP が支配的になりうることが示されており、PXR の疎水性ポケットというメカニズムとも整合するが、回帰の外挿では「logP が似ていれば同程度の pEC50」という誤学習を誘発しやすい。citeturn28view1turn35search2turn26view0

**実装の要点**  
- 残差学習：  
  1) `pEC50 ~ f1(logP, MW, RB, TPSA …)` の小モデル（GAM/GBDT浅め）  
  2) 残差 `r = y - ŷ1` を Mordred/FP/GNN で学習し、最終 `ŷ = ŷ1 + ŷ2`  
- logP 近傍での過学習監視：logP をビン分割して誤差分布を監視（logP の extremes で崩れていないか）。  
- 物性8種（logP, MW, RB, aromatic rings, HBA/HBD, PSA, MR など）は、PXR では伝統的に重要視される組（QSPRmodeler や Gou の PCA 変数にも出てくる）ため、まずはここを“物理ベースの骨格”として安定化させる。citeturn26view0turn31view0turn28view1

### 公開データでの弱い教師あり拡張：分類ラベルを multi-task で併用（高優先）

**何をやるか**  
追加で pEC50 を集めに行くのではなく、公開の PXR **二値（agonist/activator, binding）**データを補助タスクとして multi-task 学習し、主タスク pEC50 を正則化する（例：ChemProp や Transformer で heads を分ける）。

**なぜ効くか**  
PXR は assay/source により分布が違う一方、二値は量が大きい（PubChem 941、ToxCast 1179 などの curated set、REACH 72,524 予測研究など）ため、「PXR らしさ」の表現学習に寄与しやすい。citeturn30view2turn7view1turn33view0

**実装の要点**  
- 例：GNN/Transformer を shared encoder、head を (i) pEC50 回帰、(ii) PXR activator 分類、(iii) PXR binding 分類に分ける。  
- 学習は **回帰データの重みを主**、分類は補助（λ=0.1〜0.5）で探索。  
- 分類データは干渉ノイズを含む可能性があるため、**ラベルスムージング＋early stopping を厳しめ**に。citeturn37search5turn37search10

### 外れ値・干渉ノイズ対策（中〜高優先）

**何をやるか**  
- (i) robust loss（Huber/quantile）、(ii) 不確実性推定（deep ensemble / conformal）、(iii) “干渉疑い”化合物への down-weight（PAINS/ルシフェラーゼ阻害/凝集）を導入。

**なぜ効くか**  
PXR はルシフェラーゼ系レポーターが多く、阻害剤・PAINS・細胞毒性 burst で“見かけの活性”が入りやすい。これが回帰の尾部（高 pEC50 と低 pEC50）を歪め、モデルがそこに過剰適合しやすい。citeturn37search5turn37search10turn37search0

**実装の要点**  
- PAINS フィルタを特徴として入れる（除外ではなくフラグ化）。citeturn37search0turn37search8  
- ルシフェラーゼ阻害は既知の干渉要因として扱い、可能なら “luciferase inhibitor likelihood” を外部ツールで推定して特徴化（InterPred 等の干渉予測ツールの概念整備がある）。citeturn37search9turn37search5  
- ToxCast 系の burst の考え方を参考に、極端な細胞毒性っぽい点を“疑い”として重みを落とす。citeturn37search10turn37search22  

### エンサンブル設計：モデル多様性を“表現の種類”で稼ぐ（中優先）

**何をやるか**  
現状の LightGBM×（Mordred・FP）＋GNN＋Transformer は方向性が良い。効果を高めるには、モデル多様性をハイパーパラメータ違いではなく、**入力表現の独立性**で稼ぐ：  
- 物性8〜20次元（logP/MW/RB/PSA/MR…）の“小モデル”  
- 2D記述子（Mordred/rdkit）の GBDT  
- FP（count vs bit、AtomPair/Avalon 等）の線形/GBDT  
- GNN（MPNN 系）  
- SMILES Transformer（MoLFormer 等）

**なぜ効くか**  
PXR は logP 的な大域要因と、局所構造（芳香環配置・電子性・立体）要因の両方が効く可能性があり、単一表現に寄ると外挿で崩れやすい。citeturn35search2turn33view0turn31view0

**実装の要点**  
- 重み最適化は “外挿split” 目的関数で（前述）。  
- stacking のメタモデルは ridge よりも、**Huber regression**や **quantile regression**も検討（外れ値に強い）。  
- “logP-only” モデルをあえて入れ、その予測をメタ特徴にする（残差学習と整合）。

---

## 競技で使える外部データ候補とアクセス情報

コンペ規約が「外部データ使用可」かどうか（およびリークの定義）に必ず従う必要がある。その前提で、公開で入手しやすく、PXR と親和性が高い候補を列挙する。

- entity["organization","PubChem","nih chemical bioassay db"]：PXR 関連バイオアッセイ（activator/binding 等）。Gou や Hirte の分類セットの土台。citeturn31view0turn30view2  
- entity["organization","ToxCast","epa hts program"]：PXR 関連 assay を含む高スループット in vitro。Gadaleta（steatosis MIE）や Hirte（テストセット）で利用が明示。citeturn15view0turn30view2  
- entity["organization","ChEMBL","embl-ebi bioactivity db"]：pEC50/EC50 を回帰として扱える例があり、キュレーション前提なら回帰拡張の候補。citeturn26view0turn25view0  
- NURA（Nuclear Receptor Activity database）：11 NR、15,247 compounds（PXR 含む）で few-shot / multi-task 研究に利用。citeturn33view0  

アクセス用リンク（コピー＆ペースト用、コード表記）：

```text
PubChem: https://pubchem.ncbi.nlm.nih.gov/
ToxCast/CompTox: https://comptox.epa.gov/dashboard/
ChEMBL: https://www.ebi.ac.uk/chembl/
RCSB PDB: https://www.rcsb.org/
```

（注：各データセットの “PXR アッセイID/ターゲットID” レベルの具体的な取得手順は、コンペ規約とユーザーの既存パイプラインに合わせて最短手順を書けるが、本回答は文献レビュー中心のため割愛した。）

---

## 参考：結晶構造（PDB）とコンペでの使いどころ

PXR の 3D 構造は「ドッキングで当てる」より、競技 ML では次の用途が現実的：

- **3D descriptor の生成**（形状・表面疎水性・ポケットフィットの proxy）  
- **SAR 解釈（なぜ logP が効くか、なぜ特定系列が強いか）**  
- **“系列外挿が弱い”理由の説明**（ポケットが柔軟＝複数結合モード＝2D 学習が揺れる）citeturn35search2turn35search0  

また、2025 年の事例では、PXR 共結晶（**5a86**）を参照しつつ、pEC50 回帰モデルを作って「PXR 活性を下げる」設計に使っている（これは“回帰モデル＋3D 類似”を組み合わせた設計例としてコンペにも示唆がある）。citeturn25view0

---

## 付録：PXR 関連 DOI 簡易リスト（クリック用）

```text
https://doi.org/10.1021/tx500389q
https://doi.org/10.2174/1386207319666160316122327
https://doi.org/10.1021/acs.chemrestox.6b00227
https://doi.org/10.1007/s11596-016-1609-4
https://doi.org/10.1007/s11356-017-9690-1
https://doi.org/10.1016/j.comtox.2017.01.001
https://doi.org/10.1016/j.xphs.2017.03.004
https://doi.org/10.1021/acs.jcim.8b00297
https://doi.org/10.1039/d2va00182a
https://doi.org/10.3390/cells11081253
https://doi.org/10.3389/fbinf.2024.1441024
https://doi.org/10.1186/s13321-024-00902-4
https://doi.org/10.3390/ijms26157630
https://doi.org/10.3390/applbiosci4010002
https://doi.org/10.1021/jm901137j
https://doi.org/10.1021/jm8004509
```

（上の DOI は本文中で触れたもののうち、アクセス確認できた範囲を網羅している。citeturn31view0turn30view2turn25view0turn36search0turn37search0turn37search1）
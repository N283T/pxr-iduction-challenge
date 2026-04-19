# 単一固定標的の pEC50 / pIC50 回帰に関する最新文献レビュー

このブリーフに最も近い設定は、**固定タンパク質・リガンドのみ変動・ラベル数 1k–10k の小中規模 QSAR 回帰**です。文献全体を通しての結論はかなり一貫していて、**単純な「大きいモデルをそのまま微調整」よりも、低容量適応、弱ラベルの使い方の設計、分子間類似性の活用、そして不確実性付きアンサンブル**のほうが、このデータサイズでは再現性のある利得を出しやすい、というものです。逆に、**雑なマルチタスク化、信頼性の低い単一ポーズ 3D、あるいは 2D 強ベースラインを上回る前提での 3D 信仰**は、追試で崩れやすいです。なお、今回の対象は entity["organization","OpenADMET","open science organization"] の PXR blind challenge で、公式の challenge ページとチュートリアル類は entity["company","Hugging Face","AI platform company"] と entity["company","GitHub","code hosting platform"] 上で公開されています。 citeturn34search1turn34search0turn12view2turn21view1

## Foundation model fine-tuning

### ChemFM

**著者・会場**: Z. Wang ら, *Nature Machine Intelligence* 2025  
**識別子**: DOI **10.1038/s42256-025-01022-w**  
**要点**: 化学 foundation model を、**フル fine-tuning ではなく adapters / LoRA 的な軽量適応**で下流物性に合わせる設計を提示しています。小規模下流データでも、凍結 backbone に軽量更新を載せる方が、過学習を抑えながら精度と計算効率を両立しやすい、というメッセージがこのブリーフに最も近いです。  
**データ/標的**: 複数の分子物性ベンチマーク。単一タンパク質専用ではないが、小規模回帰タスクへの転移に重点。  
**報告指標**: 回帰では **RMSE / MAE / R²** を中心。  
**転用ポイント**: PXR のような 4k ラベル規模なら、**full FT より LoRA / adapter / head-only の比較を必ず並べる**価値が高いです。  
**コード**: 論文ページで公開情報あり。 citeturn1search6

### EffiChem

**著者・会場**: E. Li ら, *ChemRxiv* 2025  
**識別子**: DOI **10.26434/chemrxiv-2025-qwd4z**  
**要点**: 化学言語モデルに対して、**full fine-tuning、LoRA、adapter などの parameter-efficient fine-tuning を系統比較**した前向きな論文です。検索結果上の要約でも、**低データ・低計算予算では PEFT が強く、trainable parameter を大幅削減しつつ full FT を上回るケースがある**ことが明示されています。  
**データ/標的**: ChemBERTa-2 / MoLFormer 系を含む下流分子物性予測。単一標的専用ではない。  
**報告指標**: **RMSE / MAE / ROC-AUC** など下流タスクごとの標準指標。  
**転用ポイント**: あなたの regime では、**LoRA rank、dropout、layer-wise LR decay、frozen depth** を CV で振る価値があります。特に **全層更新は最後の数層だけに限定**した方が安全です。  
**コード**: 公開実装の案内あり。 citeturn4search0

### MoLFormer

**著者・会場**: J. Ross ら, *Nature Machine Intelligence* 2022  
**識別子**: DOI **10.1038/s42256-022-00580-7**  
**要点**: **linear-attention 系の大規模 SMILES transformer** で、分子表現学習を大規模事前学習から回帰タスクへ落とし込んだ代表例です。重要なのは「巨大事前学習そのもの」より、**小規模回帰への fine-tune で一貫した強さを出せる backbone を確保する**点です。  
**データ/標的**: 標準分子物性ベンチマーク群。単一標的 QSAR 専用ではない。  
**報告指標**: **RMSE / MAE / ROC-AUC**。  
**転用ポイント**: PXR では、**MoLFormer 埋め込みをそのまま固定特徴として使うより、PEFT で task adaptation する**方が文献の流れに合っています。  
**コード**: 論文・関連実装の案内あり。 citeturn0search13turn12view2

**短いまとめ**: foundation model 側は、現時点では「どの backbone が絶対に最強か」よりも、**小データでの適応方法**が勝敗を決めています。特にこの regime では、**PEFT > full FT** になっても不思議ではありません。さらに、後述の通り、**データが 10k 未満だと表現学習モデルが固定記述子に負けるケースも珍しくない**ので、foundation model は単独勝負ではなく、**descriptor / graph 系との異種アンサンブル前提**で考えるのが安全です。 citeturn4search0turn12view2

## GNN の設計と小規模 QSAR 向けトリック

### Analyzing Learned Molecular Representations for Property Prediction

**著者・会場**: K. Yang ら, *Journal of Chemical Information and Modeling* 2019  
**識別子**: DOI **10.1021/acs.jcim.9b00237**  
**要点**: D-MPNN / Chemprop 系の事実上の原典です。単に「GNN が強い」ではなく、**小データでは fixed descriptors が勝つことがある、D-MPNN に RDKit 記述子を足す hybrid が効く、scaffold / chronological split は random split より難しい**という、今でも実務的に重要な結論がまとめられています。  
**データ/標的**: **19 公開 + 16 社内**データセット。単一標的専用ではない。  
**報告指標**: **RMSE / MAE / ROC-AUC / PRC-AUC**。  
**転用ポイント**: あなたが D-MPNN を既に試している点を踏まえると、新規性は**「純 D-MPNN ではなく、descriptor-aware hybrid と厳しい split 設計」**にあります。  
**コード**: Chemprop 実装へ継承。 citeturn13search0turn13search12

### Chemprop

**著者・会場**: E. Heid ら, *Journal of Chemical Information and Modeling* 2024  
**識別子**: DOI **10.1021/acs.jcim.3c01250**  
**要点**: D-MPNN を中心に、**不確実性推定、校正、反応・スペクトル・原子/結合レベル予測**まで拡張した現行実装の整理です。単なるソフトウェア論文に見えますが、小規模 QSAR では **再現可能な training protocol と UQ の選択肢**がそのまま性能差になります。  
**データ/標的**: MoleculeNet, SAMPL など。  
**報告指標**: **RMSE / MAE / calibration metrics**。  
**転用ポイント**: 新しく試すなら、**point estimate だけでなく deep ensembles + post-hoc calibration を本番候補に残す**べきです。  
**コード**: GitHub: **chemprop**。 citeturn13search1turn11search10

### Molecular property prediction based on graph structure learning

**著者・会場**: B. Zhao ら, *Bioinformatics* 2024  
**識別子**: DOI **10.1093/bioinformatics/btae304**  
**要点**: 分子内グラフだけでなく、**分子同士の fingerprint 類似性グラフ**を別レベルで作り、反復的に structure learning する手法です。活動クリフや小データ性に対して、**近傍分子の情報を label propagation 的に使う**発想が非常に実務向きです。  
**データ/標的**: **10 ベンチマーク**。タンパク質特異的ではない。  
**報告指標**: **RMSE / ROC-AUC** など。  
**転用ポイント**: 固定標的 PXR では、**train + unlabeled test を含む化学空間での similarity graph 構築**がかなり自然です。CV 内でのみラベル伝播強度を学ぶ設計は、実装コストに対して期待値が高いです。  
**コード**: GitHub: **GSL-MPP**。 citeturn17search1turn15view1

### Explainable uncertainty quantifications for deep learning-based molecular property prediction

**著者・会場**: C.-I. Yang, Y.-P. Li, *Journal of Cheminformatics* 2023  
**識別子**: DOI **10.1186/s13321-023-00682-3**  
**要点**: 分子物性予測で **aleatoric / epistemic uncertainty を分離**し、しかも atom-wise に説明可能にした仕事です。さらに、**deep ensembles の aleatoric 側を post-hoc calibration する**提案があり、小規模 noisy QSAR にかなり実務的です。  
**データ/標的**: 分子物性予測タスク群。  
**報告指標**: **calibration / confidence interval quality / accuracy**。  
**転用ポイント**: Leaderboard が Spearman 主体でも、**予測平均だけでなく不確実性で重み付けしたアンサンブル**にすると rank の安定化に効きます。疑わしい外れ分子を pseudo-label 候補から落とす用途にも向きます。  
**コード**: 論文ページで方法詳細あり。 citeturn10view4

**短いまとめ**: GNN 側で最も転用しやすいのは、**(i) descriptor-hybrid、(ii) molecule–molecule similarity graph、(iii) calibrated deep ensembles**です。逆に、単純な「GNN をさらに深くする」方向は、このデータサイズでは優先度が下がります。 citeturn13search0turn17search1turn10view4turn12view2

## 自己教師あり事前学習と小規模下流タスクへの転移

### MolCLR

**著者・会場**: Y. Wang, J. Wang, Z. Cao, A. B. Barati Farimani, *Nature Machine Intelligence* 2022  
**識別子**: DOI **10.1038/s42256-022-00447-x**  
**要点**: **atom masking / bond deletion / subgraph removal** を使う contrastive pretraining の代表格です。大規模 unlabeled 分子で表現を作り、下流回帰へ fine-tune すると、複数ベンチマークで有意な改善を出します。  
**データ/標的**: **約 1,000 万分子**で事前学習し、複数の benchmark regression/classification に転移。  
**報告指標**: **RMSE / MAE / ROC-AUC**。  
**転用ポイント**: そのまま pretrain し直すより、**MolCLR 的 augment を PXR 学習内の auxiliary objective として軽く載せる**方が現実的です。  
**コード**: GitHub: **MolCLR**。 citeturn19search0turn19search7

### Uni-Mol

**著者・会場**: G. Zhou ら, *ChemRxiv* 2023  
**識別子**: DOI **10.26434/chemrxiv-2022-jjm0j-v4**  
**要点**: 3D 分子表現学習の代表で、**masked atom 学習と 3D 座標の denoising**を組み合わせています。分子だけでなく pocket 表現も扱えるため、固定標的 QSAR で **pocket-aware side feature** を作る発想に向いています。  
**データ/標的**: 大規模 3D 分子コーパスと複数下流タスク。  
**報告指標**: **RMSE / ROC-AUC / 3D 空間タスク指標**。  
**転用ポイント**: PXR の場合、3D descriptor を生で足すより、**Uni-Mol 由来の learned 3D embedding を弱く融合**した方が筋がよいです。  
**コード**: 公開プロジェクトあり。 citeturn18search3

### Mole-BERT

**著者・会場**: J. Xia ら, *ChemRxiv* 2023 / ICLR 2023  
**識別子**: DOI **10.26434/chemrxiv-2023-dngg4**  
**要点**: シンプルな AttrMasking の弱点だった「原子語彙の小ささ」を、**context-aware tokenizer と triplet masked contrastive learning**で補強した論文です。重要なのは、**pretraining objective の設計そのものが downstream の小規模 QSAR で効く**ことを明示した点です。  
**データ/標的**: 分子グラフ事前学習 + 複数下流タスク。  
**報告指標**: **RMSE / ROC-AUC**。  
**転用ポイント**: 単一標的回帰では、**masked-token 系より triplet / relative-distance 系のほうが順位情報を保ちやすい**ので、loss 設計のヒントになります。  
**コード**: GitHub: **Mole-BERT**。 citeturn18search2turn18search10

### MolFCL

**著者・会場**: X. Tang ら, *Bioinformatics* 2025  
**識別子**: DOI **10.1093/bioinformatics/btaf061**  
**要点**: **BRICS fragment 反応知識を使う contrastive pretraining**と、**functional-group prompt fine-tuning**を組み合わせた仕事です。pretrain は **ZINC15 の 25 万分子**、fine-tune は **9 MoleculeNet + 14 TDC** の計 23 データセットで行われています。  
**データ/標的**: タンパク質専用ではないが、ADMET と小規模回帰を多く含む。  
**報告指標**: **MSE / RMSE / ROC-AUC**。  
**転用ポイント**: PXR は pocket が大きく柔軟で、活性決定に**官能基の寄与が強い**ので、**functional-group prompt / pharmacophore-aware prompt**は実装優先度が高いです。  
**コード**: GitHub: **MolFCLSupplementary**。 citeturn6view1

**短いまとめ**: 事前学習で実際に効いているのは、**データを増やすこと自体**よりも、**chemically valid な augmentation と、官能基・相対距離・3D 一貫性を保つ objective**です。あなたが既に「単一濃度 readout での pretrain → pEC50 fine-tune」で伸びなかったのは、Buterez 系が示す通り、**弱ラベルを pretrain だけに閉じ込めると取り込み方が不十分**だから、と読むのが自然です。 citeturn19search0turn18search2turn18search3turn6view1turn21view1

## 隣接アッセイと weak label の使い方

### Multi-fidelity machine learning models for improved high-throughput screening predictions

**著者・会場**: D. Buterez ら, *ChemRxiv* 2022  
**識別子**: DOI **10.26434/chemrxiv-2022-dsbm5**  
**要点**: **primary single-dose と confirmatory dose-response** をまとめて使う multi-fidelity 学習の初期実証です。要約では、**平均 MAE を 12% 低下、R² を 152% 改善**とされており、単純な confirmatory-only 学習よりも、低 fidelity 情報を何らかの形で持ち込む価値を強く示します。  
**データ/標的**: 公開 HTS データ群。単一タンパク質の multi-assay。  
**報告指標**: **MAE / R²**。  
**転用ポイント**: PXR でも single-dose や counterscreen があれば、**pretrain ではなく multi-fidelity feature augmentation**を本命にすべき、という示唆です。  
**コード**: あり。 citeturn22search0turn22search3

### MF-PCBA

**著者・会場**: D. Buterez ら, *Journal of Chemical Information and Modeling* 2023  
**識別子**: DOI **10.1021/acs.jcim.2c01569**  
**要点**: **60 の multi-fidelity HTS データセット、1,660 万超の分子–タンパク質相互作用**から成る benchmark です。重要なのは、「multi-fidelity は効くこともあるが、データ相関や化学空間の粗さ次第で効果が大きく変わる」と体系化した点です。  
**データ/標的**: 多数の単一タンパク質 HTS カスケード。  
**報告指標**: benchmark ごとに **MAE / R² / MCC / AUROC** など。  
**転用ポイント**: 自前の PXR 補助 assay があるなら、**相関係数・coverage・scaffold overlap** を先に診断してから入れるべきです。  
**コード**: GitHub: **mf-pcba**。 citeturn20search0turn20search6turn22search9

### Transfer learning with graph neural networks for improved molecular property prediction in the multi-fidelity setting

**著者・会場**: D. Buterez ら, *Nature Communications* 2024  
**識別子**: DOI **10.1038/s41467-024-45566-8**  
**要点**: このテーマで最も実装価値が高い論文です。結論は非常に明快で、**single-dose の「生ラベル」や、その low-fidelity モデルが作る embedding を high-fidelity モデルに追加すると、10–40% の MAE 改善、場合によっては R² が 8 倍超まで改善**します。一方で、**sum-readout 型の low-fidelity モデルはしばしば失敗し、性能を悪化させる**と明言しています。  
**データ/標的**: AstraZeneca / PubChem の drug-discovery multi-fidelity セット。  
**報告指標**: **MAE / R² / MCC / Pearson’s r**。  
**転用ポイント**: あなたが既に試した「single-concentration pretrain → fine-tune」が伸びなかったなら、次は **low-fidelity model を別建てで学習し、その embedding を high-fidelity regressor に side feature として結合**するのが本筋です。  
**コード**: 論文ページから Zenodo / repository が参照されています。 citeturn21view1turn22search1turn22search4

### Enhancing molecular property prediction with auxiliary learning and task-specific adaptation

**著者・会場**: V. Dey ら, *Journal of Cheminformatics* 2024  
**識別子**: DOI **10.1186/s13321-024-00880-7**  
**要点**: pretrained molecular GNN を auxiliary task と一緒に適応させるときの **negative transfer** を、**RCGrad / BLO+RCGrad** で緩和する研究です。最大 **7.7% の改善**を報告し、特に **low-data downstream** で効きやすいとしています。  
**データ/標的**: 複数の molecular property task。  
**報告指標**: 主に **accuracy / ROC-AUC / 回帰精度**。  
**転用ポイント**: 補助タスクを完全に捨てる前に、**gradient surgery を入れた単一主タスク + 補助 SSL / 補助 assay** を一度試す価値があります。  
**コード**: 論文ページで方法詳細あり。 citeturn24view0

### A meta-learning framework to mitigate negative transfer in transfer learning applicable to drug design

**著者・会場**: A. Mera, M. Vogt, J. Bajorath, *Scientific Reports* 2025  
**識別子**: DOI **10.1038/s41598-025-22058-3**  
**要点**: source task のサンプルを instance-level で重み付けし、**target に有害な source 例を抑える** transfer learning front-end です。キナーゼ阻害データで、**強い negative transfer 下ではそれを約 50% 緩和**する、と報告されています。  
**データ/標的**: protein kinase inhibitor 活性。関連だが同一標的ではない。  
**報告指標**: **AUC / negative transfer index**。  
**転用ポイント**: PXR で近縁核内受容体や補助 assay を使うなら、**全部の source 分子を平等に使わない**のが重要です。  
**コード**: 論文ページで共有リンクあり。 citeturn24view2

### Improving molecular property prediction through a task similarity enhanced transfer learning strategy

**著者・会場**: H. Li ら, *iScience* 2022  
**識別子**: DOI **10.1016/j.isci.2022.105231**  
**要点**: **MoTSE** という task similarity estimator を使い、どの source task から transfer すべきかを決める研究です。重要なのは、「transfer は source が多いほど良い」ではなく、**似たタスクを選んだときだけ効く**ことをはっきり示した点です。  
**データ/標的**: 複数の molecular property task。  
**報告指標**: 下流タスク側の **ROC-AUC / RMSE** など。  
**転用ポイント**: あなたの joint multitask が失敗したのは、まさにこの論文の問題設定そのものです。**補助 assay は similarity diagnosis なしに足さない**方がよいです。  
**コード**: 論文ページ / 公開記事あり。 citeturn25search0turn25search2

**短いまとめ**: weak label は「あるだけ足せば効く」わけではありません。あなたの regime では、**生の multitask 共有より、low-fidelity embedding の side-feature 化**が最も期待値が高く、**sum pooling 的な雑な low-fidelity trunk は危ない**というのが、今の最も強い結論です。 citeturn21view1turn24view0turn24view2turn25search0

## 3D と structure-aware モデル

### ATOMICA

**著者・会場**: A. Fang ら, *bioRxiv* 2025  
**識別子**: DOI **10.1101/2025.04.02.646906**  
**要点**: 蛋白質–リガンドや他の分子間界面を横断して、**原子レベル interface representation** を学習する geometric deep learning モデルです。固定標的 QSAR へそのまま入れるより、**pocket-side encoding を作る trunk** としての価値が大きいです。  
**データ/標的**: 多様な intermolecular interface。  
**報告指標**: interface prediction / downstream task 指標。  
**転用ポイント**: PXR の pocket を 1 つに決め打ちできるなら、**ligand-only trunk に pocket embedding を cross-attend させる**発想の根拠になります。  
**コード**: プロジェクトページあり。 citeturn28search1turn28search16

### A Folding-Docking-Affinity framework for protein-ligand binding affinity prediction

**著者・会場**: M.-H. Wu, Z. Xie, D. Zhi, *Communications Chemistry* 2025  
**識別子**: DOI **10.1038/s42004-025-01506-1**  
**要点**: protein folding → docking → affinity 予測をつなげた **FDA** フレームワークです。ただし結論は節度があり、**state-of-the-art docking-free 法と同程度**で、劇的に上回るわけではありません。  
**データ/標的**: 汎用 protein–ligand affinity。  
**報告指標**: **binding-affinity accuracy 指標**。  
**転用ポイント**: 「固定標的だから 3D を入れれば勝つ」とは言えず、**信頼できる pocket / pose があるときだけ条件付きで有利**、という理解が妥当です。  
**コード**: 論文ページ参照。 citeturn32view0

### Protein-ligand binding affinity prediction using multi-instance learning with docking structures

**著者・会場**: H. Kim ら, *Frontiers in Pharmacology* 2025  
**識別子**: DOI **10.3389/fphar.2024.1518875**  
**要点**: docking pose を 1 つだけ使わず、**5–10 個の pose を bag として attention pooling する MIL 回帰**です。PDBbind と SARS-CoV-2 Mpro で、**単一トップポーズや平均化より良い** binding-affinity 予測を示しています。  
**データ/標的**: **PDBbind 2020**、SARS-CoV-2 main protease。  
**報告指標**: **RMSE / MAE / Pearson / Spearman**。  
**転用ポイント**: 3D を使うなら、**単一 pose の Boltz / docking score を直接回帰器へ入れるより、複数 pose を attention で soft-pool**する方が望ましいです。  
**コード**: 論文ページで方法詳細あり。 citeturn32view1

### Boltz-2

**著者・会場**: S. Passaro ら, *bioRxiv* 2025  
**識別子**: DOI **10.1101/2025.06.14.659707**  
**要点**: protein–ligand 共折りたたみと affinity prediction を同時に行う大型 foundation model です。公式リポジトリの説明でも、**binder probability と affinity_pred_value** を持ち、lead optimization での利用を意識しています。  
**データ/標的**: 汎用 protein–ligand。  
**報告指標**: affinity / structure の複数指標。  
**転用ポイント**: 既に trunk embedding を使っているなら次の一歩は、**生の trunk pooling で終わらせず、pose-bag 化や pocket-conditioned head を載せること**です。  
**コード**: GitHub: **boltz**。 citeturn30search0turn30search1

### 3D が本当に効くのか

ここで重要なのは、**3D は「正しい pose に近いときだけ効く」**という点です。2025–2026 の検証では、Boltz-2 の affinity head は魅力的ですが、**inter-protein な benchmark では scoring noise 問題が大きく、標的識別や lead ranking に単独で使うには危うい**ことが、新 benchmark と独立評価の両方で指摘されています。したがって、PXR での 3D は **主役ではなく side information**、しかも **単一 pose ではなく複数 pose / confidence / interaction pattern を soft-pool する**のが安全です。 citeturn29search2turn29search3turn29search6turn30search15

## PXR 固有の先行研究

### Machine learning and traditional QSAR modeling methods in predicting PXR agonism of drug-like compounds

**著者・会場**: W. M. Neal ら, *Journal of Biomolecular Structure and Dynamics* 2024  
**識別子**: DOI **10.1080/07391102.2023.2196701**  
**要点**: PXR agonism について、**traditional QSAR と ML を比較**した比較的新しい論文です。タスクの難しさを、PXR の**広く柔軟な結合ポケット**と結びつけて議論しており、challenge setting に非常に近いです。  
**データ/標的**: **human PXR agonism**。  
**報告指標**: QSAR/ML 標準分類指標。  
**転用ポイント**: PXR は「一般的な bioactivity regression」より **化学空間の許容幅が広い**ので、**官能基と lipophilic bulk の両方を見るモデル**が重要になります。  
**コード**: 公開コードは確認できず。 citeturn33search0turn33search1

### Development and Experimental Validation of Regularized Machine Learning Models to Predict PXR Activation in Humans and Rodents

**著者・会場**: S. Hirte ら, *Cells* 2022  
**識別子**: DOI **10.3390/cells11081253**  
**要点**: **正則化付き ML モデル**で PXR 活性化を予測し、しかもヒトとげっ歯類で検証した論文です。要約でも、PXR 予測が難しい理由として**large and flexible binding pocket**が前面に出ています。  
**データ/標的**: human / rodent PXR activation。  
**報告指標**: 主に分類指標。  
**転用ポイント**: PXR では **過度に sharp な decision boundary より、正則化と外れ値耐性**の方が重要という示唆です。  
**コード**: 公開コードは確認できず。 citeturn33search11

### Building a Chemical Toolbox for Human PXR Activation and Modulation

**著者・会場**: *Journal of Medicinal Chemistry* 2021  
**識別子**: DOI **10.1021/acs.jmedchem.0c02201**  
**要点**: human PXR の agonist / antagonist / modulator を体系的に整理し、**PXR 活性を設計・回避するための化学的道具箱**を構築した論文です。固定標的回帰では、こうした curated chemical toolbox が **補助学習の source 選定**に直結します。  
**データ/標的**: human PXR。  
**報告指標**: 実験活性・調節作用の定性的/定量的整理。  
**転用ポイント**: PXR 専用の weak-label source を作るなら、**この種の curated modulators を優先**すべきです。  
**コード**: なし。 citeturn33search16

### Designing Out PXR Activity on Drug Discovery Projects

**著者・会場**: A. Hall ら, *Journal of Medicinal Chemistry* 2021  
**識別子**: DOI **10.1021/acs.jmedchem.0c02245**  
**要点**: PXR 活性を medicinal chemistry でどう避けるかを扱う perspective です。単純な建模論ではなく、**どのような化学設計判断が PXR リスクの低減に結びつくか**がまとめられていて、feature engineering の着眼点として有用です。  
**データ/標的**: human PXR。  
**報告指標**: perspective のため定量指標より設計規則が主。  
**転用ポイント**: 最終モデルに、**PXR リスクに関連する substructure / lipophilicity / H-bonding motif の hand-crafted flag** を足す根拠になります。  
**コード**: なし。 citeturn33search19

### Chemical manipulation of an activation/inhibition switch in PXR-controlled pathways

**著者・会場**: E. Garcia-Maldonado ら, *Nature Communications* 2024  
**識別子**: DOI **10.1038/s41467-024-48472-1**  
**要点**: **PXR-selective agonist と antagonist の co-crystal 構造**を通じて、活性化/阻害のコンフォメーション・スイッチを説明した論文です。PXR では「同じ pocket を埋めても転写活性は同じではない」ことが、構造的にかなりはっきり示されています。  
**データ/標的**: PXR-selective agonists / antagonists。  
**報告指標**: 構造・実験活性。  
**転用ポイント**: PXR で 3D を使うなら、**単なる shape 記述子ではなく、agonist / antagonist を分ける interaction pattern**を見る必要があります。  
**コード**: なし。 citeturn33search22

### The Identification of Ligand Features Essential for PXR Activation

**著者・会場**: S. Ekins ら, *Journal of Chemical Information and Modeling* 2005  
**識別子**: DOI **10.1021/ci049722q**  
**要点**: 古いですが、PXR で今も実務的に重要な構造知識を与える論文です。要約では、**Gln285 への水素結合が PXR 活性化に不可欠で、多くの ligand は His407 に第二の相互作用を作る**とされています。  
**データ/標的**: human PXR。  
**報告指標**: SAR / docking / feature identification。  
**転用ポイント**: これは feature engineering に直結します。**Gln285 / His407 を満たしうる donor–acceptor 配置や pharmacophore count** は、今でも安価で効く追加特徴候補です。  
**コード**: なし。 citeturn33search10turn33search8

**短いまとめ**: PXR 固有の文献は、**「大きくて柔軟な pocket」「活性化と阻害を分ける微妙な相互作用差」「Gln285 / His407 を軸にした H-bonding」**を繰り返し示しています。したがって、単なる 3D descriptor 追加より、**官能基・薬理パターン・interaction-aware prompt** の方が現実的です。 citeturn33search0turn33search11turn33search22turn33search10

## コンペ系の公開レシピと実務的示唆

このトピックは、**査読付き論文より GitHub / leaderboard artifact の方が情報量が多い**です。ただし、PXR 専用の優勝解法 postmortem は、私が確認できた範囲ではまだ十分に公開されていません。そこで、再現可能性のある公開 artifacts だけを挙げます。 citeturn34search0turn34search1turn33search9

### 公式 challenge artifact

**著者・会場**: entity["organization","OpenADMET","open science organization"], 2025–2026  
**識別子**: DOI なし  
**要点**: 公式 GitHub 組織には **PXR-Challenge-Tutorial** があり、challenge space では leaderboard が公開されています。さらに公式ブログでは、**最終提出に report か GitHub repo を必須**としており、方法の透明性を重視しています。  
**データ/標的**: OpenADMET PXR blind challenge。  
**報告指標**: leaderboard metric。  
**転用ポイント**: external scoreboard を回す前に、**公開 tutorial と同じ入出力・前処理・submission 形式**にまず厳密整合させること。  
**コード**: GitHub: **PXR-Challenge-Tutorial**。 citeturn34search0turn34search1turn33search9

### jonswain の OpenADMET blind challenge リポジトリ

**著者・会場**: J. Swain, GitHub artifact 2026  
**識別子**: DOI なし  
**要点**: **ECFP(2048, r=2) + 200+ RDKit 2D descriptors + Butina cluster CV + Chemprop + classical ML + meta-selector** というかなり実務的なレシピです。README 上では、**meta-selector を入れた ChemicalMetaRegressor が、素の multitask Chemprop より overall Spearman を 0.63 → 0.67 へ改善**しています。  
**データ/標的**: OpenADMET blind challenge endpoints。PXR 専用ではない。  
**報告指標**: **MAE / R² / Spearman / Kendall’s tau**。  
**転用ポイント**: あなたは Caruana forward selection を試していますが、この artifact が面白いのは、**「分子ごとにどの base model を使うか」を学習する点**です。単純重み平均よりこちらの方が rank metric に効く可能性があります。  
**コード**: GitHub: **OpenADMET_ExpansionRx_Blind_Challenge**。 citeturn34search2

### NgoSon の OpenADMET challenge リポジトリ

**著者・会場**: NgoSon2004, GitHub artifact 2026  
**識別子**: DOI なし  
**要点**: **Chemprop を 5 split で学習し、checkpoint average** するシンプルな recipe を公開しています。新規性は高くありませんが、**single-task / small-data 競技では「派手なモデル」より「split 安定化 + checkpoint average」がまだ有効**であることを示す良い対照です。  
**データ/標的**: OpenADMET challenge artifact。  
**報告指標**: リポジトリ記載の実験指標。  
**転用ポイント**: foundation model へ移っても、**5-fold 以上の scaffold / cluster split 上で seed-average した prediction cache** を作る運用は残すべきです。  
**コード**: GitHub: **OpenADMET-ExpansionRX-Challenge**。 citeturn33search21

**短いまとめ**: verified な競技 artifact から学べることは、**特徴量の多様性、cluster-aware CV、そして rank metric を直接監視する運用**です。PXR 専用の winner レポートはまだ薄いので、今はむしろ **一般 challenge artifact を PXR に局所化して再設計**する段階です。 citeturn34search2turn33search21turn34search0

## 順位相関を伸ばす学習目的と最終提案

### Molecular property prediction based on graph contrastive learning with partial feature masking

**著者・会場**: K. Dong ら, *Journal of Molecular Graphics and Modelling* 2025  
**識別子**: DOI **10.1016/j.jmgm.2025.109014**  
**要点**: **atom/bond の partial feature masking**で化学的意味を壊さずに contrastive pretraining を行い、さらに**batch 内の relative distance**を使って regression を強化する論文です。ここが重要で、単なる MSE 置換ではなく、**相対距離を学習に混ぜる**ことで回帰の順位構造を補強しています。  
**データ/標的**: MoleculeNet / ChEMBL の **12 benchmark**。  
**報告指標**: **回帰・分類の標準指標**。  
**転用ポイント**: PXR では、**MSE/Huber 本体 + batchwise pairwise distance 補助損失**が最も実装容易な rank-aware trick です。  
**コード**: 公開コードは確認できず。 citeturn15view2

### Meta-learning for transformer-based prediction of potent compounds

**著者・会場**: H. Chen, J. Bajorath, *Scientific Reports* 2023  
**識別子**: DOI **10.1038/s41598-023-43046-5**  
**要点**: 数値回帰ではなく、**高 potency 化合物を少数例から見つける**方向に transformer を meta-learning した研究です。単一標的 SAR の文脈で、**「強い化合物を上位に押し上げる」補助タスク**の有効性を示します。  
**データ/標的**: potent compound prediction tasks。  
**報告指標**: **AUC などの識別指標**。  
**転用ポイント**: PXR pEC50 では、本回帰 head に加えて、**top-quintile / top-decile potency classifier** を auxiliary head として並列学習するのが自然です。Spearman を意識した学習としては、これがかなり現実的です。  
**コード**: 公開コードは確認できず。 citeturn27search0turn27search3

### Task-Similarity is a Crucial Factor for Few-Shot Meta-Learning of Structure-Activity Relationships

**著者・会場**: A. Kötter ら, *ChemBioChem* 2024  
**識別子**: DOI **10.1002/cbic.202400095**  
**要点**: few-shot SAR meta-learning で、**task similarity が決定的**であることを示した研究です。単独では損失関数論文ではありませんが、順位・外挿性能を上げたいなら、**似ていない task を混ぜないことが一番効く**という、非常に重要な反証でもあります。  
**データ/標的**: few-shot SAR tasks。  
**報告指標**: few-shot task performance 指標。  
**転用ポイント**: 近接アッセイや他核内受容体を使う場合、**似ている task だけを残すフィルタ**を先に作るべきです。  
**コード**: 公開コードは確認できず。 citeturn27search2turn27search5

ここまでの verified な文献を見る限り、**小規模固定標的 QSAR で differentiable Spearman や Pearson loss が、よく調整された MSE/Huber を一貫して上回る**という強い化学文献はまだ乏しいです。実際に勝っているのは、**relative-distance 補助損失、top-potency の補助分類、multi-fidelity embedding、そして calibration / uncertainty を使った予測統合**です。つまり、**loss を一つ置き換えるより、学習信号の構造を増やす**方が成功しやすい、というのが現時点の実務結論です。 citeturn15view2turn27search0turn21view1turn10view4

### 何を次に試すべきか

以下は、**あなたが既に試したものを除いたうえで**、この PXR challenge 設定に対する期待インパクト順の提案です。

1. **single-dose / counterscreen を「別モデルの埋め込み」として高 fidelity 回帰器へ足す**  
   既に試した「single-concentration pretrain → fine-tune」とは別物です。Buterez 2024 が効いているのは、**LF モデルを独立に学習し、その embedding を HF モデルへ side feature として渡す**設計です。単純 joint multitask や sum-readout はむしろ危険です。 citeturn21view1turn24view2

2. **ChemFM / MoLFormer 系 backbone に対する PEFT**  
   full FT ではなく、**LoRA / adapter / last-k-layer FT** を比較してください。4,140 ラベルでは、PEFT の方が過学習しにくく、descriptor との異種アンサンブルにも載せやすいです。 citeturn1search6turn4search0turn0search13

3. **GSL-MPP 型の分子間 similarity graph を、best 2D モデルの後段に追加する**  
   これは「モデルを替える」より、**既に強い 2D 表現の上に transductive / semi-transductive な平滑化レイヤを足す**発想です。PXR の activity cliff 対策として特に相性が良いはずです。 citeturn17search1turn15view1

4. **functional-group prompt / pharmacophore-aware prompt を使う**  
   MolFCL の transferable 部分はここです。PXR では pocket が広く柔軟な分、**官能基レベルの寄与を明示的に model に見せる**方が、無秩序な 3D descriptor 追加より筋が良いです。 citeturn6view1turn33search10turn33search22

5. **MSE/Huber 本体に、batchwise relative-distance か pairwise ranking 補助損失を足す**  
   真っ先に試すべき metric-specific trick は、differentiable Spearman そのものではなく、**FMGCL 的な relative-distance 補助**です。実装が軽く、Spearman を壊しにくいです。 citeturn15view2

6. **uncertainty-calibrated deep ensemble を本気で使う**  
   Caruana forward selection を既に試していても、**calibrated deep ensembles + uncertainty-weighted averaging** は別物です。特に PXR のような柔らかい pocket では、**平均予測よりも「確信が一致した予測」を上位に出す**方が順位が安定します。 citeturn10view4turn10view3turn13search1

7. **3D を使うなら「単一 pose」ではなく「5–10 pose の attention pooling」へ切り替える**  
   今の literature では、**信頼できない単一 pose 3D は 2D 強ベースラインに負けやすい**です。Boltz-2 trunk や docking を使うなら、**MIL で複数 pose を soft-pool**してください。 citeturn32view1turn29search2turn29search6

8. **PXR 固有の medicinal-chemistry prior を hand-crafted 特徴として戻す**  
   具体的には、**Gln285 / His407 を満たしうる donor–acceptor 配置、lipophilic bulk、PXR-selective agonist/antagonist に多い substructure flag**です。これは安価ですが、PXR では surprisingly 効く可能性があります。 citeturn33search10turn33search16turn33search19turn33search22

9. **避けるべきもの**  
   **雑な multitask、sum-readout の low-fidelity transfer、単一 pose 3D、Boltz-2 affinity の単独利用**は、2024–2026 文献では繰り返し赤信号です。ここには追加予算をあまり入れない方が良いです。 citeturn21view1turn24view0turn29search2turn29search6turn12view2

総合すると、あなたの現状から最も有望なのは、**「low-fidelity を埋め込みとして別建てで使う」「PEFT した foundation model を descriptor 系と混ぜる」「molecule-similarity graph を後段に足す」**の三本です。PXR 固有には、**官能基 prompt と Gln285/His407 周辺の化学 prior**がコスト対効果のよい追加になります。一方、3D は依然として**条件付きオプション**であり、**信頼できる pose 集合を作れない限り 2D 強ベースラインを超える保証はありません**。 citeturn21view1turn4search0turn17search1turn33search10turn29search6
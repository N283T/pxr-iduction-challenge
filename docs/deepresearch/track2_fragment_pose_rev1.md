# Track 2: フラグメント結合姿勢予測法の包括調査

> **プロベナンス**: ChatGPT Deep Research レポート (2026-04-26 取得、prompt rev1)。
> 本ファイルはレポート本文をそのまま保存し、冒頭にメタ情報を追記したもの。
> `citeturnXXsearchXX` 表記は ChatGPT が内部で付与する検索結果 ID で、
> 原典トレース用に保持している。
>
> **このレポートの限界 (重要)**:
> - 当初の prompt rev1 が**手法名を列挙しすぎていた**ため、提示済みの方法群
>   (AlphaFold3 / Boltz / DiffDock / Vina / gnina など) の比較に調査が偏っている。
>   「私 (= AI/ML 側) が知らない手法」を炙り出すという当初目的に対しては unfair に
>   narrow。
> - フォローアップ用の **prompt rev2** (discovery-focused; FBDD コミュニティ /
>   結晶学 / QM/MM / 水分子ネットワーク / multi-state refinement 等を炙り出す方向)
>   を別途用意済み。そちらの結果が返ってきたら本ファイルへ追記 or 別ファイル化予定。
> - したがって、本レポートで「網羅的」とされている部分は **rev1 の枠内での網羅性**
>   であり、真の field 全体を網羅したものではない。
>
> **このサーベイから導いた直接的な意思決定** (PR #127 〜 #130 で実装済):
> - Boltz-2 を本命とし、recycling=5、diffusion_samples=5、use_potentials=True、
>   pocket constraint なしで初回提出 (lb_submissions id=33, 2026-04-26 10:10 JST)。
> - 1 pose 提出制約下で best-of-N の必要性を確認 → Boltz は confidence_score 順で
>   model_0 が最良なので model_0 を提出。
> - apo-soak 参照と AI 予測のズレ (本レポート「結論と実務判断」末尾) は懸念事項として
>   認識。LB 結果が反映され次第、rev2 サーベイ結果と合わせて re-ranking / pose
>   refinement の追加方針を検討する。

---

PXR のような**大きく、柔らかく、かつ多様な化学骨格を許容するポケット**では、現時点で最も再現性が高いのは「単一の最先端モデル」ではなく、**受容体アンサンブルに対する古典ドッキング**を主軸に置き、**co-folding 系モデルを代替姿勢生成器として併用し、最後に幾何学妥当性と相互作用整合性で 1 姿勢へ落とす**ハイブリッド運用です。公開ベンチマークでは、entity["organization","Google DeepMind","ai research lab"]の AlphaFold 3、Boltz、Chai-1 などの co-folding 系は Astex や PoseBusters 由来の apo/blind 条件で高い top-line RMSD を示しますが、**PB-valid と PLIF 特異性が常に追随するわけではない**こと、そして新規ポケット・多状態系・allosteric 系では優位性が縮むことが繰り返し示されています。逆に、Vina/gnina 系は top-line RMSD では見劣りしても、**受容体アンサンブル**と**良い再スコア**を組み合わせたときの頑健性が高く、PXR のような「正解姿勢が一意でない」課題に向いています。さらに、fragment は弱結合ゆえに**複数の準安定姿勢**を取りやすく、apo-soak 結晶では holo-like 誘導適合と異なる参照姿勢が現れうるため、**hard constraint よりも広めの soft box と multi-pose reranking**の方が安全です。citeturn11view2turn16view0turn35search9turn37search5turn25search12turn25search15turn34search5

## 結論と実務判断

まず、**fragment 特有の難しさは「接触数の少なさ」だけではありません**。弱い疎水接触しか作らない、対称や擬似対称の向きが多い、ポケットの局所水和や側鎖ロータマーの違いで姿勢順位が入れ替わる、という三重苦があります。CASP15 の総括でも、**小さく剛直なリガンドは全体として予測しやすい一方、大きく柔軟なリガンドは難しい**と報告されましたが、これは「fragment が簡単」という意味ではありません。むしろ fragment は**RMSD 的には近い代替姿勢が何通りも成立**しやすく、PXR のような malleable pocket ではその傾向が強くなります。公開ベンチマークでも、HA≤16 の slice を明示している論文は非常に少なく、SOTA 主張の多くは drug-like 分子を含む集計値に基づいています。これは本テーマにおける文献上の最大の欠落です。citeturn18search1turn11view2turn51search5

次に、**co-folding 系と古典ドッキング系の使い分け**です。PoseBench の Astex Diverse と PoseBusters Benchmark 由来セットでは、AlphaFold 3、Chai-1、Boltz-1、RFAA などの co-folding 系が blind/apo 条件で Vina を明確に上回るケースが多い一方、Morehead らは同時に、**DL 法が structural accuracy と chemical specificity の両立に苦しむ**こと、また AF3 が Chai-1 や Boltz-1 よりも **MSA 依存性が高い**ことも示しました。さらに同研究では、方法間性能が**2021年以前の PDB 類似性に相関**し、特に MSA あり Boltz-1、AF3、Chai-1 でその傾向が強いと報告されています。つまり、PXR の blind challenge で真に重要なのは、「見栄えのする 1 姿勢」を作ることではなく、**学習済み holo バイアス・PDB 類似性バイアス・局所幾何学破綻**を同時に抑えることです。citeturn13view0turn16view0turn36search5turn34search15

三点目に、**single pose 提出制約の下では “best-of-N” が必須**です。DiffDock の原論文は、multiple samples を confidence model で順位付けする設計で、**confidence と −RMSD の Spearman 相関 0.74**、さらに**信頼度上位 3 分の 1 に絞ると成功率が 38% から 80%**へ大きく上がることを示しました。PoseBench も generative 法の結果を**各複合体につき 3 independent runs の mean ± s.d.**で報告しており、run-to-run variance が無視できない前提で設計されています。したがって、PXR で 1 姿勢しか出せないなら、**単一 seed の 1 回出力をそのまま出す戦略は損**です。やるべきなのは、N サンプルを作り、**RMSD 平均座標ではなく medoid / consensus pose を選ぶ**ことです。座標平均は contact をぼかすだけで、LDDT-PLI 的にも理にかなっていません。これは文献値というより、DiffDock の confidence 設計と PoseBench の ensemble-ranking 実装からの実務的帰結です。citeturn35search3turn35search9turn10view0turn13view0

最後に、**apo-soak 参照と AI 予測のズレ**は現実にあります。結晶 soak は、結晶格子・溶媒チャネル・温度・拡散経路の制約下で成立した姿勢を与えます。co-folding はしばしばそれよりも「solution-like」「holo-like」な pocket arrangement を好みます。co-crystallization と soaking の比較研究では、**特に柔軟タンパク質や大きめリガンドで co-crystal の方が正しい結合様式を捉えやすい**とされ、別研究では soaked fragment 系で**室温でのみ見える別姿勢・別サイト・別の水和様式**まで確認されています。PXR 自体についても、近年の構造研究は**ligand flexibility と pocket malleability が協調して結合を成立させる**こと、そして**ポケット拡張は起こるが常に有利ではない**ことを示しました。従って、今回のような apo-soak 参照では、AI が作る「もっともらしい holo-like pose」を過信しない方が良いです。citeturn25search12turn25search3turn25search15turn29search0turn29search2turn26search20

## 方法分類表

以下の表は、**運用面**に焦点を当てた分類です。ライセンス欄は、取得できた公式 repo / 公式文書で確認できたものだけ明記し、それ以外は **未確認** としました。特に商用ソフトの細かな利用条件、研究コードの重み配布条件、再配布可否は導入前に必ず再確認してください。AlphaFold 3、Boltz、Chai-1、RFAA、NeuralPLexer、DiffDock、AutoDock Vina、RTMScore、EquiScore、PoseBusters については公開状況・ライセンスを取得ソースで確認しました。citeturn43search3turn43search1turn43search13turn44search0turn47view0turn48search1turn48search25turn50search0turn50search12turn52search5turn52search2turn51search5

### AI と ML

| 手法 | 区分 | 典型入力 | 典型出力 | GPU | ライセンス | 公開状況 |
|---|---|---|---|---|---|---|
| AlphaFold 3 | A: co-folding | 配列 + ligand 記述 | full complex | 実質Y | CC BY-NC-SA + weights terms | 公開コードあり、商用制限あり |
| Boltz-1 | A: co-folding | 配列 + ligand | full complex | Y | MIT | 公開 OSS |
| Boltz-2 | A: co-folding/affinity | 配列 + ligand | complex + affinity | Y | MIT（公式告知ベース） | 公開 OSS 系列 |
| RoseTTAFold-All-Atom | A: co-folding | 配列 + ligand/SDF | full complex | Y | BSD | 公開 OSS |
| Chai-1 | A: co-folding | 配列 + ligand | full complex | Y | Apache-2.0 | 公開 OSS |
| NeuralPLexer | A: co-folding | 配列 + ligand グラフ | full complex | Y | BSD-3-Clause-Clear | 公開 OSS |
| NeuralPLexer-2 | A: co-folding | 配列 + ligand | full complex | Y | 未確認 | 公開記事/白書中心、実装状況要確認 |
| ESMFold + ligand head | A: co-folding派生 | 配列 + custom ligand head | apo or complex | Y | 実装依存 | 標準実装なし、研究ワークフロー |
| DiffDock | A: diffusion docking | protein 구조 + ligand | multi-pose + confidence | Y | MIT 系 | 公開 OSS |
| DiffDock-L | A: diffusion docking | protein + ligand | multi-pose + confidence | Y | MIT 系 | 公開 OSS |
| DiffDock-Pocket | A: pocket-conditioned docking | protein + ligand + pocket | multi-pose | Y | 未確認 | 公開研究実装あり |
| TankBind | A: docking | protein + ligand | pose + score | Y | 未確認 | 公開研究実装あり |
| EquiBind | A: docking | protein + ligand | single pose | Y | 未確認 | 公開研究実装あり |
| GeoMol | A: conformation/pose subroutine | ligand or complex context | conformers/pose features | Y | 未確認 | 公開研究実装あり |
| ConfPass | A: diffusion/post | protein + ligand | refined conformers/poses | Y | 未確認 | 公開研究実装あり |
| FABind | A: docking | protein + ligand | pose + pocket | Y | 未確認 | 公開研究実装あり |
| Re-Dock | A: docking/re-ranking | protein + ligand set | refined poses | Y | 未確認 | 研究実装要確認 |
| PocketGen | A: pocket-conditioned generation | pocket + seed ligand | generated ligand/pose | Y | 未確認 | 公開研究実装あり |
| ResGen | A: generative pose/design | pocket + fragments | generated ligand/pose | Y | 未確認 | 公開研究実装あり |
| GraphBP | A: generative pose/design | pocket graph + ligand graph | generated pose | Y | 未確認 | 公開研究実装あり |
| PocketFlow | A: generative pose/design | pocket + ligand context | generated ligand/pose | Y | 未確認 | 公開研究実装あり |
| FLAG | A: generative pose/design | pocket + seed | generated ligands/poses | Y | 未確認 | 公開研究実装あり |
| TargetDiff | A: generative pose/design | pocket + conditioning | generated ligands/poses | Y | 未確認 | 公開研究実装あり |
| gnina | A/D: CNN rescoring+docking | receptor + ligand | docked poses + CNNscore | Y推奨 | GPL 系 | 公開 OSS |
| DeepDock | A: ML rescoring | receptor + pose set | rescored poses | Y | 未確認 | 公開研究実装あり |
| RTMScore | A: ML rescoring | receptor + pose set | residue-atom score | Y推奨 | MIT | 公開 OSS |
| EquiScore | A: ML rescoring | receptor + pose set | rescored poses | Y推奨 | MIT | 公開 OSS |
| PoSEIDON | A: ML rescoring | receptor + pose set | rescored poses | Y | 未確認 | 研究実装要確認 |
| ScoreFormer | A: ML rescoring | receptor + pose/graph | rescored poses | Y | 未確認 | 研究実装要確認 |

### 古典ドッキング

| 手法 | 区分 | 典型入力 | 典型出力 | GPU | ライセンス | 公開状況 |
|---|---|---|---|---|---|---|
| AutoDock Vina | B | receptor + ligand + box | ranked poses | N | Apache-2.0 | 公開 OSS |
| AutoDock-GPU | B | receptor + ligand + grid | ranked poses | Y | 未再確認 | 公開 OSS 系 |
| AutoDock4 | B | receptor + ligand + grid | ranked poses | N | 未再確認 | 公開 OSS 系 |
| Glide HTVS/SP/XP | B | prepared receptor + ligand | ranked poses | N | vendor EULA | 商用 |
| Induced Fit Docking / IFD-MD | B | receptor + ligand | induced-fit poses | Y推奨 | vendor EULA | 商用 |
| OpenEye FRED | B | receptor + conformers | ranked poses | N | vendor EULA | 商用 |
| OpenEye HYBRID | B | receptor + reference ligand | template-guided poses | N | vendor EULA | 商用 |
| OpenEye POSIT | B | receptor + reference ligands | template pose | N | vendor EULA | 商用 |
| GOLD | B | receptor + ligand + site | ranked poses | N | vendor EULA | 商用 |
| Surflex-Dock | B | receptor + protomol | ranked poses | N | vendor EULA | 商用 |
| FlexX | B | receptor + ligand | ranked poses | N | vendor EULA | 商用 |
| rDock / RxDock | B | receptor + cavity + ligand | ranked poses | N | LGPL-3.0 | 公開 OSS |
| PLANTS | B | receptor + ligand + site | ranked poses | N | 未再確認 | 学術配布中心 |
| LeDock | B | receptor + ligand | ranked poses | N | 未再確認 | 学術配布中心 |
| Smina | B/D | receptor + ligand | ranked poses | N | 未再確認 | 公開 OSS 系 |
| Ensemble docking | B | receptor ensemble + ligand | per-conformer poses | N | エンジン依存 | ワークフロー |

### 物理ベースと MD

| 手法 | 区分 | 典型入力 | 典型出力 | GPU | ライセンス | 公開状況 |
|---|---|---|---|---|---|---|
| Steered MD | C | 初期 complex | pulled trajectories | Y推奨 | engine依存 | ワークフロー |
| Well-tempered metadynamics | C | 初期 complex + CV | free-energy surface | Y推奨 | engine/PLUMED依存 | ワークフロー |
| Parallel-bias metadynamics | C | 同上 | pose basin map | Y推奨 | engine依存 | ワークフロー |
| Infrequent metadynamics | C | 同上 | kinetics-aware exits | Y推奨 | engine依存 | ワークフロー |
| Funnel metadynamics | C | pocket + ligand | binding-mode landscape | Y推奨 | engine依存 | ワークフロー |
| T-REMD | C | system | replica ensemble | Y推奨 | engine依存 | ワークフロー |
| H-REMD | C | system | replica ensemble | Y推奨 | engine依存 | ワークフロー |
| REST2 | C | system | hot-region ensemble | Y推奨 | engine依存 | ワークフロー |
| Umbrella sampling | C | CV windows | PMF / pose ranking | Y推奨 | engine依存 | ワークフロー |
| Adaptive sampling / MSM | C | many short traj | metastable pose ensemble | Y推奨 | engine依存 | ワークフロー |
| FEP+ pose scoring | C | aligned poses | relative ΔG ranking | Y | vendor EULA | 商用 |
| MM-PBSA / MM-GBSA | C | short MD ensemble | endpoint ΔG estimates | N/Y | engine依存 | ワークフロー |
| ATM | C | aligned ligands/poses | alchemical ranking | Y推奨 | 実装依存 | 研究実装 |
| AToM-OpenMM | C | system + perturbation | free-energy ranking | Y | OpenMM系/実装依存 | 公開研究実装 |

### ハイブリッドと後処理

| 手法 | 区分 | 典型入力 | 典型出力 | GPU | ライセンス | 公開状況 |
|---|---|---|---|---|---|---|
| Docking → short MD → rescoring | D | docked poses | relaxed & reranked poses | Y推奨 | 混成 | ワークフロー |
| Co-folding → minimization / MD | D | co-folded complex | clash-repaired pose | Y推奨 | 混成 | ワークフロー |
| Multi-docking → consensus scoring | D | heterogeneous pose sets | single consensus pose | N/Y | 混成 | ワークフロー |
| PoseBusters | D | predicted poses | validity flags/checks | N | BSD-3-Clause | 公開 OSS |
| PLIPify / ProLIF interaction fingerprints | D | pose set | interaction fingerprints | N | 実装依存 | 公開 OSS 系 |
| Cross-docking benchmark suites | D | method outputs | benchmark scores | N | suite依存 | データ/評価基盤 |

この分類表からの実務的含意は明快です。**PXR challenge で実際に戦力になるのは**、現実には AF3 / Boltz / Chai-1 / RFAA / NeuralPLexer / DiffDock-L / Vina / gnina / PoseBusters / ProLIF あたりの組み合わせであり、残りの多くは「研究の方向性として重要」でも、**24 時間・1 GPU・184 ligands を回す本番運用**では主役ではありません。特に fragment-rich 条件で価値が高いのは、(i) 多数姿勢を出せること、(ii) 妥当性フィルタを通せること、(iii) 受容体多状態性を扱えること、の三点です。citeturn43search20turn44search17turn49search2turn48search25turn37search1turn51search5

## ベンチマークと定量比較

まず強調したいのは、**公開ベンチマークの “勝ち” は評価設定に強く依存する**ことです。PoseBench は **predicted apo structure + blind pocket** を標準化し、PoseBusters original は**crystal receptor を使うことが多い redocking/cross-docking 的文脈**で、目的が少し違います。したがって、PXR apo-soak challenge に近いのは、むしろ PoseBench の Astex / PoseBusters / DockGen-E 側です。なお PoseBench の図版は **mean ± s.d. over three runs** として描画されていますが、図中に明示されている数値ラベルは平均値のみなので、下表も平均値を載せています。citeturn13view0turn11view2

### PoseBench の主要結果

| 手法 | Astex Diverse RMSD<2Å | Astex PB-valid | Astex PLIF-WM | PoseBusters Benchmark RMSD<2Å | PoseBusters PB-valid | PoseBusters PLIF-WM | コメント |
|---|---:|---:|---:|---:|---:|---:|---|
| P2Rank + Vina | 40.9 | 38.1 | 53.4 | 28.3 | 25.6 | 39.3 | blind 定常ベースライン |
| DiffDock-L | 83.3 | 24.6 | 79.0 | 81.3 | 16.3 | 79.9 | pose localization は強いが validity が弱い |
| NeuralPLexer | 77.4 | 21.0 | 84.9 | 68.0 | 22.7 | 80.3 | PLIF は強いが PB-valid が伸びない |
| Chai-1 | 88.9 | 71.0 | 88.1 | 82.4 | 55.7 | 84.7 | バランスが良い |
| Boltz-1 | 69.0 | 53.2 | 66.2 | 79.7 | 58.1 | 74.5 | PB-valid が比較的高い |
| AlphaFold 3 | 89.7 | 74.2 | 86.2 | 90.4 | 57.9 | 84.3 | top-line RMSD は最強クラス |

出典: PoseBench Fig. 2 と Fig. 4 の平均値ラベル。citeturn15view0turn16view0

この表の読み方は非常に重要です。**DiffDock-L は “当たりを引けば近い” が、PB-valid が低い**。一方で **AF3 / Chai-1 / Boltz-1 は RMSD と PLIF の両方を一定以上保ちやすい**。PXR のような fragment-rich flexible pocket では、提出が 1 pose だけなので、**“高 RMSD 成功率”より “高 PB-valid + 高 interaction consistency” を重く見るべき**です。PoseBench でも、後処理の relaxation は NeuralPLexer や AF3-Single-Seq ではかなり効いた一方、DiffDock-L には本質的解決になっていませんでした。つまり、**clash repair は有用だが、そもそもの pose family が違うと救い切れない**ということです。citeturn16view0turn15view0

さらに難しい DockGen-E では、著者ら自身が**AF3 ですら structurally and chemically accurate pose を 75% 超で取り逃がす**と書いており、新規ポケット一般化では co-folding 優位が鈍ることを示しています。CASP15 でも、multiligand では AF3 が目立つ一方、single-ligand では**AutoDock Vina、NeuralPLexer、Boltz-1 が他の co-folding 法より PLIF modeling で優位**なケースが報告されました。これは、「blind flexible docking の本質は、global fold prediction ではなく、局所相互作用の列挙と選別である」ことを示唆します。citeturn11view2turn16view0

### ほかの主要定量ポイント

| 指標 | 数値 | 含意 |
|---|---:|---|
| DiffDock top-1 success on PDBBind | 38% | 従来 docking 23%、既存 DL 20% を上回るが、blind docking の絶対値としてはまだ十分ではない |
| DiffDock confidence 上位 1/3 に絞った success | 80% | 内部 confidence は「選別器」として強い |
| DiffDock confidence と −RMSD の相関 | Spearman 0.74 | multi-sample 前提なら confidence は使える |
| gnina Top1 redocking | 58% → 73% | Vina score から CNN rescoring に置き換える価値が大きい |
| gnina Top1 cross-docking | 27% → 37% | cross-docking でも改善するが、まだ難しい |
| CASP15 総括 | small ions / small rigid organics は比較的高精度、large flexible ligands は難しい | fragment は「平均的には有利」だが、弱結合多義性では別の難しさが残る |

出典: DiffDock 原論文と gnina 原論文、CASP15 assessment。citeturn35search3turn35search9turn37search5turn18search1

ここから引ける結論は、**PXR fragment pose では “DiffDock 単独” も “AF3/Boltz 単独” も勧めにくい**ということです。DiffDock は internal reranking がうまくても PB-valid が弱く、co-folding は holo-like bias や training-similarity bias を受けやすい。したがって、**Vina/gnina の pose family、co-folding の pose family、そして receptor ensemble の 3 系統から候補集合を作る**のが最も安全です。citeturn16view0turn35search9turn34search5

## フラグメントと柔軟ポケットの論点

**HA≤16 の fragment subset を真正面から数値化した近年論文は、驚くほど少ない**です。PoseBusters も PoseBench も fragment-only split を main table では出しておらず、ここは文献空白です。そのため、今回の意思決定では「fragment 専用の公開 SOTA 表」ではなく、**fragment に本質的に必要な性質を持つ手法**を選ぶ必要があります。その性質とは、(i) 多峰性を潰さずに複数姿勢を保持できること、(ii) 局所 pocket rearrangement と水和を扱えること、(iii) 弱い接触でも幾何破綻せずに出力できること、の三点です。citeturn11view2turn51search5

この観点から見ると、fragment に最も理屈が合うのは **sampling-first** の物理系です。Linker らは、**apo protein 構造と ligand 化学構造だけから、unbiased MD + MSM で fragment-like molecule の binding site と binding mode を自動的に予測できる**ことを示しました。さらに Poole らの 2025 年論文では、GCNCMC を使って**occluded fragment binding site を効率的に見つけ、多数の binding mode を再現**できることが示されました。これは fragment 課題でしばしば起こる「正解が 1 個でない」「入口経路が狭い」「結晶では局所 site が見えるが solution では別 basin もある」という状況に非常に噛み合います。もちろん 184 ligands 全件にこの種のサンプリングを回すのは予算超過ですが、**曖昧な fragment だけに使う 2nd-stage refiner** としては極めて理にかなっています。citeturn38search2turn38search25

PXR や他の核内受容体に目を向けると、この必要性はさらに強まります。PXR の近年の構造研究は、**ligand flexibility と pocket malleability が協調して promiscuity を生む**こと、そして構造最適化により pocket expansion を誘導しても、必ずしも好ましい結合様式にならないことを示しました。PXR の「canonical site を知っているからそこに hard constrain すればよい」という発想は危ういです。PXR では canonical cavity 内にも**複数の subpocket occupancy** があり、apo-soak crystal では誘導適合の程度が限られる可能性もあります。類似の柔軟性は PPARγ でも 2025 年の advanced MD 研究で確認されており、apo と ligand-bound の間に**安定したが互いに異なる basin**が存在しました。核内受容体型ポケットは、まさに multiple receptor states を前提に扱うべき対象です。citeturn29search0turn29search2turn26search20turn31search1

**binding-site constraint のトレードオフ**については、実務上の答えは「soft box は使う、hard residue lock は避ける」です。PoseBench は blind 条件で一般化を評価するため pocket を与えていませんが、実務では pocket が既知なことも多く、その場合に**ポケット情報を全く使わない理由はありません**。ただし、fragment soak では unexpected micro-site や alternative pose が現れます。co-crystallization と soaking の比較研究、そして室温 fragment crystallography の研究は、**soaked crystal が solution の induced-fit 全部を映さない**こと、さらに**別姿勢や別サイトが温度や結晶条件で可視化されうる**ことを示しています。従って、PXR では canonical cavity 全体と隣接 subpocket を含む**広めの docking box**は使うべきですが、**特定残基との接触を必須にする constraint**は fragment ほど禁物です。これは soak 文献と PXR malleability からの実務的推論です。citeturn25search12turn25search3turn25search15turn29search0

**confidence / uncertainty estimator**については、現時点で一番エビデンスが強いのは DiffDock confidence です。一方で、AF3 の confidence や一般的な pLDDT / interface confidence は「protein side の品質指標」としては役立っても、**虚偽陽性 pose をふるい落とす ligand-pose score としては弱い**ことが、557 Mac1 複合体と 3 つの prospective virtual screen を使った 2026 年の大規模評価で示されました。同研究では、AF3 pose confidence は false positive 分離で docking score や Boltz-2 affinity ほど効かず、Boltz-2 affinity の方が potency とよく相関しました。したがって PXR では、**raw pLDDT / ipTM / confidence_score をそのまま pose rank に使わない**方が良いです。より現実的には、(i) model-internal confidence、(ii) gnina CNNscore、(iii) PoseBusters pass/fail、(iv) seed 間 spread、の 4 軸を併用するのが安全です。citeturn35search9turn32search8turn34search1turn34search4turn34search5turn51search5

## 推奨ワークフロー

以下の 3 つは、**1×RTX 5080 16 GB、24 時間、184 ligands、1 pose 提出**という制約を前提にした提案です。**期待 LDDT-PLI は文献に直接載っている値ではなく、あなたの現在値と公開ベンチマークの差分からの外挿**です。その点は明示しておきます。

| ワークフロー | 中核スタック | 実装難易度 | 24h 適合性 | 期待 LDDT-PLI | ねらい |
|---|---|---|---|---|---|
| 省予算 | Boltz-2 継続 + PoseBusters + gnina rescoring + seed medoid 選択 | 低 | 高い | **0.46–0.48** | まず variance を減らす |
| バランス型 | receptor ensemble Vina/gnina + Boltz-2/Chai-1 補助 + consensus rerank | 中 | 高い | **0.48–0.50** | fragment/柔軟 pocket に強い実戦解 |
| 品質重視 | バランス型 + 不確実 fragment のみ短時間 OpenMM 緩和 / MM-GBSA 2–3 pose tiebreak | 中〜高 | 条件付き | **0.49–0.51+** | 上位差分を詰める |

### 省予算

これは、今の Boltz-2 パイプラインを大きく壊さずに、**“1 pose の選び方” を改善する**案です。各 ligand について Boltz-2 の複数 seed 出力を保持し、(i) PoseBusters fail を落とし、(ii) local minimization 後に gnina CNNscore を付け、(iii) 互いの pose RMSD または contact fingerprint 類似度で **medoid** を 1 つ選ぶ、という流れです。fragment では single best confidence pose よりも **seed 間で最も「合意」される pose** の方が外れにくいことが多いからです。根拠は、DiffDock の confidence-as-selector の有効性と PoseBench の inter-run variability です。citeturn35search9turn13view0turn51search5

実運用での細部としては、**hard pocket constraint は入れず**、ただしローカル minimization と gnina rescoring のときだけ canonical cavity を含む box に落とし込むのが安全です。PXR fragment は unexpected subpocket に逃げる可能性がある一方、全表面 blind rescoring はノイズが増えるためです。難易度が低く、今日から即導入できる一方で、改善幅は大きくても数ポイント程度に留まると見ます。

### バランス型

これが今回の本命です。考え方は単純で、**pose generator を 1 種にしない**ことです。具体的には、PXR の apo 構造を中心に、可能なら既知 holo 構造や短い緩和で開閉の異なる 4–8 conformer を用意し、各 conformer に対して Vina または gnina で docking します。そこへ、Boltz-2 あるいは Chai-1 single-seq の pose family を足し、**古典探索系と co-folding 系の候補集合を融合**します。その後、PoseBusters、gnina CNNscore、contact fingerprint、一貫性スコアで単一 pose を選びます。citeturn37search5turn16view0turn44search0turn43search17

この案が fragment-rich PXR に向く理由は三つあります。第一に、古典ドッキングは **局所 subpocket occupancy の列挙**に強い。第二に、co-folding は **受容体の側鎖・ループの別解**を提案できる。第三に、両者が一致した pose は強い根拠になる。提出 1 姿勢制約下では、**“平均座標” ではなく “合意された実在 pose”** を選べることが重要です。実装難易度は上がりますが、24 時間内には十分収まります。fragment だけに exhaustiveness や conformer 数を厚くするなど、**targeted spending** がしやすい点も強みです。

### 品質重視

最も高く狙うなら、バランス型の上に**2 段階目の限定的物理緩和**を載せます。ただし全 184 ligands に長い MD をかけると予算超過なので、対象は以下のどれかに絞るべきです。  
第一に、top2 poses の score 差が小さいもの。  
第二に、fragment で接触が少なく seed spread が大きいもの。  
第三に、co-folding と docking が食い違うもの。  

これらに対して 100–500 ps 程度の restrained OpenMM 緩和か very short MD を回し、最後に MM-GBSA か単純な interaction stability を使って 2–3 候補から 1 つ選びます。ここで重要なのは、**MM-GBSA を全件の主スコアにしない**ことです。endpoint ΔG は弱結合 fragment の順位付けで不安定だからです。使うならあくまで **near-tie の tiebreaker** です。この workflow は、apo-soak 参照と holo-like 予測のズレを少しだけ埋めるのに向いています。物理緩和を ambiguity subset に限定する限り、24 時間でも実施可能です。citeturn25search12turn25search15turn38search2turn38search25

## 推奨論文

| 文献 | 一言でいうと |
|---|---|
| Abramson J. et al., **Accurate structure prediction of biomolecular interactions with AlphaFold 3**, *Nature* (2024), DOI: 10.1038/s41586-024-07487-w. citeturn44search20 | 現在の co-folding 系の基準点。小分子を含む複合体構造予測の土台。 |
| Wohlwend J. et al., **Boltz-1: Democratizing biomolecular interaction modeling**, *bioRxiv* (2024/2025), DOI: 10.1101/2024.11.19.624167. citeturn43search20turn11view1 | 商用利用可能な OSS co-folding の代表。PXR 実務でも最も導入しやすい。 |
| Passaro S. et al., **Boltz-2: Towards accurate and efficient binding affinity prediction**, *bioRxiv* (2025), DOI: 10.1101/2025.06.14.659707. citeturn43search17 | pose だけでなく affinity も出す方向に進んだ最新版。confidence の使い方を再考させる。 |
| Krishna R. et al., **Generalized biomolecular modeling and design with RoseTTAFold All-Atom**, *Science* (2024), DOI: 10.1126/science.adl2528. citeturn44search17 | small molecule を含む all-atom co-folding の重要論文。 |
| Qiao Z. et al., **State-specific protein–ligand complex structure prediction with a multi-scale deep generative model**, *Nature Machine Intelligence* (2024), DOI: 10.1038/s42256-024-00792-z. citeturn49search2 | NeuralPLexer の本論文。柔軟 pocket と state-specific 予測の文脈で重要。 |
| Morehead A. et al., **Assessing the potential of deep learning for protein–ligand docking**, *Nature Machine Intelligence* (2026), DOI: 10.1038/s42256-025-01160-1. citeturn11view0 | Astex, PoseBusters, DockGen-E, CASP15 を blind/apo 条件で比較した、今読むべき benchmark。 |
| Buttenschoen M., Morris G.M., Deane C.M., **PoseBusters: AI-based docking methods fail to generate physically valid poses or generalise to novel sequences**, *Chemical Science* (2024), DOI: 10.1039/D3SC04185A. citeturn51search5 | validity を評価軸の中心に持ち込んだ分水嶺。Pose が近いだけでは不十分。 |
| Robin X. et al., **Assessment of protein–ligand complexes in CASP15**, *Proteins* (2023), DOI: 10.1002/prot.26601. citeturn18search0turn18search1 | lDDT-PLI と BiSyRMSD の文脈で必読。OpenADMET 評価系との接続が良い。 |
| Corso G. et al., **DiffDock: Diffusion steps, twists, and turns for molecular docking**, *ICLR* (2023). citeturn35search3turn35search9 | diffusion docking の基礎論文。confidence が pose 選別器として有効。 |
| McNutt A.T. et al., **gnina 1.3: the next increment in molecular docking with deep learning**, *Journal of Cheminformatics* (2025), DOI: 10.1186/s13321-025-00973-x. citeturn37search13 | CNN rescoring を実務で使うなら最重要。Vina 系との相性が良い。 |
| Lin W. et al., **Structure-guided approach to modulate small molecule binding to a promiscuous ligand-activated protein**, *PNAS* (2023), DOI: 10.1073/pnas.2217804120. citeturn29search0 | PXR の promiscuity と pocket remodeling を構造的に扱う、今回の標的理解に直結する論文。 |
| Falbo E. et al., **From Apo to Ligand-Bound: Unraveling PPARγ-LBD Conformational Shifts via Advanced Molecular Dynamics**, *ACS Omega* (2025), DOI: 10.1021/acsomega.4c11128. citeturn31search1turn31search3 | 核内受容体の apo↔ligand-bound 多状態性を MD でどう扱うかの良い具体例。 |
| Kim J. et al., **Large scale prospective evaluation of co-folding across 557 Mac1-ligand complexes and three virtual screens**, reviewed preprint / preprint (2026), eLife reviewed preprint 110475 / bioRxiv 2025.12.25.696505. citeturn34search1turn34search5 | co-folding confidence が false positive 分離に弱いことを大規模に示した最近最重要の実証。 |

もし 3 本だけ読むなら、**PoseBench 2026、PoseBusters 2024、PXR PNAS 2023**を優先するのがよいです。理論、実務、標的理解の 3 本柱になるからです。citeturn11view0turn51search5turn29search0

## 落とし穴と限界

最初の落とし穴は、**benchmark leakage と見かけの SOTA**です。PoseBench の exploratory analysis は、DL 手法の精度が**PDB 類似性に相関**することを示し、特に Boltz-1、AF3、Chai-1 で有意な相関が見られました。したがって、Astex のような “既知に近い” セットで高い数値が出ても、そのまま PXR blind fragment へ外挿するのは危険です。PoseBench 公式 repo でも、v1.0.0 の ligand scoring bug 修正により **平均で約 15% 性能が下がった**と明記されており、再現実験のバージョン固定が不可欠です。citeturn16view0turn10view0

二つ目は、**構造っぽく見えるが PoseBusters で落ちる pose**です。PoseBusters は、芳香環の平面性、標準結合長、立体化学、分子内・分子間の steric plausibility を検査し、これを満たして初めて “state-of-the-art” と呼ぶべきだと主張しました。co-folding でも diffusion docking でも、RMSD がそこそこでも**幾何学的に破綻した pose**は珍しくありません。PXR の fragment では contact 数が少ないため、視覚的には自然に見える偽 pose を人間が誤採用しやすい点にも注意が必要です。citeturn51search5turn36search5

三つ目は、**allosteric / unusual pocket への過信**です。最近の allosteric kinase 研究では、generic DiffDock-L をそのまま使うより、**allosteric 系へ fine-tune**した方が pose recovery が大きく改善しました。つまり、汎用モデルは canonical orthosteric pocket では強くても、**化学空間や pocket topology がずれると急に弱くなる**ことがあります。PXR も canonical nuclear receptor pocket を持つ一方で、promiscuous かつ reshape しやすいので、この問題に近い側です。citeturn48search22turn29search2

四つ目は、**confidence の誤用**です。DiffDock confidence は内部 reranking に向きますが、AF3 の pose confidence は prospective screen で false positive 分離に弱く、Boltz-2 affinity ほど効きませんでした。pLDDT や ipTM は、少なくとも fragment pose submitter の最終選択には直接使わない方がよいです。実務では、**raw confidence ではなく、複数 seed 間 spread・PoseBusters・gnina・contact agreement** をまとめた不確実性指標へ置き換える方が安全です。citeturn35search9turn34search1turn34search4turn32search8

最後に、この調査自体の限界も明示します。**HA≤16 の fragment-only 定量 slice は近年文献でほとんど報告されていません**。また、Glide、IFD-MD、OpenEye、GOLD など商用法の最新 blind-pose 比較は、同一条件・同一データでの公開値が不足しています。したがって、今回の最終提言は、**公開再現性の高い OSS benchmark を重視した結論**であり、商用ソフトが劣ると言っているのではありません。むしろ柔軟 pocket では、文献的には今なお**ensemble docking + local refinement**が最も堅い、という結論です。citeturn11view2turn37search5turn25search12

## 未解決点と限界

文献を通覧しても、今回の意思決定に直結する未解決点は三つ残ります。第一に、**PoseBusters/Astex を HA≤16 で切った公開表がない**こと。第二に、**apo-soak fragment 参照に対する AF3/Boltz-2 の系統誤差**を、PXR そのもの、あるいは近縁核内受容体で定量した論文が見当たらないこと。第三に、**LDDT-PLI を主指標にした公開比較**が CASP や blind challenge 以外ではまだ少ないことです。したがって、あなたの PXR setting では、文献上の “best method” を追うより、**pose family の多様性を確保してから 1 pose 選抜を洗練する**方が、期待値の高い戦略です。citeturn18search1turn11view2turn34search5
---

## 我々の repo に落とした key takeaways

> 上記の Deep Research 本文をそのまま読みつつ、PXR Track 2 への直接適用部分を
> 整理したもの。本文との差分は「我々の現在の Boltz-2 設定で **既に対応済み** /
> **未対応・要追加** / **次の試行候補**」のラベル付け。

| # | 学び | repo への反映 |
|---:|---|---|
| 1 | co-folding は holo-like / training-similarity bias を持つ。pose RMSD が良くても PB-valid / PLIF が追従しない | LB 結果次第で PoseBusters でフィルタ + gnina rescoring を後段に追加する候補 |
| 2 | 1 pose 提出制約下では best-of-N が必須。confidence-as-selector が機能する | `--diffusion_samples 5` で 5 pose 生成、Boltz の confidence_score 順 (= model_0) で 1 pose 選択 → **既に PR #130 で実装済** |
| 3 | apo-soak 結晶 ≠ holo cocrystal。AI は holo-like 誘導適合を好む | rev2 サーベイで「結晶学コミュニティ寄りの後処理 (PanDDA / qFit-ligand 等)」を発掘予定 |
| 4 | confidence の生値を rank に使うと危険 (AF3/pLDDT は false-positive 分離が弱い、Mac1 large-scale eval) | 将来の re-ranking では Boltz raw confidence を**単独では**使わず、複数 seed spread + PoseBusters + gnina + contact agreement の合成指標へ移行候補 |
| 5 | hard pocket constraint は fragment では危険、soft box まで | 現運用は constraint なし。fragment が極端に散る case (例 x00543-1 / x00261-1 / x01334-1: 5-10Å spread 3 件) の 2nd-stage refine 時のみ広めの box を検討 |
| 6 | unbiased MD + MSM は fragment binding mode 探索に向くが 184 件全件は予算超過 | fragment + 不確実 case のみへの 2nd-stage refiner として保留 (Deep Research の「品質重視」案) |
| 7 | HA≤16 fragment-only の公開 SOTA 表は存在しない | 我々の LB 結果が出れば、LDDT-PLI を HA bin で集計 (公開もできる) |

## 次に書きたい / 決めたいこと

- LB 反映後 (~14:10 JST 想定): id=33 のスコアと公式 baseline 0.4632 / rank-1 0.5012 の差分を見る
- prompt rev2 の Deep Research が返ってきたら本ファイルに追記、または `_v2.md` 別ファイル化
- 5-10Å spread の 3 化合物 (x00543-1, x00261-1, x01334-1) は fragment-rich で binding mode 不確定の典型例 → 2nd pose / pose ensembling の試行候補
- pose re-ranking の合成指標 (PoseBusters + gnina + 5 seed spread) は別 PR で組む

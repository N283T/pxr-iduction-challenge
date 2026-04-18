# Boltz-2 論文サマリ — Affinity予測パート

- **Title**: Boltz-2: Towards Accurate and Efficient Binding Affinity Prediction
- **Authors**: Saro Passaro, Gabriele Corso, Jeremy Wohlwend, Mateo Reveiz, Stephan Thaler, et al. (MIT CSAIL / Jameel Clinic / Valence Labs / Recursion / ETH Zurich)
- **Preprint**: bioRxiv 2025.06.14.659707v1 (posted 2025-06-18)
- **License**: CC-BY 4.0, weights + training + inference コード公開 (https://github.com/jwohlwend/boltz)
- **Source PDF**: `/mnt/c/Users/kitak/Downloads/2025.06.14.659707v1.full.pdf`
- **Raw parse**: `docs/papers/boltz2_raw/{boltz2_full.json, boltz2_full.txt}` (LiteParse v0.1.0, 55 pages, OCR済み)

---

## 0. 3行サマリ

1. **FEP級の精度 × 1000倍速 (~20 GPU sec/ligand)**。FEP+ 4-target subsetで Pearson R=0.66、OpenFE 876-complex で R=0.62、CASP16 blind で R=0.65 (top参加者より全て上)。
2. Affinity moduleはtrunkの潜在表現 + 予測構造の pair representation を PairFormer で処理し、**binding likelihood (分類)** と **affinity value (回帰)** の2つのヘッドを出力。
3. 学習データは ChEMBL/BindingDB/PubChem/BindingDB HTS/CeMM/MIDAS を `log10(μM)` に統一。**Assay内での pairwise 差分を主損失**にして assay-specific confounder を打ち消すのが肝。

---

## 1. Affinity データ (Section 2 + Table 1)

### 1.1 連続値 (regression) データソース
| Source | Type | Supervision | #Binders | #Decoys | #Targets | #Compounds |
|---|---|---|---|---|---|---|
| ChEMBL + BindingDB | optimization | values | 1.2M (1.45M) | 0 | 2k (2.5k) | 600k (700k) |
| PubChem small assays | hit-discovery | both | 10k (13k) | 50k (70k) | 250 (300) | 20k (25k) |

- Ki, Kd, IC50, AC50, EC50, XC50 を **全て `log10(μM)` スケールに統一**し、単一ヘッドで学習。
- **Cheng–Prusoff 補正**は多くのassayで metadata 欠損により不可能 → 同一assay内の pairwise差分で打ち消す方針。
- 信頼度の低いassay・活性値の分散が小さいassayは除外（assay内差分を学ばせるため）。
- **単一protein target** かつ biochemical/functional assayのみ採用。

### 1.2 バイナリ (classification) データソース
| Source | #Binders | #Decoys | #Targets | #Compounds |
|---|---|---|---|---|
| PubChem HTS | 200k (400k) | 1.8M (3.5M) | 300 (500) | 400k (450k) |
| CeMM Fragments | 25k (45k) | 115k (200k) | 1.3k (2.5k) | 400 |
| MIDAS Metabolites | 2k (3.5k) | 20k (35k) | 60 (100) | 400 |
| ChEMBL+BindingDB synthetic decoys | 0 | 1.2M (1.45M) | 2k (2.5k) | 600k (700k) |

- PubChem HTS は ≥100化合物 & ヒット率 <10% のみ。他のassayで定量値があるbinderをfilterして false positive 低減。
- **合成デコイ**: 別ターゲットの既知binderを流用。ただし Tanimoto<0.3 で類似化合物除外 → 本当のchemical space周りをカバー。
- `()` 内は **構造品質フィルタ (iptm ≥ 0.75) 適用前** の数値。つまり iptm<0.75 の複合体は affinity 学習から落としている。

### 1.3 データクリーニングの4原則
1. 高品質assayのみ残す
2. synthetic decoy で bias 軽減
3. iptm ≥ 0.75 で構造品質担保
4. PAINS除外 & heavy atom 数 ≤ 50 の化合物のみ

---

## 2. Affinity Module アーキテクチャ (Section 3 + Appendix B.5)

### 2.1 入出力
- **入力**:
  - Trunk の pair representation `z_trunk` (5 recycling後)
  - Trunk の input features `s_inputs`
  - 予測座標の distogram `D_{i,j}`
  - 予測座標は **5 samples × 200 diffusion steps** の中から **protein-ligand ipTM が最高** のものを選択
- **出力**:
  - `binding_likelihood`: softmax確率 (結合するか)
  - `affinity_value`: 連続値 (≈ `log10(IC50[μM])` 相当)

### 2.2 Algorithm 1 の要点
```
z = LinearNoBias(LayerNorm(z_trunk))
z += LinearNoBias(s_inputs_i) + LinearNoBias(s_inputs_j)
z += PairwiseConditioner(z, one_hot(D_{i,j}))
z += PairFormerModule(z, pair_mask=protein_ligand_mask + intra_ligand_mask)

# protein-ligand + intra-ligand interactionsのみを mean pool (intra-proteinはマスクアウト)
g = MeanPool(z, mask = protein_ligand_mask + intra_ligand_mask * (1 - Id))
g = ReLU(Linear(ReLU(Linear(g))))

binding_likelihood = SoftMax(MLP_cls(g))
affinity_value      = MLP_reg(g)
```

- **重要**: protein-protein (intra-protein) の pair はマスクアウト。protein-ligand と intra-ligand のみを見る。
- Mean pool で scalar に落としてから別々のMLPヘッドへ。

### 2.3 アンサンブル (2モデル)
| Hyperparam | Member 1 | Member 2 |
|---|---|---|
| PairFormer layers | 8 | 4 |
| λ_focal (binder vs decoy weight) | 0.8 | 0.6 |
| Training samples | 55M | 12.5M (early-stopped) |

- **Binary**: 2モデルの平均
- **Regression**: **molecular-weight 補正付きアンサンブル**
  ```
  ŷ = C0 * (y1 + y2) + C1 * MW_binder + C2
  ```
  `C0, C1, C2` はholdout validation setで fit。**MWによる系統誤差を明示的に引く**のがポイント。

### 2.4 学習まわり
- Backbone (trunk) は **gradient detach**。affinity moduleだけ学習。
- 128 × A100, AdamW, weight decay 0.001, lr 1e-4。
- Pocket cropping: 全 ligand tokens + 最近傍 protein token で 256 tokens (最大 protein 200)。iptm≥0.75フィルタにより複合体→タンパク単位になってO(#proteins)でクロップ済み pair を保持。

---

## 3. 損失関数 (Appendix C.2.5)

### 3.1 Affinity value loss (Huber ベース、`δ=0.5`)
- **絶対値ロス** `L_abs(y, ŷ, s)`:
  - `s` = `=`: 通常のHuber
  - `s` = `>`: (下限値)、`ŷ < y` のときのみHuberを適用 (model が下限を下回る予測をしたときだけ罰する)
- **Pairwise差分ロス** `L_dif(y1, y2, ŷ1, ŷ2, s1, s2)`:
  - 両方 `=`: `Huber(y1-y2, ŷ1-ŷ2)`
  - 片方 `>`: 差分の符号が一致しないときだけ罰する (censor-aware)
  - 両方 `>`: 0
  - **同一assay内** での差分なので Cheng-Prusoff 補正項が打ち消される。

### 3.2 Binary loss
- Focal loss (`γ=1`, `α=λ_focal`) で class imbalance に対応

### 3.3 最終目的関数
```
L_total = 0.9 * L_dif + 0.1 * L_abs + L_binary
```
**pairwise差分ロスが90%**。絶対値は補助的。

### 3.4 サンプラー (C.2.4)
- Binary: 1 binder + 4 decoys (同一assay内) = batch size 5
- Regression: 同一assayから B=5 complexes。assayサンプリング確率は **IQR (75%tile - 25%tile) に比例** → **activity cliff のあるassayを優先的に学習**。
- データソース別サンプリング重み:
  | Source | Weight |
  |---|---|
  | ChEMBL+BindingDB values | 0.25 |
  | ChEMBL+BindingDB synthetic decoys | 0.25 |
  | PubChem HTS | 0.44 |
  | PubChem small (both) | 0.005 |
  | PubChem small binary | 0.02 |
  | CeMM Fragments | 0.03 |
  | MIDAS Metabolites | 0.005 |

---

## 4. ベンチマーク結果 (Section 5.3 + Appendix D.2 + E.2)

### 4.1 FEP+ OpenFE 876 complexes (Table 11)
| Method | Type | Pearson R (target avg.) | Kendall τ | Centered MAE (kcal/mol) |
|---|---|---|---|---|
| **Boltz-2** | ML | **0.62** | 0.46 | 0.64 |
| BACPI | ML | 0.29 | 0.19 | 0.85 |
| GAT (ligand-only) | ML | 0.28 | 0.20 | 0.91 |
| OpenFE | physics (6-12 GPU h) | 0.63 | 0.47 | 0.94 |
| FEP+ | physics | 0.72 | 0.53 | 0.64 |

### 4.2 FEP+ 4-target subset: CDK2/TYK2/JNK1/P38 (Table 12)
| Method | Type | Time | Pearson R | Kendall τ | Centered MAE |
|---|---|---|---|---|---|
| **Boltz-2** | ML | **20 GPU sec** | **0.66** | 0.48 | 0.59 |
| BACPI | ML | 0.48 GPU ms | 0.14 | 0.09 | 0.82 |
| GAT | ML | 0.18 GPU ms | 0.40 | 0.28 | 0.71 |
| OpenFE | physics | 6-12 GPU h | 0.66 | 0.51 | 0.75 |
| FEP+ | physics | (commercial) | 0.78 | 0.63 | 0.53 |
| ABFE | physics | >20 GPU h | 0.75 | 0.54 | 0.65 |
| FMO | physics | 2-10 CPU min | 0.55 | 0.38 | - |
| MM/PBSA | physics | 10-15 CPU min | 0.18 | 0.16 | - |
| Chemgauss4 (dock) | physics | 20-30 CPU sec | 0.26 | 0.17 | - |

> **Take-away**: Boltz-2 は OpenFE と同等 (R=0.66)、FEP+ にはまだ届かない (R=0.78) が、**1000倍以上速い**。

### 4.3 CASP16 blind challenge (Table 14)
- Boltz-2 **R=0.65** で **top-6参加者全員を上回る**。No fine-tuning, out-of-the-box。
- 2位 LG016 は R=0.54。

### 4.4 Hit-to-lead validation set (16 assays from BindingDB/ChEMBL, Table 15)
- Boltz-2 R=0.42, centered MAE=0.86 kcal/mol
- Assay間で R=0.056〜0.732 と大きくばらつく (protein classとの系統的関係は**見えず**)

### 4.5 Recursion private benchmarks (Table 16, 8 blind assays)
- Boltz-2 R=0.39 (8 assay平均)、targetごとに R=0.165〜0.634
- centered MAE=1.36 kcal/mol (validation 0.86 より悪化)
- **GPCRなど特定protein classでは物理ベースでも苦戦するのと同様の traits**

### 4.6 Virtual screening (Table 13, MF-PCBA)
- Boltz-2 AP=0.0248, EF@0.5%=18.4, AUROC=0.81
- 先行ML (BACPI, GAT) を ~2倍、docking系を大幅に上回る
- ipTM単体は全くダメ (AUROC=0.57) → **affinity head が専用に学習した価値**

### 4.7 Similarity dependence (Figure 10)
- FEP+ 化合物の **max Tanimoto similarity** を affinity training set に対して計算
- Bin [0.3-0.5] / [0.5-0.65] / [0.65-0.8] / [0.8-1.0] で assay-level Pearson R を見る
- **類似度と性能に強い依存なし** → ある程度 structurally novel でも効く主張

---

## 5. 本コンペ (PXR) への示唆

### 5.1 ポジティブ
- **Affinity value head の出力は `log10(IC50[μM])` 相当**で、**pEC50 の反転 (-1×)** とほぼ同じスケールに近い。PXR の pEC50 (μM単位・log10) との対応は素直。
  - ただし assay-specific な offset があるので **生値より順位 (rank) / 同一assay内差分 に意味がある**。
- Recursion private assay でも target-average R=0.39 → **単一targetなら十分 signal**。
- Similarity耐性あり (Figure 10) → analog compound が多い PXR train/test でも過学習しにくい。
- OpenFE 876-complex で centered MAE=0.64 kcal/mol ≈ **~0.47 log10 units** (RT ≈ 0.6 kcal/mol at 300K)。PXR の pEC50 MAE 0.4〜0.5 相当の信号がある可能性。

### 5.2 注意点
- **ligand_atom_count > 56 の 9化合物は affinity head の training cap 外** (実際は 50 heavy atom でフィルタされている → 56 は Boltz-2 affinity head 内部のカット)。`compound_boltz2.ligand_oversize=TRUE` の化合物は **低信頼として扱う**。
- **`all_predictions_failed` な化合物** (failure list) は predictions 自体なし → 特徴量欠損扱い。
- **iptm<0.75 は affinity学習時に落とされている** → 低iptmのposeに対する affinity value の信頼度は低い。`compound_boltz2.confidence_iptm` でフィルタ or 重み付けしたい。
- **MW補正項が既にbaked in** (`ŷ = C0*(y1+y2) + C1*MW + C2`) → Boltz-2 affinity を feature として使う場合、**別途 MW を concat するのは冗長**。ただし MW とペアで比べて Boltz-2 の incremental value を確認するのは有益。
- Value は `log10(μM)` 上の **IC50-like** 量。PXR の pEC50 (ラベル定義: `-log10(EC50[M])`) と比較するには `pEC50 ≈ 6 - affinity_value` の関係で近似 (μM → M 変換)。

### 5.3 Track 1 特徴量としての使い方候補
1. **affinity_pred_value** (main head + 2 ensemble members) をそのまま LightGBM / TabPFN に concat
2. **binding_likelihood** (分類確率) も concat — 低確率=decoy 的な化合物は regression signal も疑わしい
3. **confidence (ipTM, plddt) を gating feature** として使う (信頼度低いposeの affinity 値をdown-weight)
4. **Pocket-ligand geometry** (`ligand_to_pocket_distance_a` + PoseBustersの physical-plausibility booleans) を sanity signal として使う
5. **affinity_pred_value × all_passed** みたいな interaction term を手動で作る

### 5.4 期待値キャリブレーション
- Recursion private (R=0.39, centered MAE=1.36 kcal/mol) を **PXR で期待できる上限値** と見る (PXRは単一target, ただし Recursion ほど潔白な blind ではない)
- ただしBoltz-2 training に ChEMBL が含まれる → PXR の train_activity に使われた assay が **一部リーク**している可能性 (「PXR 関連 assay ID が ChEMBL に登録されていないか要確認」)
- **リーク検証**: ChEMBL target_id = PXR (NR1I2) の assay が training に含まれているかを issues で追跡

---

## 6. 実装上のtips (boltz CLI 関連)

- **Inference time**: 論文では 20 GPU sec/ligand (H100)。RTX 5080 で ~70 sec/ligand (構造+affinity同時)。Track 2 の full run 4日間と整合。
- **Recycling** 5回 + **diffusion** 200 steps + **5 samples**。`ipTM` 最大の構造を affinity に渡す。論文の pipeline と boltz CLI のデフォルトは一致。
- **Gradient detached**: affinity の loss は trunk に流れない → **PXR で fine-tune する場合は affinity module だけ更新すれば済む** (trunk は凍結)。
- **Assay-conditioning はない**: `(protein, ligand)` の関数としてのみ動く。同じ (protein, ligand) ペアは同じ予測を返す。PXR のように single target なら assay ID condition は不要。

---

## 7. 引用すべき数値 (PR / ドキュメント用)

- "Boltz-2 achieves R=0.66 on the 4-target FEP+ subset, matching OpenFE (R=0.66) and approaching FEP+ (R=0.78), while running ≥1000× faster"
- "On Recursion's 8 blinded private hit-to-lead assays, Boltz-2 reaches target-average R=0.39 and centered MAE=1.36 kcal/mol, compared to BACPI R=0.11 and GAT R=0.16"
- "Boltz-2 outperforms all top-6 CASP16 participants out-of-the-box (R=0.65 vs 2nd place 0.54)"
- Train data: 1.2M binder affinity values from ChEMBL+BindingDB + 2.0M HTS binary labels; filtered by iptm ≥ 0.75

---

## Open questions (次に調査したい項目)

1. **PXR関連のChEMBL assayがBoltz-2のtraining setに含まれているか?** (sequence-level filter は 90% similarity で行っているが、PXR LBD は unique sequence なので filter されていないはず)
2. **Boltz-2 affinity value を pEC50 に直接 regression する線形補正** (`pEC50 = a + b × affinity_value`) を OOF で fit したときの OOF MAE はどれくらいか?
3. **Ensemble member 1 vs member 2** の予測乖離 (`abs(y1 - y2)`) は **不確実性の proxy** になるか? (Track 1 の confidence-aware ensemble で使えるか)
4. **binding_likelihood × affinity_value** という main head のスコアリング式 (Section 5.4 の generative screening で使用) が Track 1 でも MAE を下げるか?

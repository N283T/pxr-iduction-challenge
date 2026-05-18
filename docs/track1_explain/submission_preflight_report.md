# Track 1 submission preflight report

確認日: 2026-05-18 JST

このreportは、Track 1で提出前に使っていたpreflight checkを説明するためのもの。
preflightはpublic LBを当てるoracleではなく、
提出CSVの事故や、すでに悪いと分かった方向への大きな移動を止めるための安全装置である。

実装:

- script: `track1_activity/scripts/submission_preflight.py`
- unit test: `track1_activity/scripts/test_submission_preflight.py`
- output: `track1_activity/analysis/submission_preflight/outputs/<name>/`

## 1. 何のために作ったか

Phase 1後半では、OOF上は良いのにpublic LBで悪化する候補が増えた。
特に id56 の optuna top500 SWAP は、localには強く見えたがpublic LBで大きく悪化した。

この経験から、提出前に以下を必ず見るようにした。

| check | 目的 |
|---|---|
| CSV sanity | test setの行順、SMILES、Molecule Nameが正しいか |
| anchor shift | 信頼済みsubmissionからどれだけ動くか |
| prediction distribution | 予測平均や分散が急に変わっていないか |
| known-bad axis | id56のような悪い方向に揃っていないか |
| largest shifts | どのcompoundが大きく動くか |
| experiment metadata | CSVがDB実験と紐づくか |

preflightは、提出判断を自動化するためではなく、
「このCSVは何を動かしているのか」を短時間で把握するための道具だった。

## 2. Anchorという考え方

preflightは、candidate CSVを単独では見ない。
必ず信頼済みのanchor submissionと比較する。

default anchorは:

```text
track1_activity/submissions/ens_id51_top500_potent46_t40_soft_g35.csv
```

これはid55相当のCSVで、Phase 1終盤のbest public LB anchorとして使っていたもの。
id57系のcandidateを比べるときは、id57 anchorを別途指定することもあった。

考え方は単純で、
「すでに悪くないことが分かっている提出から、どれだけ、どの方向に動いたか」
を見る。

## 3. 出力されるファイル

1回のpreflightで、以下が出る。

| file | 内容 |
|---|---|
| `report.md` | 人が読むsummary。verdict, shift, known-bad axis, reasons |
| `summary.csv` | 1行の機械可読summary |
| `largest_shifts.csv` | anchorから大きく動いた上位50 compounds |
| `bad_axis_correlations.csv` | known-bad axisとの相関とprojection |
| `experiment_rows.csv` | DBの `experiment_summary` に同じCSV pathがあれば保存 |

代表的な実行例:

```bash
pixi run python track1_activity/scripts/submission_preflight.py \
  --candidate track1_activity/submissions/ens_candidate.csv \
  --anchor track1_activity/submissions/ens_id51_top500_potent46_t40_soft_g35.csv \
  --name candidate_vs_id55
```

## 4. 判定は PASS / CAUTION / HOLD

判定は3段階。

| verdict | 意味 |
|---|---|
| `PASS` | anchorからの移動が小さく、明確な危険signalがない |
| `CAUTION` | rank order、scale、known-bad axis alignmentなどに注意 |
| `HOLD` | 提出前に止めて再確認すべき大きな移動がある |

重要なのは、`PASS` は「LBで良くなる」という意味ではないこと。
あくまで「明らかな事故や既知の悪い移動は見えていない」という意味。
実際、PASSしたcandidateでもpublic LBで微妙に悪化することはあった。

逆に `HOLD` はかなり強い警告。
特に、anchorから広く大きく動くcandidateや、id56方向に強く揃うcandidateは、
OOFが良くても提出を控える理由になった。

## 5. 現在の数値ルール

`submission_preflight.py` の `classify_risk()` は、以下のようなruleで判定する。

| rule | threshold | verdictへの影響 | 意味 |
|---|---:|---|---|
| `|shift| > 0.10` のcompound数 | >= 100 | `HOLD` | anchorから大きく動くcompoundが多すぎる |
| `|shift| > 0.20` のcompound数 | >= 25 | `HOLD` | extreme moveが多すぎる |
| max absolute shift | >= 0.35 | `HOLD` | 1 compoundだけでも極端に動きすぎ |
| Spearman vs anchor | < 0.995 | `CAUTION` | rank orderが変わりすぎ |
| std difference vs anchor | >= 0.08 | `CAUTION` | prediction scaleが変わりすぎ |
| known-bad Pearson | >= 0.70 and projection > 0 | `CAUTION` | id56型の悪い方向に揃う |

このthresholdは理論的な最適値ではなく、
Phase 1中に観察した提出失敗を避けるための実務的な警告灯である。

## 6. Known-bad axis

preflightでは、過去にpublic LBで悪化した方向を「bad axis」として保存し、
candidateのshiftがそこに揃っていないかを見る。

現在の代表例:

| bad axis | 定義 | 意味 |
|---|---|---|
| `id56_minus_id55` | id56 candidate - id55 anchor | optuna top500 SWAPで悪化した方向 |
| `id56_minus_id51` | id56 candidate - id51 anchor | id51から見た同じtop500過集中方向 |

計算しているのは、candidateのdeltaとbad-axis deltaの:

- Pearson correlation
- Spearman correlation
- projection係数

projectionが正でPearsonが高いと、
candidateが過去の悪い移動を再現している可能性がある。

## 7. 例: id55はPASS

id55相当の `ens_id51_top500_potent46_t40_soft_g35.csv` を
id51 anchorと比べると、preflightはPASSだった。

| metric | value |
|---|---:|
| mean abs shift | 0.013165 |
| p90 abs shift | 0.038982 |
| max abs shift | 0.158316 |
| `|shift| > 0.05` | 34 |
| `|shift| > 0.10` | 4 |
| `|shift| > 0.20` | 0 |
| projection on `id56_minus_id55` | -0.164066 |

読み方:

- anchorからの移動は小さい。
- 一部compoundは大きく動くが、全体としては局所的。
- id56の悪い方向とはむしろ逆向き。

このため、id55は「小さく、局所的に、known-bad axisを避けて動いたcandidate」
として説明できる。

## 8. 例: id56はHOLD

一方、id56の `ens_swap_optuna_t10_top500_calibrated_importance.csv` を
id55 anchorと比べるとHOLDになる。

| metric | value |
|---|---:|
| mean abs shift | 0.052693 |
| p90 abs shift | 0.111276 |
| max abs shift | 0.356443 |
| `|shift| > 0.05` | 206 |
| `|shift| > 0.10` | 63 |
| `|shift| > 0.20` | 10 |
| projection on `id56_minus_id55` | 1.000000 |

読み方:

- candidateそのものがknown-bad axisなので、projectionは1。
- anchorからの移動が広く大きい。
- max shiftも0.35を超え、`extreme_single_compound_shift` に該当する。

この例は、OOFが良いcandidateでも、
test predictionを大きく動かしすぎると危険という代表例になった。

## 9. preflightの限界

preflightはかなり便利だったが、限界も明確。

| limitation | 説明 |
|---|---|
| LB oracleではない | PASSしてもLB改善は保証しない |
| anchor依存 | anchor自体が悪い場合、その近傍だけを安全と見なしてしまう |
| known-bad依存 | まだ観測していない悪い方向は検出できない |
| small moveには弱い | 小さいが一貫して悪いcalibration driftは見逃すことがある |
| OOF品質は直接見ない | candidate CSVだけのcheckなので、training evidenceとは別に見る必要がある |

したがって、preflightは以下の順で使うのが良い。

```text
OOF / CV / calibrated OOF
  -> family share / prediction correlation
  -> submission preflight
  -> cooldown and submit decision
```

## 10. 説明用の短い言い方

preflightは、提出前にcandidate CSVを信頼済みanchorと比較し、
どれだけ動いたか、行順は正しいか、過去にLBで悪化した方向へ揃っていないかを確認する仕組み。

PASSは「良くなる保証」ではなく、
「少なくとも既知の危険な動きではなさそう」という意味。
HOLDは、OOFが良くても提出前に止めるべき強い警告だった。

この仕組みのおかげで、Phase 1後半のようにOOF/LB mismatchが増えた局面でも、
candidateのリスクを同じ物差しで比較できるようになった。

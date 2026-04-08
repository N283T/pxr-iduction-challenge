# Auxiliary Data Analysis — counter_assay & single_concentration

Issue: [#21](https://github.com/N283T/pxr-iduction-challenge/issues/21)
Reproducer: `pixi run python track1_activity/scripts/eda_auxiliary_data.py`
Figures: `docs/figures/aux_data_*.png`

## TL;DR

1. **single_concentration log2fc @ 8.25e-6 M is nearly a second pEC50 measurement** (r = 0.72 with train pEC50). This is the strongest auxiliary signal by far.
2. **counter_assay pEC50 is nearly independent of train pEC50** (r = 0.11) — a clean off-target signal, ideal for multi-task learning.
3. Combining both proxies, only **3.2% of train** compounds are *doubly* suspicious (counter > train AND flat single-conc). These are high-confidence denoising targets, with mean pEC50 = 2.32.
4. Counter-assay alone over-flags (14.8% sel<0); most of those still show real PXR signal in single-conc — don't downweight from counter-assay alone.

## Data overview

| Table | Rows | Unique compounds | Overlap w/ train | Overlap w/ test |
|---|---:|---:|---:|---:|
| train_activity | 4,140 | 4,140 | — | 0 |
| counter_assay | 2,860 | 2,860 | 2,860 (64% of train w/ valid pEC50) | 0 |
| single_concentration | 21,014 | 10,875 | 2,392 (57.8% of train) | 0 |
| test_activity | 513 | 513 | 0 | — |

**Test has zero overlap** with both auxiliary tables, so these cannot be used as direct test features. They are only useful for improving training quality or as auxiliary targets.

## Counter-assay findings

- train pEC50: 4.74 ± 0.88  | counter pEC50: 3.13 ± 1.12
- **Pearson r(train, counter) = 0.108** → nearly independent signal
- Selectivity = train − counter: mean 1.61, median 1.72
- Mean pEC50 monotonically increases with selectivity bin (3.84 → 5.15) — **low-pEC50 compounds are enriched for non-specific activity**, consistent with noise from the assay floor.

### Selectivity bins (n=2,648)

| Bin | Count | % | Mean pEC50 |
|---|---:|---:|---:|
| sel < 0 | 391 | 14.8% | 3.84 |
| 0 ≤ sel < 0.5 | 247 |  9.3% | 4.46 |
| 0.5 ≤ sel < 1 | 282 | 10.7% | 4.52 |
| 1 ≤ sel < 2 | 594 | 22.4% | 4.76 |
| sel ≥ 2 | 1,129 | 42.7% | 5.15 |

## Single-concentration findings

- 4 concentrations, but only 2 are well-populated: **8.25e-6 M (10,753 compounds)** and **3.30e-5 M (9,527 compounds)**
- Most compounds have 2 concentrations measured (2,053 of 2,392 train overlap), 264 have 3
- **Pearson r(pEC50, log2fc @ 8.25e-6 M) = 0.724** ← strongest proxy in this challenge's aux data
- r(pEC50, log2fc @ 3.30e-5 M) = 0.496 (likely saturation at high conc)
- Aggregate: r(pEC50, mean_log2fc) = 0.679, r(pEC50, max_log2fc) = 0.608

### Mean pEC50 by log2fc bin @ 8.25e-6 M (n=2,374)

| log2fc bin | Count | Mean pEC50 |
|---|---:|---:|
| ≤ −0.5 | 12 | 2.70 |
| −0.5 to 0 | 113 | 2.58 |
| 0 to 0.5 | 378 | 4.40 |
| 0.5 to 1 | 1,107 | 4.85 |
| 1 to 2 | 738 | 5.37 |
| > 2 | 26 | 5.54 |

Sharp monotonic relationship. log2fc ≤ 0 → essentially inactive (pEC50 ≈ 2.6).

## Triangulation (counter × single-conc × train)

On the overlap n = 2,176:

|  | pec50 | counter_pec50 | log2fc @ 8.25e-6 | selectivity |
|---|---:|---:|---:|---:|
| **pec50** | 1.000 | 0.080 | **0.719** | 0.518 |
| counter_pec50 | 0.080 | 1.000 | 0.191 | −0.811 |
| log2fc_hi | 0.719 | 0.191 | 1.000 | 0.258 |
| selectivity | 0.518 | −0.811 | 0.258 | 1.000 |

### "Doubly suspicious" filter

Compounds where selectivity < 0 AND log2fc @ 8.25e-6 < 0.3:

- **n = 69 (3.2% of overlap)**
- Mean pEC50 = **2.32** ± 0.90 (far below global mean 4.74)
- These are prime candidates for downweighting or removal.

### Mean single-conc log2fc by selectivity bin

| sel bin | n | mean log2fc @ 8.25e-6 |
|---|---:|---:|
| < 0 | 294 | 0.64 |
| 0 – 0.5 | 195 | 0.75 |
| 0.5 – 1 | 233 | 0.70 |
| 1 – 2 | 498 | 0.73 |
| > 2 | 956 | 0.89 |

Only a modest spread. **A compound flagged as "non-specific" by counter-assay alone (sel<0) still averages log2fc ≈ 0.64** — well above zero. Counter-assay alone over-flags artifacts; require single-conc agreement.

## Strategic implications for modeling

### High-impact ideas (ranked)

1. **Pseudo-label semi-supervised training** — For the ~8,483 single-conc compounds with no train pEC50, fit `pEC50 ~ log2fc @ 8.25e-6 + log2fc @ 3.30e-5` on the 2,374 overlap, then predict pseudo-pEC50 for the extras. Train main model with those as weak labels (lower sample weight, e.g. 0.3). This nearly quadruples the effective training set.

2. **Multi-task learning (counter_pec50 as aux target)** — Because r(train, counter) ≈ 0.11, joint prediction forces the shared representation to separate PXR-specific from generic nuclear receptor signal. Candidates: ChemProp, AttentiveFP, MoLFormer head. For the 1,492 train compounds without counter data, use masked loss.

3. **Label denoising via triangulation** — Drop or downweight the 69 doubly-suspicious compounds (sel<0 AND log2fc<0.3). Low risk, small sample, likely moves noise out of the low-pEC50 tail.

4. **Sample weighting by single-conc agreement** — `weight = 1 - |pred_from_log2fc - pEC50| / sigma`. Compounds where the two signals disagree are noisier labels.

### Lower priority

- Dose-response curve fitting (2-point Hill fit) — most compounds have only 2 concentrations, so slope is unstable
- Counter-assay as direct feature — 1,492 train compounds lack it, and test has zero
- Pretraining on single-conc data — r=0.72 at one concentration means this is not a weak signal anymore; direct label use dominates

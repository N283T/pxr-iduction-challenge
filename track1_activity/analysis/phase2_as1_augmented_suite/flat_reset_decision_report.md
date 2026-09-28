# Phase 2 Flat Reset Decision Note

Date: 2026-06-27

This note resets the late Phase 2 decision using evidence tiers rather than
submission pressure. The key distinction is that AS1-augmented model-only
metrics are answer-check diagnostics, not honest generalization estimates.

## Evidence Tiers

Strongest evidence:

- Pre-AS1 AS1 replay: old predictions evaluated on released AS1 labels.
- Train+AS1 cross-fit OOF: each row is predicted out of fold.
- AS2 movement vs the trusted current/id55 anchors.

Weak diagnostic evidence:

- AS1-augmented model-only AS1 replay, because AS1 labels are in the fit pool.
- AS1-tuned gate thresholds and shifts.

## Boltz Evidence

Pre-AS1 AS1 replay does not support Boltz as a strong standalone AS2 reset:

| model | AS1 MAE | Spearman |
|---|---:|---:|
| id55 anchor | 0.406566 | 0.848762 |
| old top500 seed10 | 0.421414 | 0.833405 |
| old pooled_boltz | 0.487915 | 0.766702 |
| old pooled_boltz_allpairs | 0.490475 | 0.773127 |

AS1-augmented answer-check strongly favors Boltz, but this is leaky:

| model | AS1 MAE | AS2 p90 shift vs id55 |
|---|---:|---:|
| pooled_boltz | 0.076706 | 0.584846 |
| cheme_seed10_top500 | 0.099926 | 0.248937 |
| pooled_boltz_allpairs | 0.100723 | 0.599915 |
| kermt | 0.118605 | 0.394096 |

Train+AS1 OOF proxy also does not support Boltz as a large direct component:

| member | OOF/source-AS1 read |
|---|---|
| top500 optuna/top500 family | best proxy, all MAE about 0.388-0.396 |
| kermt | middle, source-AS1 MAE about 0.473 |
| pooled_boltz/allpairs | weak, source-AS1 MAE about 0.485-0.497 |

Conclusion: Boltz is useful as a mechanism/residual axis only. Direct reset or
large replacement is not supported by honest evidence.

## Classifier Gate Probe

A fresh train+AS1 cross-fit high-activity classifier probe was run across
pooled_boltz, pooled_boltz_allpairs, kermt, chemprop_embed, and top500 features.
The classifiers can rank high activity, but converting probabilities into pEC50
shifts barely improves the current proxy.

Best cross-fit AP examples:

| feature | target | AS1 AUC | AS1 AP |
|---|---:|---:|---:|
| chemprop_embed | >=5.5 | 0.894268 | 0.672425 |
| kermt | >=5.5 | 0.894268 | 0.613605 |
| pooled_boltz_allpairs | >=5.5 | 0.856804 | 0.560952 |
| pooled_boltz | >=5.5 | 0.848341 | 0.528356 |

Best gate conversion only improved AS1 proxy by about 0.0018 MAE. A new
classifier gate is therefore not worth using as a final correction.

## Candidate Comparison

All listed candidates preserve AS1 label fill and were preflighted against the
current submitted CSV:

| candidate | preflight | mean abs shift | p90 shift | max shift |
|---|---|---:|---:|---:|
| current final | anchor | 0 | 0 | 0 |
| current + consensus b0.2 | PASS | 0.008088 | 0.035488 | 0.085424 |
| flat augens top90/boltz10 a0.6 | PASS | 0.011889 | 0.035840 | 0.099518 |
| current + consensus b0.3 | PASS | 0.012132 | 0.053232 | 0.128137 |
| capped extreme c0.1 | PASS | 0.040419 | 0.100000 | 0.100000 |

AS1 proxy gains are leaky/diagnostic, but relative shift efficiency was:

- current + consensus b0.2: about 0.040 AS1 proxy gain with very small movement.
- flat augens top90/boltz10 a0.6: about 0.065 AS1 proxy gain, also small
  movement, but uses leaky AS1-aug model selection more directly.
- current + consensus b0.3: about 0.061 AS1 proxy gain, slightly larger
  movement.
- capped extreme c0.1: about 0.050 AS1 proxy gain but touches many more rows and
  is more Boltz-directional.

## Recommendation

The unbiased recommendation is not to make a large Boltz move. The best
evidence-supported default remains the current final candidate.

If choosing one incremental research candidate despite the weak validation
surface, prefer `current + consensus_boltz_top500 b0.2` over direct Boltz or
extreme gates. It adds the smallest controlled consensus movement, passes
preflight, avoids resetting the ensemble around leaky AS1-aug Boltz, and does
not rely on a newly tuned classifier gate.

The `flat augens top90/boltz10 a0.6` candidate is an interesting backup: it has
better AS1 proxy gain with similarly small preflight movement, but its support
comes more directly from AS1-leaky retraining and should be treated as more
speculative than `current + consensus b0.2`.

# Phase 2 flat reset decision

Date: 2026-06-27 JST.

This note summarizes the late Phase 2 reset after AS1-augmented top500, Boltz,
pairrank gates, and classifier gates were all reconsidered without treating the
current final submission as automatically wrong or automatically safe.

## Metric hygiene

The main correction from this audit is to separate three evidence types:

- **Pre-AS1 AS1 replay**: already-built test predictions scored against the
  later released AS1 labels. This is the cleanest answer to "did the old model
  generalize to AS1?"
- **Train+AS1 cross-fit OOF**: a development proxy where each labeled row is
  predicted out of fold after adding AS1 to the labeled pool.
- **AS1-augmented model-only answer-checks**: final heads are fitted on
  train+AS1 and then scored on AS1. These are leaky diagnostics, not
  generalization estimates.

AS2 shift metrics are useful stability guardrails, but they do not prove blind
accuracy.

## Boltz evidence

The pre-AS1 AS1 replay did not support Boltz as a strong standalone model:

| model | AS1 MAE | Spearman |
|---|---:|---:|
| `id55_anchor` | 0.406566 | 0.848762 |
| `old_top500_seed10` | 0.421414 | 0.833405 |
| `old_pooled_boltz` | 0.487915 | 0.766702 |
| `old_pooled_boltz_allpairs` | 0.490475 | 0.773127 |

AS1-augmented model-only answer-checks look much better, but they are leaky:

| model | AS1 MAE | AS2 p90 shift vs id55 |
|---|---:|---:|
| `pooled_boltz` | 0.076706 | 0.584846 |
| `cheme_seed10_top500` | 0.099926 | 0.248937 |
| `pooled_boltz_allpairs` | 0.100723 | 0.599915 |
| `kermt` | 0.118605 | 0.394096 |

The train+AS1 OOF scoreboard also does not support a large direct Boltz move:
top500 is the best proxy family, KERMT is middle, and pooled Boltz/allpairs are
near the bottom. The old Caruana weights already kept pooled Boltz small
(`0.0456` and `0.0350` for the two pooled Boltz members).

Conclusion: Boltz is a useful residual/mechanism axis, but not a trustworthy
direct reset for AS2.

## Classifier gate check

A fresh cross-fit high-activity classifier probe was run over pooled Boltz,
pooled Boltz allpairs, KERMT, Chemprop embeddings, and top500 features. The
classifiers can rank high activity, but converting probabilities into pEC50
shifts barely helped the current proxy.

Best examples:

| feature | target | AS1 AUC | AS1 AP |
|---|---:|---:|---:|
| `chemprop_embed` | `>=5.5` | 0.894268 | 0.672425 |
| `kermt` | `>=5.5` | 0.894268 | 0.613605 |
| `pooled_boltz_allpairs` | `>=5.5` | 0.856804 | 0.560952 |
| `pooled_boltz` | `>=5.5` | 0.848341 | 0.528356 |

The best gate conversion improved current AS1 proxy MAE by only about `0.0018`.
A new classifier gate was therefore not adopted.

## Candidate comparison

All candidates below preserve AS1 label fill and were preflighted against the
current final CSV:

| candidate | preflight | mean abs shift | p90 shift | max shift |
|---|---|---:|---:|---:|
| current final | anchor | 0.000000 | 0.000000 | 0.000000 |
| `current + consensus_boltz_top500 b0.2` | PASS | 0.008088 | 0.035488 | 0.085424 |
| `flat_augens top90/boltz10 a0.6` | PASS | 0.011889 | 0.035840 | 0.099518 |
| `current + consensus_boltz_top500 b0.3` | PASS | 0.012132 | 0.053232 | 0.128137 |
| capped extreme Boltz `c0.1` | PASS | 0.040419 | 0.100000 | 0.100000 |

The AS1 proxy gains are diagnostic only. Among the small-movement candidates,
`current + consensus_boltz_top500 b0.2` has the cleanest risk/reward profile
because it only makes a small consensus move from the current final candidate.

`flat_augens top90/boltz10 a0.6` is an interesting backup, but it is more
directly supported by the leaky AS1-augmented model-only signal. The capped
extreme Boltz gate touches too many rows for the amount of independent evidence
available.

## Decision

The unbiased default is to keep the current final candidate. If an incremental
research candidate is needed, prefer `current + consensus_boltz_top500 b0.2`.

Avoid:

- direct pooled Boltz replacement,
- large Boltz blends,
- learned residual heads selected on AS1 proxy,
- uncapped or broad extreme gates,
- a new classifier gate as a final pEC50 shifter.

Generated supporting artifacts are intentionally left out of git because they
are analysis/submission outputs, not source.

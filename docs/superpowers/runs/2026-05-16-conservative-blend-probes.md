# Conservative Blend Probes

Date: 2026-05-16

## Goal

Check whether basic competition-style blend improvements can improve the current
Track 1 ensemble without finding a new molecular feature axis.

The probe uses only existing member OOF predictions and test submission vectors.
The main candidate family is an MAE-optimized simplex blend regularized toward
the latest DB `ens_caruana_bag20` weights.

## Script

```bash
pixi run python track1_activity/scripts/run_conservative_blend_probes.py
```

The script loads the latest `ens_caruana_bag20` weight map from the database,
then evaluates:

- simplex MAE blends with nonnegative weights and sum-to-one constraint
- L2 penalty toward the current Caruana weights
- optional max-weight caps
- prediction-only Ridge, positive linear, and Huber stackers
- rank-average diagnostics with linear/isotonic calibration

Primary diagnostic is outer 5-fold UMAP CV, where each fold learns blend weights
on four folds of member OOF predictions and evaluates on the held-out fold. This
is stricter than fitting weights once on all OOF rows.

## Results

Latest DB `ens_caruana_bag20` raw anchor:

- OOF MAE: `0.391444`
- Spearman: `0.849031`
- main weights:
  - optuna trial10 top500: `0.3092`
  - optuna trial10 default: `0.2879`
  - chemprop pretrain embed: `0.1515`
  - KERMT pretrain embed: `0.1107`

Outer-CV summary:

| method | OOF MAE | delta vs raw anchor | test p95 abs shift vs raw anchor |
|---|---:|---:|---:|
| simplex MAE, no anchor penalty | 0.381649 | -0.009795 | 0.210947 |
| simplex MAE, L2=0.01 | 0.382708 | -0.008736 | 0.164546 |
| simplex MAE, L2=0.03 | 0.384066 | -0.007378 | 0.122799 |
| simplex MAE, L2=0.1 | 0.386345 | -0.005099 | 0.074717 |
| simplex MAE, L2=0.3 | 0.388714 | -0.002730 | 0.039292 |
| simplex MAE, L2=1.0 | 0.390432 | -0.001012 | 0.014480 |
| current raw Caruana | 0.391444 | 0.000000 | 0.000000 |
| rank anchor isotonic | 0.396142 | +0.004698 | 0.229593 |
| rank anchor linear | 0.464177 | +0.072733 | 0.473846 |

Unregularized simplex optimization is too aggressive. The full-fit no-penalty
weights put `0.8672` on the optuna trial10 top500 member, which is exactly the
kind of destructive reallocation that has failed on LB before.

Conservative L2 settings (`0.1`, `0.3`, `1.0`) looked locally interesting
against the raw Caruana anchor, so CSVs were written and checked by
`submission_preflight.py` against the current trusted LB anchor
`ens_id51_top500_potent46_t40_soft_g35.csv` (id55).

All four checked candidates were `HOLD`:

| candidate | verdict | key issue |
|---|---|---|
| L2=0.1 | HOLD | large shift, aligned with known bad `id56-id55` axis |
| L2=0.3 | HOLD | large shift, aligned with known bad `id56-id55` axis |
| L2=1.0 | HOLD | large shift, aligned with known bad `id56-id55` axis |
| L2=0.03 with cap 0.35 | HOLD | large shift, aligned with known bad `id56-id55` axis |

Example for L2=1.0 vs id55:

- Pearson vs id55: `0.995388`
- mean abs shift: `0.092442`
- p90 abs shift: `0.166788`
- max abs shift: `0.353051`
- projection on known bad `id56_minus_id55`: `0.934881`

## Interpretation

The basic blend work did find a real OOF improvement over the raw Caruana
weights, even under outer CV. However, the improvement moves predictions toward
the already-tested id56 direction, which was worse on LB than the id55
potent46-soft correction.

This is useful as a negative control: the raw OOF objective still wants to
increase top500-family weight, but the public LB evidence says that direction is
not safe.

## Decision

Do not submit these conservative blend candidates as-is.

Keep the script because it is a good low-cost diagnostic for future phase-2
reweighting, especially after Phase 1 labels reveal which ensemble directions
actually worked. Before Phase 1 ends, current id55/id57-style anchors remain safer
than OOF-optimized reblends around raw Caruana.

# Track 1 Next-Step Recommendations

Generated from compound-level shift analysis on 2026-05-04.

## High-Confidence Takeaways

1. The current best id43/id47 is not a new signal source. It is a small,
   rank-preserving movement along the baseline9 -> family-meta axis:
   mean absolute shift 0.020 pEC50, Pearson r 0.9994 vs id32.

2. The successful direction is mostly range compression:
   low baseline-prediction quintiles move up, high baseline-prediction
   quintiles move down. This matches the earlier "good ordering, wrong
   scale" diagnosis.

3. Recent regressions are not caused by being too timid:
   id46 region routing moved much more than id43 (mean abs shift 0.047,
   p90 0.102, Pearson 0.9969) and lost 0.0013 MAE vs id32. Larger movement
   without a real new axis is harmful.

4. ADMET-AI and anchor residual changes are deceptively tiny and highly
   correlated, but still lost on LB. That argues against spending more
   cooldowns on borderline OOF gains unless the shift is on a previously
   LB-validated axis.

## Submission Candidates

The only candidate family with real LB interpolation support is the
baseline9 -> family-meta axis. A quadratic fit through alpha 0.0, 0.5, 1.0
predicts optimum alpha 0.343:

- `outputs/meta_axis_candidates/ens_meta_axis_a343.csv`
- predicted MAE 0.40739 vs id43 0.40748
- expected upside ~0.0001 MAE, below normal LB noise

This is a low-upside calibration A/B, not a path to rank 1.

## Recommended Work

1. Do not submit another full new member ADD unless it passes:
   Gate 1 strong single, Gate 2 residual r <= 0.85 before GPU-scale work,
   Gate 3 nonzero Caruana weight, and test shift no larger than id43 unless
   the axis has an LB anchor.

2. If using a cooldown for defense, submit alpha 0.343 or 0.35 on the
   meta axis. Prefer 0.343 if treating the quadratic fit literally; prefer
   0.35 if using a round, human-readable candidate.

3. For offense, look for a truly external signal source rather than another
   foundation encoder permutation. The recent failures show that r > 0.99
   prediction variants are exhausted and r ~0.997 variants can still regress.

4. Before training anything expensive, create the cheapest possible OOF/test
   probe and append it to this analysis. The first question should be:
   "does its prediction delta look unlike id42/id46, and is residual r <= 0.85?"

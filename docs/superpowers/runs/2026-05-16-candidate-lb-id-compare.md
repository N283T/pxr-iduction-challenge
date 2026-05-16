# Candidate LB ID Direction Compare

Date: 2026-05-16

Goal: compare current candidate movements against historical LB submission IDs
before spending the next cooldown.

## Setup

- Script:
  `track1_activity/analysis/compound_level_lb/compare_candidate_to_lb_ids.py`
- Outputs:
  `track1_activity/analysis/compound_level_lb/outputs/candidate_lb_id_compare/`
- Base:
  `track1_activity/submissions/ens_id51_top500_potent46_t40_soft_g50.csv`
  (id57)
- Compared candidates:
  - conservative rank2:
    `track1_activity/submissions/ens_id57_high_activity_lift_rank2.csv`
  - bold rank5:
    `track1_activity/submissions/ens_id57_bold_lf_calib_rank5.csv`

The comparison uses two views:

1. Direct similarity to each historical ID's `id - id57` vector.
2. Pairwise historical directions `to_id - from_id`, split into directions that
   improved or worsened LB MAE.

## Summary

| candidate | mean abs shift | p90 shift | max shift | direct id58 corr | id55→id58 corr | id55→id58 projection |
|---|---:|---:|---:|---:|---:|---:|
| conservative rank2 | 0.007616 | 0.024373 | 0.030000 | 0.272645 | 0.374446 | 0.347103 |
| bold rank5 | 0.016824 | 0.058290 | 0.080000 | 0.251492 | 0.359770 | 0.798698 |

Both candidates have some overlap with the recent id55→id58 bad direction, but
the bold candidate projects much more strongly onto it. This is the clearest
reason not to replace the conservative rank2 candidate with bold rank5.

## Interpretation

The direct-ID view is not very discriminative because many historical rows share
identical or near-identical CSV paths. Pairwise directions are more useful.

The conservative rank2 candidate remains safer:

- smaller movement by every shift metric
- lower projection onto id55→id58
- all preflight reports pass
- enough pseudo-public high-y correction to test the underprediction hypothesis

The bold rank5 candidate has stronger local proxy evidence, but its projection
onto id55→id58 is too large after id58 just regressed on LB.

Recommendation unchanged:

1. Submit `ens_id57_high_activity_lift_rank2.csv`.
2. Keep `ens_id57_bold_lf_calib_rank5.csv` as a diagnostic only.
3. Use `ens_id57_high_activity_lift_lfmean_g020.csv` only if we want to make the
   move even smaller.

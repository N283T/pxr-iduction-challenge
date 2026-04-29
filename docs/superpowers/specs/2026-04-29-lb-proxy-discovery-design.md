# LB-proxy Metric Discovery

Date: 2026-04-29
Status: Initial finding (calibrated OOF MAE candidate, N=2 LB cases — needs more data)
Owner: N283T

## Background

Track 1 OOF→LB reverse amplification documented in 4 cases (id=38, 40, 41, 44).
Codex consult 2026-04-29 (thread 019dd7a2-...): suggested the issue may not
be the model pool but the **judgment metric** — i.e. which OOF/test
diagnostic predicts LB outcomes.

Codex specifically recommended retroactive testing of candidate metrics
against historical LB outcomes, prioritising:
- calibrated OOF MAE (apply production affine before MAE)
- importance-weighted calibrated OOF MAE
- top-shift subset MAE (NN-Tanimoto >= 0.30 to potent-46)
- subset Spearman/Kendall
- fold worst-case MAE
- delta-to-baseline on weighted subset

## Data

- 38 LB submissions in `lb_submissions` (track=activity, lb_mae present)
- BUT only 12 unique submission CSV file paths (most overwrite each other)
- Only `ens_caruana_bag20` retains weights in DB (1 row, latest only); older
  rows overwritten with `on_conflict_replace=True`

→ Retro analysis limited to a small set with full reconstruction:

| case | LB known | OOF reconstructable |
|---|---|---|
| `id43_hybrid_meta_baseline_5050` | 0.4075 / Sp 0.847 (rank 2 best) | partial (family_meta member missing, used base9pool) |
| `id44_anchor_residual` | 0.4090 / Sp 0.845 (rank 3 reverse) | yes (re-run today's script) |

Plus today's bakeoff variants (LB unknown, but predicted outcome):
- `proximity_v2` (degenerate, no submit)
- `admet_standalone` (caruana weight 0.005, no submit)
- `admet_swap_top500` (tied, no submit)
- `admet_add_top500` (would LB regress per family-share memory)

## Method

Implemented `track1_activity/scripts/lb_proxy_metric_battery.py` computing
8 candidate metrics on each case's reconstructed OOF:

| metric | description |
|---|---|
| M1 | raw OOF MAE (current default gate) |
| M2 | **calibrated OOF MAE** (apply production importance affine before MAE) |
| M3 | importance-weighted OOF MAE (raw) |
| M4 | **importance-weighted CALIBRATED OOF MAE** |
| M5 | top-shift subset OOF MAE (NN-Tanimoto >= 0.30, n_train ~223) |
| M6 | top-shift subset Spearman |
| M7 | fold worst-case raw OOF MAE |
| M8 | M4 delta vs base9pool |

## Finding

**M2 (calibrated OOF MAE) and M4 (importance-weighted calibrated OOF MAE)
correctly distinguish the two LB-known cases**, while M1 (current gate) and
all raw-OOF metrics fail.

| metric | id43 (LB best) | id44 (LB regress) | Δ | LB-correct? |
|---|---|---|---|---|
| M1 raw OOF MAE | 0.3971 | **0.3956** | -0.0015 | **✗ favored id44 (wrong)** |
| **M2 calibrated** | 0.3962 | **0.4069** | **+0.0108** | ✓ correct |
| M3 imp-weighted raw | 0.4080 | 0.4019 | -0.0061 | ✗ |
| **M4 iw-calibrated** | 0.4006 | **0.4063** | **+0.0057** | ✓ correct |
| M5 shift_subset_mae | 0.4800 | 0.4706 | -0.0095 | ✗ |
| M6 shift_subset_sp | 0.7530 | 0.7439 | -0.0091 | ✓ correct (Sp) |
| M7 fold worst | 0.4198 | 0.4195 | -0.0002 | ✗ (within noise) |

### Mechanism

LB scoring evaluates the **calibrated** test predictions. Current gate (M1)
measures raw OOF MAE, ignoring what the production importance affine will
do to predictions. The id44 anchor residual lowered raw OOF MAE by 0.0015
(visible) but the residual term was *un-calibrated relative to base*, so
applying the production affine on (base + residual) shifted predictions
in a direction that increased post-calibration MAE by +0.0108. This is
the OOF→LB reverse mechanism.

In short: **the residual was fit to optimise raw MAE, but the production
calibrator partially undoes residual contribution since it was fit for
base alone**. M2 (calibrated MAE) measures the actual quantity that
shows up at LB.

### Today's 4-strike sanity check (M2 Δ vs base9pool, MAE-direction)

| case | M2 Δ | Verdict |
|---|---|---|
| id43_hybrid (LB best) | 0.0000 | endorse ✓ |
| id44_anchor_residual (LB regress) | **+0.0108** | reject ✓ (would have caught) |
| proximity_v2 (degenerate) | +0.0151 | reject ✓ |
| admet_standalone | +0.1210 | reject ✓ |
| admet_swap_top500 | -0.0003 | tied (no decision) ✓ |
| admet_add_top500 | -0.0020 / M4 -0.0043 | M2 endorses, but family share trap |

M2 catches all the bad cases. The only exception is `admet_add_top500`
which has positive M2 signal but the family-share constraint (separate
memory `project_family_share_lb_u_curve`) flags it.

## Limitations

1. **N=2 LB-known cases**. The id43 vs id44 ordering is only a 2-point test.
   Statistical confidence on a single-pair ordering is low.
2. **id43 OOF reconstruction was approximate** (family_meta member OOF
   missing in DB). The "id43 ≈ base9pool OOF" used as a proxy may slightly
   distort the comparison.
3. **Family-share trap is not captured** by M2/M4. Need a multi-gate
   approach.
4. Future submissions will accumulate more LB-known data points; the
   metric should be re-validated as N grows.

## Recommendation (initial; awaits more data)

**Replace M1 (raw OOF MAE) with M2 (calibrated OOF MAE) as the primary
submit gate**, with these thresholds:

- M2 Δ ≤ -0.003 vs base → endorse
- M2 Δ ≥ +0.005 → reject
- Δ in [-0.003, +0.005] → noise band; supplement with M4, family-share gate

For added robustness, augment with:
- **family-share gate**: chemprop family weight 0.65-0.80 (per
  `project_family_share_lb_u_curve`)
- **M4 (importance-weighted calibrated)**: cross-validates M2; if M2 says
  endorse but M4 says reject, hold

## Deliverables

- `track1_activity/scripts/lb_proxy_metric_battery.py` (new, 491 lines)
- `docs/superpowers/runs/2026-04-29-lb-proxy-battery.log`
- This spec

## Open questions

- N more historical submissions can be added by **regenerating their OOF**
  (re-running the relevant pool/calibrator combos) — would need to walk
  back the ENSEMBLE_MODELS history from `run_ensemble.py` git log.
- Calibrated metric per submission requires the **specific calibrator used
  at that time** (production importance affine has been stable since 2026-04-21,
  but earlier submissions used different calibrators).
- Future submissions with this gate adopted: track which submissions M2
  endorsed → did they LB-improve? Add to evidence pile.

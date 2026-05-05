# Track 1 Selectivity Axis Design

## Goal

Build a new internal-only Track 1 axis that uses counter-assay and assay-shape
signals to identify PXR-specific versus non-specific activity regimes, then
produce small gated corrections to the current id48 anchor.

## Constraints

- No external data.
- Do not modify the production ensemble scripts.
- Do not use global residual correction.
- Keep candidate test shifts small: target mean absolute shift <= 0.02 pEC50.
- Treat any OOF gain below 0.002 MAE as weak unless the correction is very small
  and directionally distinct from the failed id50 axis.

## Approach

Create a standalone analysis script under
`track1_activity/analysis/selectivity_axis/`. The script will:

1. Load train/test compounds, narrow RDKit descriptors, and Morgan fingerprints.
2. Derive auxiliary train labels from counter-assay:
   - `counter_active`: counter pEC50 is present.
   - `nonselective`: counter pEC50 is close to PXR pEC50.
   - `selectivity_delta`: PXR pEC50 minus counter pEC50 for active counter rows.
3. Train cross-fit auxiliary predictors using only test-computable molecular
   features.
4. Fit a low-dimensional gated residual correction against the id48 OOF anchor.
5. Materialize candidate CSVs only if they obey shift caps.

## Safety Gates

- The correction model may only use auxiliary predictions and the id48 anchor,
  not arbitrary high-dimensional molecular features.
- Candidate corrections are clipped per compound.
- Outputs must include OOF diagnostics, test shift diagnostics, and a markdown
  report.
- KERMT/global blend is out of scope; it remains a possible later gated feature
  but is not part of this first selectivity-axis pass.

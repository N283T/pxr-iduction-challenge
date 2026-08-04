# Curated lightweight data

This directory contains a small, Git-tracked subset of the local runtime data.
It is intended to make analysis and presentation work possible from a fresh
clone without transferring the multi-gigabyte embedding and structure stores.

## Included data

### OpenADMET source data

The following Parquet files are format-converted copies of the corresponding
files in
[`openadmet/pxr-challenge-train-test`](https://huggingface.co/datasets/openadmet/pxr-challenge-train-test):

- `default_train.parquet`
- `default_test.parquet`
- `counter_assay_train.parquet`
- `single_concentration_train.parquet`
- `crudes_htchem_train.parquet`
- `semi_pure_htchem_train.parquet`
- `phase_1_unblinded_test.parquet`
- `phase_2_unblinded_test.parquet`

The upstream dataset is marked Apache-2.0. A copy of that license is stored in
`LICENSE-OPENADMET-APACHE-2.0.txt`. Run `pixi run python download_data.py` to
refresh these files directly from the upstream CSV files.

### Derived tables

- `train_activity_db.parquet` and `test_activity_db.parquet`: deterministic
  exports of the local database activity rows, including `compound_id` and the
  SMILES used by the feature tables.
- `train_rdkit_descriptors_full.parquet` and
  `test_rdkit_descriptors_full.parquet`: 217 RDKit 2D descriptors plus
  `compound_id`, in database `train_activity.id` / `test_activity.id` order.
- Root-level `*predictions*.parquet`, `emax_predictions*.parquet`, and
  `pseudo_labels.parquet`: compact model outputs used by the Track 1 workflow.
- `eda_cv_prep/` and selected `eda_redo/` files: compact analysis tables useful
  for plots and presentation material.
- `track1_explain/dataset_report/all_compound_morgan_umap.parquet`: plotting
  coordinates for the public dataset report.

The full embedding matrices, model checkpoints, Boltz/3D artifacts, OpenEye
outputs, ChEMBL mirror, database files, caches, backups, and smoke-test outputs
remain intentionally excluded.

## Integrity and provenance

`MANIFEST.sha256` records every Git-tracked file in this directory other than
the manifest itself. Verify it from the repository root with:

```bash
sha256sum --check data/MANIFEST.sha256
```

The source datasets retain their upstream license. The derived files are
research outputs from this repository; consult the scripts and documentation
that produced each file when tracing model or tool provenance.

The upstream `default_train.parquet` downloaded on 2026-08-04 has 4,139 rows.
The existing local database has 4,140 training rows from an earlier challenge
snapshot. Use `train_activity_db.parquet` when joining the published RDKit
descriptor table; do not join it to the current upstream file by row position.

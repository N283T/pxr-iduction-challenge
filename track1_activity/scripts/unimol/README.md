# Uni-Mol Axis

Status: active hold / promising revisit candidate.

This directory contains the Uni-Mol v2 Track 1 experiments. Earlier pool attempts
were null, including the PR #114 pretrain-embed framework and the PR #160
closeout. However, this axis should not be treated as dead: the directory also
contains later log2fc, filtered-label, layer, and multi-seed follow-ups that may
still be useful if revisited more carefully.

## Scripts

| Script | Purpose |
|---|---|
| `01_prepare_log2fc_data.py` | Export single-concentration log2fc labels and all-compound SMILES for Uni-Mol pretraining/extraction. |
| `02_pretrain_molv2.sh` | Run Uni-Mol v2 log2fc pretraining. |
| `03_extract_repr.sh` | Extract Uni-Mol representations for all compounds. |
| `04_npz_to_parquet.py` | Convert extracted representations to parquet for `run_train.py`. |
| `05_prepare_multitask_data.py` | Prepare multitask Uni-Mol training data. |
| `06_pretrain_multitask.sh` | Run multitask Uni-Mol pretraining. |
| `07_pretrain_pec50_ft.sh` | Fine-tune/pretrain on pEC50-oriented data. |
| `08_prepare_pec50_data.py` | Prepare pEC50 data for Uni-Mol fine-tuning. |
| `09_npz_to_parquet_v2.py` | Convert the second representation format to parquet. |
| `10_extract_repr_v2.sh` | Extract the second Uni-Mol representation set. |
| `11_prepare_filtered_log2fc.py` | Prepare filtered log2fc labels. |
| `12_pretrain_log2fc_filtered.sh` | Run filtered-label log2fc pretraining. |
| `13_compare_embed_drift.py` | Compare embedding drift across Uni-Mol variants. |
| `14_extract_intermediate_layers.py` | Extract intermediate-layer embeddings. |
| `15_pretrain_log2fc_multiseed.sh` | Run multi-seed log2fc pretraining. |
| `16_build_seed5ens.py` | Average five Uni-Mol log2fc embedding seeds. |

## Revisit Ideas

- Treat Uni-Mol as a variance-reduction axis, not a single weak pool member.
- Recheck intermediate-layer embeddings and seed ensembles against the current
  ensemble pool, not against older PR #114/#160 pools.
- Compare filtered-label and unfiltered log2fc pretraining with the same
  calibration and Caruana swap logic used for the current ChemProp recipe.

## Cleanup Stance

Do not delete or bulk-archive this directory. Keep it separate from closed axes
until a focused revisit confirms that the later seed/layer variants are also
unhelpful.

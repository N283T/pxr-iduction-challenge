# Track 2 (Structure Prediction) — Submission Spec & Reference

Compiled 2026-04-25 from the official challenge Space, the announcement post,
and the official tutorial repository. Source URLs at the bottom.

## Dataset

- **184 ligand SMILES** in `data/structure_test.parquet`
  (`structure`, `smiles`), refreshed by
  `pixi run python download_data.py --configs structure`.
- A mix of fragments (soaked into apo crystals, P2₁2₁2₁, 10 mM, NSLS-II AMX/FMX)
  and active compounds drawn from the activity track.
- 68 PDB-derived re-refined structures are provided as Track 2 training data.
- The HF Space `STRUCTURE_DATASET_SIZE` constant is hard-coded to **184**, and
  the submission validator rejects any zip with a different file count.
- Late additions are possible — announced via Discord `#pxr-challenge`.

## Submission package

| Requirement | Value |
| --- | --- |
| Container | `.zip` archive |
| Number of files | **exactly 184** PDB files |
| Per-ligand file name | `{structure_id}.pdb` (e.g. `x01378-1.pdb`) — `Path(name).stem` is the id |
| Per-file format | PDB (no CIF/mmCIF) |
| Per-file content | Full predicted **protein–ligand complex**, monomeric protein |
| Ligand residue name | **`LIG`** (single residue, exactly one such residue per file) |
| Chain count | ≤ 2 (typical layout: protein chain `A`, ligand chain `B`) |
| Ligand connectivity | Must match the expected SMILES — verified by RDKit `AssignBondOrdersFromTemplate` |

Boltz-2 outputs the ligand as `LIG1` by default — we must rewrite the residue
name to `LIG`. The official tutorial provides a one-liner using `biotite`:

```python
import biotite.structure.io as strucio
stack = strucio.load_structure(cif_path)
stack.res_name[stack.res_name == "LIG1"] = "LIG"
strucio.save_structure(out_pdb, stack)
```

A failed/missing/malformed structure is **not** silently skipped; the scoring
pipeline assigns the worst possible score (LDDT-PLI = 0.0, BiSyRMSD = 20.0 Å)
to penalise omission. Always submit something for every compound.

## Scoring (OpenStructure / OST)

Each predicted complex is scored against its blinded crystallographic reference
with two OST scorers:

- `SCRMSDScorer` — superposes Cα atoms within 8 Å of the reference ligand,
  reports symmetry-corrected ligand RMSD (BiSyRMSD) and LDDT-LP as a by-product.
- `LDDTPLIScorer` — computes LDDT-PLI without superposition; evaluates how
  many reference protein–ligand contacts are preserved.

When multiple ligand assignments are possible, the best (highest LDDT-PLI then
lowest BiSyRMSD) is kept. Metrics are bootstrapped over **1000** resamples.

| Metric | Direction | Notes |
| --- | :---: | --- |
| **LDDT-PLI** *(primary)* | ↑ | Superposition-free, symmetry-aware |
| BiSyRMSD | ↓ | Binding-site superposition |
| LDDT-LP | ↑ | Lining-residue LDDT |
| Coverage | ↑ | Fraction with a valid ligand assignment |

Failure penalty constants (from `evaluation/config.py`):
`BISYRMSD_NAN_PENALTY = 20.0`, LDDT-PLI / LDDT-LP fallback = 0.0.

### Live vs. blinded split
- 92 / 184 compounds are scored on the **live leaderboard**.
- The other 92 are **fully blinded** until the deadline.
- We do not know which is which, so all 184 must be optimised.

## Submission cadence

- 4 h cooldown between submissions (same as Track 1, `HOURS_BETWEEN_SUBMISSIONS = 4`).
- API endpoint: `gradio_client.Client("openadmet/pxr-challenge")`,
  `api_name="/submit_predictions"`, `track_select="Structure Prediction"`.
- Result back-fill (LB rank/metrics) takes ≲ 2 h.

## Timeline

| Date | Event |
| --- | --- |
| 2026-03-17 | Challenge announced |
| 2026-04-01 | Train / test SMILES released, submissions open |
| 2026-05-25 | Phase 1 (Activity only) closes |
| 2026-05-26 | Analog Set 1 unblinded |
| **2026-07-01** | **Track 2 + Activity Phase 2 deadline** |

Track 2 has no Phase 1 / 2 split — single deadline.

## Official tutorial / baseline

- Repo: `https://github.com/OpenADMET/PXR-Challenge-Tutorial` (cloned via ghq to
  `~/ghq/github.com/OpenADMET/PXR-Challenge-Tutorial`).
- `outputs/example_structure_submission/` ships **all 184 Boltz-2 PDB files**
  preformatted with `LIG` residue name → can be zipped and submitted as-is for
  a baseline LB entry.
- `validation/structure_validation.py` — drop-in pre-flight check (zip layout,
  file count, residue name, chain count, SMILES connectivity match).
- `evaluation/evaluate_predictions.py` — local OST scoring; requires
  `conda-forge::openstructure`. Useful for sanity checks on the 68 re-refined
  PDB training compounds (we cannot self-score the blinded test set).

### Reference Boltz-2 input (tutorial default)
```yaml
version: 1
sequences:
  - protein: {id: A, sequence: <293-aa PXR LBD>}
  - ligand:  {id: B, smiles: <SMILES>}
properties:
  - affinity: {binder: B}    # optional, not scored in Track 2
```
- **No MSA.**
- **No pocket constraint.**
- Single `recycling_steps` / `diffusion_samples_per_input` defaults.

## Differences vs. our Track 1 Boltz-2 setup

| Knob | Track 1 (`track1_activity/boltz2/`) | Tutorial reference | Track 2 default direction |
| --- | --- | --- | --- |
| Protein sequence | 293-aa LBD (`structures/boltz2/constants.py`) | Same 293-aa LBD | unchanged |
| MSA | AlphaFold-DB MSA at `structures/boltz2/msa/pxr.a3m` | None | TBD — adding MSA usually helps quality but the official baseline does without it |
| Pocket constraint | Core-pocket residues constrained | None | TBD — no constraint = unbiased docking; constraint risks misleading the model on fragments that may bind differently |
| Affinity head | Enabled | Optional | Not scored, leave on for free signal |
| `diffusion_samples_per_input` | 1 (default) | 1 (default) | Bump to take best-of-N; pose A/B will tell |
| `recycling_steps` | 3 (default) | 3 (default) | Increase later if VRAM allows |

These are the levers we want to A/B before the production 184-compound run.

## Track 1 ⇄ Track 2 compound overlap

72 / 184 Track 2 SMILES already exist in the Track 1 DB (39 % overlap):
- 57 in `test_activity` (Track 1 blinded test)
- 12 in `train_activity`
- 10 in `counter_assay`
- 13 in `single_concentration`

69 of those 72 already have Track 1 Boltz-2 poses on disk. **We are intentionally
not reusing them**: Track 2 sampling settings differ and we want a clean,
independent run for fair LB attribution.

## File layout we will adopt for Track 2

```
track2_structure/
  src/track2/         # PDB rewrite, validation, OST scoring helpers
  scripts/            # input builder, full run, postprocess, package_zip
  configs/            # YAML sweeps (MSA on/off, pocket on/off, samples, …)
  inputs/             # 184 Boltz-2 input YAMLs (per config)
  outputs/            # CIFs / PDBs / per-config submission zips
  notebooks/          # marimo EDA + score analysis
```
Boltz-2 itself stays installed as the existing `uv tool` (`boltz[cuda]`).

## Source URLs (verified 2026-04-25)

- Challenge Space — https://huggingface.co/spaces/openadmet/pxr-challenge
- Announcement post — https://openadmet.ghost.io/announcing-the-next-openadmet-blind-challenge-predicting-pxr-induction/
- Tutorial repo — https://github.com/OpenADMET/PXR-Challenge-Tutorial
- Dataset — https://huggingface.co/datasets/openadmet/pxr-challenge-train-test
- Discord — `#pxr-challenge`, https://discord.gg/MY5cEFHH3D
- OST documentation — https://openstructure.org/

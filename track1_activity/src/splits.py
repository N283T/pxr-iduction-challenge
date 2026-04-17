"""Cross-validation split strategies for Track 1."""

from collections import defaultdict

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold


def get_murcko_scaffold(smiles: str) -> str:
    """Get Murcko scaffold for a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        return scaffold
    except Exception:
        return ""


def _groups_to_splits(
    groups: list[list[int]],
    n_splits: int,
    rng: np.random.RandomState,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Distribute groups across folds (greedy) and return train/val splits."""
    groups_sorted = sorted(groups, key=len, reverse=True)

    fold_indices = [[] for _ in range(n_splits)]
    fold_sizes = [0] * n_splits

    for group in groups_sorted:
        min_fold = int(np.argmin(fold_sizes))
        fold_indices[min_fold].extend(group)
        fold_sizes[min_fold] += len(group)

    for indices in fold_indices:
        rng.shuffle(indices)

    splits = []
    for val_fold in range(n_splits):
        val_idx = np.array(fold_indices[val_fold])
        train_idx = np.concatenate(
            [np.array(fold_indices[i]) for i in range(n_splits) if i != val_fold]
        )
        splits.append((train_idx, val_idx))

    return splits


def scaffold_split_indices(
    smiles_list: list[str],
    n_splits: int = 5,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate scaffold-based k-fold split indices.

    Groups compounds by Murcko scaffold, then distributes scaffold groups
    across folds to ensure no scaffold appears in both train and validation.

    Returns list of (train_indices, val_indices) tuples.
    """
    rng = np.random.RandomState(seed)

    scaffold_to_indices = defaultdict(list)
    for i, smi in enumerate(smiles_list):
        scaffold = get_murcko_scaffold(smi)
        scaffold_to_indices[scaffold].append(i)

    return _groups_to_splits(list(scaffold_to_indices.values()), n_splits, rng)


def umap_split_indices(
    smiles_list: list[str],
    n_splits: int = 5,
    n_clusters: int = 50,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """UMAP + KMeans clustering split (strictest separation).

    Projects Morgan FPs into UMAP space, clusters with KMeans,
    then distributes clusters across folds.
    """
    import umap
    from sklearn.cluster import KMeans

    rng = np.random.RandomState(seed)

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    invalid = [i for i, m in enumerate(mols) if m is None]
    if invalid:
        raise ValueError(f"Invalid SMILES at indices: {invalid[:10]}")

    fps = np.array(
        [gen.GetFingerprintAsNumPy(m) for m in mols],
        dtype=np.float32,
    )

    reducer = umap.UMAP(
        n_components=10, metric="jaccard", random_state=seed, n_neighbors=30
    )
    embedding = reducer.fit_transform(fps)

    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    cluster_labels = km.fit_predict(embedding)

    cluster_to_indices = defaultdict(list)
    for i, cl in enumerate(cluster_labels):
        cluster_to_indices[cl].append(i)

    return _groups_to_splits(list(cluster_to_indices.values()), n_splits, rng)


def analog_aware_split_indices(
    smiles_list: list[str],
    pec50: np.ndarray,
    selectivity: np.ndarray,
    n_splits: int = 5,
    potent_pec50_threshold: float = 6.0,
    potent_sel_threshold: float = 1.5,
    analog_tanimoto_threshold: float = 0.25,
    seed: int = 42,
    verbose: bool = False,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Analog-aware k-fold split that mimics the LB test-generation process.

    Rationale
    ---------
    The Octant pipeline constructs the test set from (a) 46 potent
    drug-like inducers in train (pEC50 >= 6 AND selectivity >= 1.5) and
    (b) Enamine/FDA analogs of those seeds. EDA in `docs/track1_cv_prep.md`
    confirms 48.9% of test compounds have their NN inside the potent-46
    subset (a ~45x enrichment over the 1.1% base rate) and that the LB
    RAE gap (+0.096) is mostly a narrower test pEC50 dispersion, not
    prediction degradation.

    To produce OOF val subsets whose y dispersion matches LB's, this
    split:
      1. Always keeps the 46 potent seeds in the training side (matching
         "LB train sees all potents").
      2. Classifies every non-potent train compound by its Morgan-FP
         Tanimoto NN to any potent seed. "Analogs" = NN >= threshold.
      3. Distributes the analog pool across `n_splits` folds (random,
         seeded). Each fold's val is one analog bucket.
      4. Non-analogs (NN < threshold) always stay in train.

    The val set therefore consists of "analogs of the potents that are
    in train" — the same relation as LB test <-> LB train.

    Parameters
    ----------
    smiles_list:
        Length-N list of SMILES (use standardised SMILES for consistency).
    pec50:
        Length-N array of pEC50 values (train_activity.pec50).
    selectivity:
        Length-N array of (train_pec50 - counter_pec50). NaN where the
        compound lacks a counter_assay row.
    n_splits:
        Number of folds (default 5).
    potent_pec50_threshold, potent_sel_threshold:
        Definition of the "potent seed" subset. Defaults match the
        Octant-confirmed "46 potents".
    analog_tanimoto_threshold:
        Minimum NN Tanimoto to any potent seed for a non-potent compound
        to be classified as "analog" (eligible for val). Train is much
        less analog-rich than the synthetically expanded test set, so
        thresholds calibrated on test-side NN distributions don't
        transfer: threshold=0.4 yields only 27 train analogs (val~=6
        per fold -- too noisy); threshold=0.3 yields 177 (val~=35 --
        still fold-noisy); threshold=0.25 yields 849 (val~=170 -- stable
        estimates); threshold=0.2 yields 2744 (val~=549 -- similar size
        to the actual LB test set).

        The default 0.25 is a bias-variance compromise. Val y-dispersion
        at 0.25 is 0.68, already significantly below train-wide 0.86 and
        bracketing LB's 0.80 on the low side. For sensitivity analysis
        try 0.20 (widest; matches LB val size) or 0.30 (tightest; noisy
        but purest "close analog" definition).
    seed:
        RNG seed for analog bucket assignment.
    verbose:
        If True, print subset sizes for each fold.

    Returns
    -------
    List of (train_indices, val_indices) tuples, one per fold. Each
    val_indices contains indices of analog compounds only; train always
    includes the 46 potent seeds + all non-analog compounds + the
    other (n_splits - 1) analog buckets.
    """
    n = len(smiles_list)
    if len(pec50) != n or len(selectivity) != n:
        raise ValueError(
            f"Length mismatch: smiles={n}, pec50={len(pec50)}, "
            f"selectivity={len(selectivity)}"
        )

    pec50_arr = np.asarray(pec50, dtype=np.float64)
    sel_arr = np.asarray(selectivity, dtype=np.float64)

    potent_mask = (pec50_arr >= potent_pec50_threshold) & (
        np.nan_to_num(sel_arr, nan=-np.inf) >= potent_sel_threshold
    )
    potent_idx = np.where(potent_mask)[0]
    non_potent_idx = np.where(~potent_mask)[0]

    if len(potent_idx) == 0:
        raise ValueError(
            f"No compounds meet potent criteria "
            f"(pEC50>={potent_pec50_threshold} AND sel>={potent_sel_threshold})"
        )

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    invalid = [i for i, m in enumerate(mols) if m is None]
    if invalid:
        raise ValueError(f"Invalid SMILES at indices: {invalid[:10]}")

    all_fps = [gen.GetFingerprint(m) for m in mols]
    potent_fps = [all_fps[i] for i in potent_idx]

    nn_to_potent = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if potent_mask[i]:
            nn_to_potent[i] = np.nan
            continue
        sims = DataStructs.BulkTanimotoSimilarity(all_fps[i], potent_fps)
        nn_to_potent[i] = max(sims) if sims else 0.0

    # NaN >= threshold is False in numpy, so potent indices (which have
    # nn_to_potent=NaN) are automatically excluded here.
    analog_mask = nn_to_potent >= analog_tanimoto_threshold
    analog_idx = np.where(analog_mask)[0]
    non_analog_mask = (~potent_mask) & (~analog_mask)
    non_analog_idx = np.where(non_analog_mask)[0]

    if len(analog_idx) < n_splits:
        raise ValueError(
            f"Only {len(analog_idx)} analog compounds found but "
            f"{n_splits} folds requested. Lower analog_tanimoto_threshold."
        )

    rng = np.random.RandomState(seed)
    shuffled_analogs = analog_idx.copy()
    rng.shuffle(shuffled_analogs)
    analog_folds = np.array_split(shuffled_analogs, n_splits)

    if verbose:
        print(
            f"analog_aware_split: n={n}, potent={len(potent_idx)}, "
            f"analog={len(analog_idx)} (threshold={analog_tanimoto_threshold}), "
            f"non_analog={len(non_analog_idx)}"
        )

    splits = []
    always_train = np.concatenate([potent_idx, non_analog_idx])
    for k in range(n_splits):
        val_idx = analog_folds[k].astype(np.int64)
        other_analog_idx = np.concatenate(
            [analog_folds[j] for j in range(n_splits) if j != k]
        ).astype(np.int64)
        train_idx = np.concatenate([always_train, other_analog_idx])
        if verbose:
            print(
                f"  fold {k}: train={len(train_idx)} "
                f"(potent={len(potent_idx)} + non_analog={len(non_analog_idx)} + "
                f"analog_other={len(other_analog_idx)}), val={len(val_idx)}"
            )
        splits.append((train_idx, val_idx))

    return splits

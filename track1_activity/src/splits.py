"""Cross-validation split strategies for Track 1."""

from collections import defaultdict
from dataclasses import dataclass

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


def _morgan_fp_matrix(smiles_list: list[str]) -> np.ndarray:
    """Build a (n, 2048) Morgan r=2 fingerprint matrix. Used by Morgan-space
    UMAP splits and shared with other callers that want consistent FPs."""
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    invalid = [i for i, m in enumerate(mols) if m is None]
    if invalid:
        raise ValueError(f"Invalid SMILES at indices: {invalid[:10]}")
    return np.array(
        [gen.GetFingerprintAsNumPy(m) for m in mols],
        dtype=np.float32,
    )


@dataclass
class UmapSplitResult:
    """Intermediate artifacts and final folds from a UMAP cluster split."""

    splits: list[tuple[np.ndarray, np.ndarray]]
    embedding: np.ndarray
    cluster_labels: np.ndarray
    fold_labels: np.ndarray


def build_umap_split(
    smiles_list: list[str],
    n_splits: int = 5,
    n_clusters: int = 50,
    seed: int = 42,
    features: np.ndarray | None = None,
    metric: str = "jaccard",
) -> UmapSplitResult:
    """Build a UMAP + KMeans split and retain its intermediate artifacts.

    The defaults are the canonical Track 1 CV recipe. Callers that only need
    train/validation indices should continue to use :func:`umap_split_indices`.
    """
    import umap
    from sklearn.cluster import KMeans

    rng = np.random.RandomState(seed)

    if features is None:
        features = _morgan_fp_matrix(smiles_list)
    elif len(features) != len(smiles_list):
        raise ValueError(
            f"features length {len(features)} != smiles_list length {len(smiles_list)}"
        )

    reducer = umap.UMAP(
        n_components=10, metric=metric, random_state=seed, n_neighbors=30
    )
    embedding = reducer.fit_transform(features)

    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    cluster_labels = km.fit_predict(embedding)

    cluster_to_indices = defaultdict(list)
    for i, cl in enumerate(cluster_labels):
        cluster_to_indices[cl].append(i)

    splits = _groups_to_splits(list(cluster_to_indices.values()), n_splits, rng)
    fold_labels = np.full(len(smiles_list), -1, dtype=np.int64)
    for fold, (_, val_idx) in enumerate(splits):
        fold_labels[val_idx] = fold
    if np.any(fold_labels < 0):
        raise RuntimeError("UMAP split did not assign every row to a validation fold")

    return UmapSplitResult(
        splits=splits,
        embedding=embedding,
        cluster_labels=cluster_labels,
        fold_labels=fold_labels,
    )


def umap_split_indices(
    smiles_list: list[str],
    n_splits: int = 5,
    n_clusters: int = 50,
    seed: int = 42,
    features: np.ndarray | None = None,
    metric: str = "jaccard",
) -> list[tuple[np.ndarray, np.ndarray]]:
    """UMAP + KMeans clustering split (strictest separation).

    Default behaviour: projects Morgan FPs (r=2, 2048 bits) into UMAP
    space with Jaccard metric, clusters with KMeans, then distributes
    clusters across folds. This is the canonical split used throughout
    the project up to PR #69.

    Parameters
    ----------
    smiles_list:
        Length-N SMILES (used only to build the default Morgan matrix
        when ``features`` is None; ignored otherwise).
    n_splits, n_clusters, seed:
        Standard K-fold / KMeans knobs.
    features:
        Optional (N, D) feature matrix to replace the default Morgan
        fingerprints. Intended for non-fingerprint embedding spaces
        (ChemBERTa, MoLFormer, Mordred, etc.). When provided the caller
        should also pick an appropriate ``metric`` (see below).
    metric:
        UMAP distance metric. Default ``"jaccard"`` matches the
        binary Morgan FP case. For dense embeddings pass
        ``"cosine"`` (or ``"euclidean"``) -- Jaccard is undefined for
        non-binary features and will give meaningless neighbours.
    """
    return build_umap_split(
        smiles_list=smiles_list,
        n_splits=n_splits,
        n_clusters=n_clusters,
        seed=seed,
        features=features,
        metric=metric,
    ).splits


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
    (b) Enamine/FDA analogs of those seeds. EDA in `https://github.com/N283T/pxr-iduction-challenge/issues/181`
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


def mixed_analog_diversity_split_indices(
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
    """Mixed analog + diversity k-fold split (full coverage variant of
    analog-aware CV; PR #70 "Still-open CV ideas" follow-up).

    Motivation
    ----------
    PR #69's `analog_aware_split_indices` captured the LB
    analog-expansion narrative but failed on LB A/B (PR #70) with
    three compounding issues:
      * partial coverage (~20%) - incompatible with ensemble OOF;
      * val size 170 - HPO overfit to a small val set;
      * diversity tail (~42% of test has NN < 0.3 to potent-46) never
        appears in val, so the split is blind to half of LB.

    This split fixes all three:
      1. Stratifies by category (potent / analog / diversity),
         shuffles each stratum independently, distributes each across
         `n_splits` buckets.
      2. fold k val = potent_bucket_k ∪ analog_bucket_k ∪
         diversity_bucket_k - full 100% coverage.
      3. Val composition per fold is approximately proportional to
         LB's mix (seed ratio ~1.1%, analog ratio ~48.9%, diversity
         ratio ~50%).

    This is deliberately NOT "analog_aware with potent in val for
    coverage" - it is a different design that accepts putting potent
    seeds in val (~9 per fold) as the cost of full coverage. The
    philosophical "potent always in train" purity from PR #69 is
    dropped; it was never strongly calibrated to LB (PR #70 showed
    UMAP-tuned models beat analog-tuned on LB despite analog-tuned
    having lower OOF).

    Parameters match `analog_aware_split_indices`.
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

    analog_mask = nn_to_potent >= analog_tanimoto_threshold
    analog_idx = np.where(analog_mask)[0]
    diversity_mask = (~potent_mask) & (~analog_mask)
    diversity_idx = np.where(diversity_mask)[0]

    rng = np.random.RandomState(seed)

    def _shuffle_split(idx: np.ndarray, k: int) -> list[np.ndarray]:
        shuffled = idx.copy()
        rng.shuffle(shuffled)
        return [b.astype(np.int64) for b in np.array_split(shuffled, k)]

    potent_buckets = _shuffle_split(potent_idx, n_splits)
    analog_buckets = _shuffle_split(analog_idx, n_splits)
    diversity_buckets = _shuffle_split(diversity_idx, n_splits)

    if verbose:
        print(
            f"mixed_analog_diversity_split: n={n}, potent={len(potent_idx)}, "
            f"analog={len(analog_idx)} (threshold={analog_tanimoto_threshold}), "
            f"diversity={len(diversity_idx)}"
        )

    splits = []
    for k in range(n_splits):
        val_idx = np.concatenate(
            [potent_buckets[k], analog_buckets[k], diversity_buckets[k]]
        )
        train_idx = np.concatenate(
            [
                np.concatenate([potent_buckets[j] for j in range(n_splits) if j != k]),
                np.concatenate([analog_buckets[j] for j in range(n_splits) if j != k]),
                np.concatenate(
                    [diversity_buckets[j] for j in range(n_splits) if j != k]
                ),
            ]
        )
        if verbose:
            print(
                f"  fold {k}: train={len(train_idx)}, val={len(val_idx)} "
                f"(potent={len(potent_buckets[k])} + "
                f"analog={len(analog_buckets[k])} + "
                f"diversity={len(diversity_buckets[k])})"
            )
        splits.append((train_idx, val_idx))

    return splits


def test_nn_split_indices(
    smiles_list: list[str],
    test_smiles: list[str],
    n_splits: int = 5,
    test_nn_threshold: float = 0.25,
    seed: int = 42,
    verbose: bool = False,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split stratified by each train compound's nearest-neighbour
    Tanimoto to the (known) test SMILES. Analog of
    `mixed_analog_diversity_split_indices` but seeded with the actual
    test SMILES instead of the potent-46 proxy.

    Rationale
    ---------
    PR #69's analog split used potent-46 (the 46 train hits that
    seeded test generation) as a proxy for "test-likeness". This
    misses the ~51% of test that are not close analogs of potent-46
    (see EDA 1: 48.9% of test has NN inside potent-46).

    Since the 513 test SMILES are distributed as part of the
    challenge (labels hidden, structures public), we can compute
    each train compound's Morgan NN directly to test. Then:

      1. train compounds with NN >= threshold are classified
         "test-like" (analog of E1's "analog" stratum).
      2. train compounds with NN < threshold are "diversity".
      3. Each stratum is shuffled and split into n_splits buckets.
      4. fold k val = test_like_bucket_k ∪ diversity_bucket_k
         (full coverage).

    Unlike E1 there is no "potent always in train" rule — we let
    potent-46 fall wherever their actual test-NN puts them (usually
    in the test-like stratum by construction, since test seeds them).

    Parameters
    ----------
    smiles_list:
        Length-N SMILES of the train set.
    test_smiles:
        The 513 test SMILES, used only for NN distance computation.
        No labels are read.
    n_splits:
        Number of folds (default 5).
    test_nn_threshold:
        Tanimoto threshold for "test-like". Default 0.25 matches
        mixed_analog_diversity_split_indices' analog threshold for
        direct comparison.
    seed:
        RNG seed for bucket assignment.
    verbose:
        Print stratum sizes and per-fold breakdown if True.
    """
    n = len(smiles_list)
    n_test = len(test_smiles)
    if n_test == 0:
        raise ValueError("test_smiles is empty")

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    train_mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    test_mols = [Chem.MolFromSmiles(s) for s in test_smiles]

    invalid_train = [i for i, m in enumerate(train_mols) if m is None]
    invalid_test = [i for i, m in enumerate(test_mols) if m is None]
    if invalid_train:
        raise ValueError(f"Invalid train SMILES at indices: {invalid_train[:10]}")
    if invalid_test:
        raise ValueError(f"Invalid test SMILES at indices: {invalid_test[:10]}")

    train_fps = [gen.GetFingerprint(m) for m in train_mols]
    test_fps = [gen.GetFingerprint(m) for m in test_mols]

    nn_to_test = np.zeros(n, dtype=np.float64)
    for i in range(n):
        sims = DataStructs.BulkTanimotoSimilarity(train_fps[i], test_fps)
        nn_to_test[i] = max(sims) if sims else 0.0

    test_like_mask = nn_to_test >= test_nn_threshold
    test_like_idx = np.where(test_like_mask)[0]
    diversity_idx = np.where(~test_like_mask)[0]

    if len(test_like_idx) < n_splits:
        raise ValueError(
            f"Only {len(test_like_idx)} test-like compounds at "
            f"threshold {test_nn_threshold}; need >= n_splits ({n_splits}). "
            f"Lower the threshold."
        )

    rng = np.random.RandomState(seed)

    def _shuffle_split(idx: np.ndarray, k: int) -> list[np.ndarray]:
        shuffled = idx.copy()
        rng.shuffle(shuffled)
        return [b.astype(np.int64) for b in np.array_split(shuffled, k)]

    test_like_buckets = _shuffle_split(test_like_idx, n_splits)
    diversity_buckets = _shuffle_split(diversity_idx, n_splits)

    if verbose:
        print(
            f"test_nn_split: n_train={n}, n_test={n_test}, "
            f"test_like={len(test_like_idx)} (nn>={test_nn_threshold}), "
            f"diversity={len(diversity_idx)}"
        )

    splits = []
    for k in range(n_splits):
        val_idx = np.concatenate([test_like_buckets[k], diversity_buckets[k]])
        train_idx = np.concatenate(
            [
                np.concatenate(
                    [test_like_buckets[j] for j in range(n_splits) if j != k]
                ),
                np.concatenate(
                    [diversity_buckets[j] for j in range(n_splits) if j != k]
                ),
            ]
        )
        if verbose:
            print(
                f"  fold {k}: train={len(train_idx)}, val={len(val_idx)} "
                f"(test_like={len(test_like_buckets[k])} + "
                f"diversity={len(diversity_buckets[k])})"
            )
        splits.append((train_idx, val_idx))

    return splits


def adversarial_split_indices(
    smiles_list: list[str],
    test_smiles: list[str],
    n_splits: int = 5,
    n_top: int = 849,
    classifier_folds: int = 5,
    seed: int = 42,
    verbose: bool = False,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """Adversarial-validation based split.

    Trains a binary classifier on (train=0) vs (test=1) using Morgan
    fingerprints, collects out-of-fold P(test) for every train
    compound, and uses the top-N ranking to define the "test-like"
    stratum. Remainder becomes "diversity". Full coverage 5-fold via
    same stratified bucketing as E1/E2.

    This differs from test_nn_split_indices in the test-likeness
    metric: E2 uses Morgan Tanimoto NN, E3 uses a LightGBM classifier
    that can pick up on higher-order feature combinations (e.g.
    "compounds with feature A AND B but NOT C"). If the train/test
    boundary is smooth in fingerprint space, E2 and E3 agree. If the
    classifier finds non-linear separations, they diverge.

    Returns
    -------
    (splits, p_test) where splits is the standard list-of-tuples and
    p_test is the (n_train,) array of out-of-fold P(test|features)
    for diagnostic inspection.
    """
    import lightgbm as lgb
    from sklearn.model_selection import KFold

    n = len(smiles_list)
    n_test = len(test_smiles)
    if n_test == 0:
        raise ValueError("test_smiles is empty")
    if n_top >= n:
        raise ValueError(f"n_top={n_top} >= n_train={n}")

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    train_mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    test_mols = [Chem.MolFromSmiles(s) for s in test_smiles]
    invalid_train = [i for i, m in enumerate(train_mols) if m is None]
    invalid_test = [i for i, m in enumerate(test_mols) if m is None]
    if invalid_train:
        raise ValueError(f"Invalid train SMILES: {invalid_train[:10]}")
    if invalid_test:
        raise ValueError(f"Invalid test SMILES: {invalid_test[:10]}")

    X_train = np.array(
        [gen.GetFingerprintAsNumPy(m) for m in train_mols], dtype=np.float32
    )
    X_test = np.array(
        [gen.GetFingerprintAsNumPy(m) for m in test_mols], dtype=np.float32
    )

    if verbose:
        print(
            f"adversarial_split: n_train={n}, n_test={n_test}, "
            f"{classifier_folds}-fold classifier..."
        )

    # K-fold classifier: for each fold, train on (4/5 train + all test),
    # predict P(test) for held-out 1/5 of train. This gives unbiased
    # out-of-fold P(test) for every train compound.
    p_test = np.zeros(n, dtype=np.float64)
    kf = KFold(n_splits=classifier_folds, shuffle=True, random_state=seed)
    for fold_k, (tr_idx, va_idx) in enumerate(kf.split(np.arange(n))):
        X_fit = np.vstack([X_train[tr_idx], X_test])
        y_fit = np.concatenate([np.zeros(len(tr_idx)), np.ones(n_test)]).astype(
            np.float32
        )
        clf = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            random_state=seed,
            verbose=-1,
        )
        clf.fit(X_fit, y_fit)
        p_test[va_idx] = clf.predict_proba(X_train[va_idx])[:, 1]
        if verbose:
            print(
                f"  clf fold {fold_k}: train_size={len(X_fit)}, "
                f"val P(test) mean={p_test[va_idx].mean():.3f} "
                f"std={p_test[va_idx].std():.3f}"
            )

    # Rank train by P(test), top n_top = test-like, rest = diversity
    sorted_idx = np.argsort(-p_test)  # descending
    test_like_idx = sorted_idx[:n_top].astype(np.int64)
    diversity_idx = sorted_idx[n_top:].astype(np.int64)

    rng = np.random.RandomState(seed)

    def _shuffle_split(idx: np.ndarray, k: int) -> list[np.ndarray]:
        shuffled = idx.copy()
        rng.shuffle(shuffled)
        return [b.astype(np.int64) for b in np.array_split(shuffled, k)]

    test_like_buckets = _shuffle_split(test_like_idx, n_splits)
    diversity_buckets = _shuffle_split(diversity_idx, n_splits)

    if verbose:
        thresh_p = p_test[sorted_idx[n_top - 1]]
        print(
            f"  P(test) distribution: min={p_test.min():.3f} "
            f"max={p_test.max():.3f} median={np.median(p_test):.3f}"
        )
        print(f"  test_like boundary: top {n_top} => P(test) >= {thresh_p:.3f}")

    splits = []
    for k in range(n_splits):
        val_idx = np.concatenate([test_like_buckets[k], diversity_buckets[k]])
        train_idx = np.concatenate(
            [
                np.concatenate(
                    [test_like_buckets[j] for j in range(n_splits) if j != k]
                ),
                np.concatenate(
                    [diversity_buckets[j] for j in range(n_splits) if j != k]
                ),
            ]
        )
        if verbose:
            print(
                f"  fold {k}: train={len(train_idx)}, val={len(val_idx)} "
                f"(test_like={len(test_like_buckets[k])} + "
                f"diversity={len(diversity_buckets[k])})"
            )
        splits.append((train_idx, val_idx))

    return splits, p_test

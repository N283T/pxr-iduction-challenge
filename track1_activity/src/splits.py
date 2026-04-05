"""Cross-validation split strategies for Track 1."""

from collections import defaultdict

import numpy as np
from rdkit import Chem
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

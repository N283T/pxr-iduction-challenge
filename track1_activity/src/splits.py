"""Cross-validation split strategies for Track 1."""

from collections import defaultdict

import numpy as np
import psycopg2
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from data import DB_PARAMS


def get_murcko_scaffold(smiles: str) -> str:
    """Get Murcko scaffold for a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol, includeChirality=False
        )
        return scaffold
    except Exception:
        return ""


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

    # Group indices by scaffold
    scaffold_to_indices = defaultdict(list)
    for i, smi in enumerate(smiles_list):
        scaffold = get_murcko_scaffold(smi)
        scaffold_to_indices[scaffold].append(i)

    # Sort scaffolds by size (largest first) for balanced distribution
    scaffold_groups = list(scaffold_to_indices.values())
    scaffold_groups.sort(key=len, reverse=True)

    # Assign scaffold groups to folds (greedy: assign to smallest fold)
    fold_indices = [[] for _ in range(n_splits)]
    fold_sizes = [0] * n_splits

    for group in scaffold_groups:
        # Find the fold with the fewest samples
        min_fold = int(np.argmin(fold_sizes))
        fold_indices[min_fold].extend(group)
        fold_sizes[min_fold] += len(group)

    # Shuffle within each fold
    for indices in fold_indices:
        rng.shuffle(indices)

    # Generate train/val splits
    splits = []
    for val_fold in range(n_splits):
        val_idx = np.array(fold_indices[val_fold])
        train_idx = np.concatenate(
            [np.array(fold_indices[i]) for i in range(n_splits) if i != val_fold]
        )
        splits.append((train_idx, val_idx))

    return splits


def load_train_scaffolds() -> list[str]:
    """Load Murcko scaffolds for train compounds from DB."""
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    cur.execute("""
        SELECT d.murcko_scaffold
        FROM train_activity t
        JOIN compound_descriptors d ON d.compound_id = t.compound_id
        ORDER BY t.id
    """)
    scaffolds = [r[0] or "" for r in cur.fetchall()]
    cur.close()
    conn.close()
    return scaffolds

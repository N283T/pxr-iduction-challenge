import sys
from pathlib import Path

import numpy as np
from rdkit import Chem

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from track1_activity.scripts.build_boltz2_contact_features import (
    classify_ligand_atom,
    summarize_residue_contacts,
)


def test_classify_ligand_atom_marks_aromatic_acceptor_and_hydrophobe():
    mol = Chem.MolFromSmiles("c1ccncc1O")
    classes = [classify_ligand_atom(atom) for atom in mol.GetAtoms()]

    assert "aromatic" in classes[0]
    assert "hydrophobe" in classes[0]
    assert "acceptor" in classes[3]
    assert "donor" in classes[-1]


def test_summarize_residue_contacts_counts_near_ligand_atom_classes():
    ligand_coords = np.array([[0.0, 0.0, 0.0], [8.0, 0.0, 0.0]], dtype=np.float32)
    ligand_classes = [{"hydrophobe"}, {"acceptor"}]
    residue_coords = np.array([[3.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float32)

    features = summarize_residue_contacts(
        residue_number=209,
        residue_coords=residue_coords,
        ligand_coords=ligand_coords,
        ligand_classes=ligand_classes,
        near_cutoff=4.5,
        shell_cutoff=6.0,
    )

    assert features["res209_min_dist"] == 2.0
    assert features["res209_n_lig_atoms_4p5"] == 2
    assert features["res209_n_lig_atoms_6p0"] == 2
    assert features["res209_hydrophobe_n_4p5"] == 1
    assert features["res209_acceptor_n_4p5"] == 1


if __name__ == "__main__":
    test_classify_ligand_atom_marks_aromatic_acceptor_and_hydrophobe()
    test_summarize_residue_contacts_counts_near_ligand_atom_classes()

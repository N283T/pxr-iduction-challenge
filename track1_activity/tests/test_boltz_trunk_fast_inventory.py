import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT.joinpath(
    "track1_activity", "scripts", "boltz_affhead", "37_trunk_fast_inventory.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("_trunk_fast_inventory", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_trunk_fast_inventory"] = module
    spec.loader.exec_module(module)
    return module


def test_missing_ids_returns_sorted_difference():
    mod = load_module()

    missing = mod.missing_ids([3, 1, 2, 5], [1, 3])

    assert missing == [2, 5]


def test_format_recycling_counts_sorts_by_recycling_steps():
    mod = load_module()

    text = mod.format_recycling_counts({3: 4652, 1: 8482})

    assert "rcycle=1: 8482" in text
    assert text.index("rcycle=1") < text.index("rcycle=3")


def test_summarize_npz_reports_shape_and_token_counts(tmp_path):
    mod = load_module()
    path = tmp_path.joinpath("embeddings_00001.npz")
    s = np.zeros((1, 457, 384), dtype=np.float32)
    z = np.zeros((1, 457, 457, 128), dtype=np.float32)
    np.savez(path, s=s, z=z)

    summary = mod.summarize_npz(path, protein_n_res=434)

    assert summary["readable"] is True
    assert summary["s_shape"] == [1, 457, 384]
    assert summary["z_shape"] == [1, 457, 457, 128]
    assert summary["ligand_tokens"] == 23
    assert summary["size_mb"] > 0


def test_is_boltz_experiment_name_filters_expected_family():
    mod = load_module()

    assert mod.is_boltz_experiment_name("tabpfn_pooled_boltz_allpairs_umap_default")
    assert mod.is_boltz_experiment_name("tabpfn_boltz_trunk_pretrain_embed_c_umap")
    assert not mod.is_boltz_experiment_name(
        "tabpfn_chemprop_pretrain_embed_umap_default"
    )


def test_classify_boltz_experiment_separates_trunk_from_descriptor_mix():
    mod = load_module()

    assert (
        mod.classify_boltz_experiment("tabpfn_pooled_boltz_allpairs_umap_default")
        == "trunk_only"
    )
    assert (
        mod.classify_boltz_experiment("tabpfn_cheme_2d_full_boltz_log2fc_pred")
        == "descriptor_mix"
    )
    assert (
        mod.classify_boltz_experiment("tabpfn_boltz2_tabular_tier0_umap_default")
        == "structure_tabular"
    )


if __name__ == "__main__":
    test_missing_ids_returns_sorted_difference()
    test_format_recycling_counts_sorts_by_recycling_steps()
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        test_summarize_npz_reports_shape_and_token_counts(Path(td))
    test_is_boltz_experiment_name_filters_expected_family()
    test_classify_boltz_experiment_separates_trunk_from_descriptor_mix()

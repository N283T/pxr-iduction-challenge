import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT.joinpath(
    "track1_activity", "scripts", "boltz_affhead", "38_repool_trunk_npz.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("_repool_trunk_npz", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_repool_trunk_npz"] = module
    spec.loader.exec_module(module)
    return module


def test_region_indices_are_zero_indexed_and_include_core_pocket():
    mod = load_module()

    regions = mod.region_indices()

    assert regions["nterm"][0] == 0
    assert regions["h11_h12"][-1] == 433
    assert 208 in regions["core_pocket"]


def test_pool_npz_region_zstats_returns_expected_columns(tmp_path):
    mod = load_module()
    path = tmp_path.joinpath("embeddings_00001.npz")
    # 4 protein tokens + 2 ligand tokens. Use small dims for helper behavior.
    s = np.arange(1 * 6 * 3, dtype=np.float32).reshape(1, 6, 3)
    z = np.arange(1 * 6 * 6 * 2, dtype=np.float32).reshape(1, 6, 6, 2)
    np.savez(path, s=s, z=z)
    regions = {
        "first_two": np.array([0, 1], dtype=np.int64),
        "last_two": np.array([2, 3], dtype=np.int64),
    }

    row = mod.pool_npz_region_zstats(path, protein_n_res=4, regions=regions)

    assert row["ligand_tokens"] == 2.0
    assert row["s_lig_mean_000"] == 13.5
    assert "z_first_two_mean_000" in row
    assert "z_last_two_q90_001" in row
    assert all(np.isfinite(v) for v in row.values())


if __name__ == "__main__":
    test_region_indices_are_zero_indexed_and_include_core_pocket()
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        test_pool_npz_region_zstats_returns_expected_columns(Path(td))

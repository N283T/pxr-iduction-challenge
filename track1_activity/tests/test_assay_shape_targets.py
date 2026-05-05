import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from track1_activity.scripts.run_chemprop_assay_shape_pretrain import (
    TASKS,
    build_dose_response_latent_targets,
    build_target_matrix,
    standardize_targets,
)


def test_build_target_matrix_preserves_missing_auxiliary_labels():
    df = pd.DataFrame(
        {
            "pxr_pec50": [5.0, np.nan, np.nan],
            "pxr_emax": [80.0, np.nan, np.nan],
            "counter_pec50": [4.7, np.nan, np.nan],
            "counter_emax": [40.0, np.nan, np.nan],
            "log2fc_8p25": [0.2, 0.8, np.nan],
            "log2fc_33": [0.5, np.nan, -0.3],
            "log2fc_99": [np.nan, np.nan, 1.0],
        }
    )

    targets, mask = build_target_matrix(df)

    assert targets.shape == (3, len(TASKS))
    assert mask.shape == targets.shape
    assert mask[:, TASKS.index("pxr_pec50")].tolist() == [True, False, False]
    assert mask[:, TASKS.index("pxr_minus_counter")].tolist() == [True, False, False]
    assert targets[0, TASKS.index("pxr_minus_counter")] == 0.3
    assert mask[:, TASKS.index("log2fc_8p25")].tolist() == [True, True, False]
    assert mask[:, TASKS.index("log2fc_33")].tolist() == [True, False, True]
    assert mask[:, TASKS.index("log2fc_99")].tolist() == [False, False, True]


def test_standardize_targets_keeps_nan_and_returns_observed_counts():
    targets = np.array(
        [
            [1.0, np.nan, 3.0],
            [2.0, np.nan, np.nan],
            [3.0, 4.0, 9.0],
        ],
        dtype=np.float32,
    )

    z, means, stds, counts = standardize_targets(targets)

    assert np.isnan(z[0, 1])
    assert np.isnan(z[1, 2])
    assert counts.tolist() == [3, 1, 2]
    assert means.tolist() == [2.0, 4.0, 6.0]
    assert stds[1] == 1.0
    assert np.isclose(z[0, 0], -1.2247449)


def test_build_dose_response_latent_targets_excludes_sparse_rows_and_pec50():
    df = pd.DataFrame(
        {
            "pxr_pec50": [6.0, 7.0, 8.0, 9.0],
            "pxr_emax": [80.0, 70.0, np.nan, np.nan],
            "pxr_emax_vs_pos_ctrl": [0.8, 0.7, np.nan, np.nan],
            "counter_present": [1.0, 1.0, np.nan, np.nan],
            "counter_pec50": [5.0, 5.5, np.nan, np.nan],
            "counter_emax": [20.0, 25.0, np.nan, np.nan],
            "counter_emax_vs_pos_ctrl": [0.2, 0.3, np.nan, np.nan],
            "log2fc_8p25": [0.1, 0.2, 0.3, np.nan],
            "log2fc_33": [0.4, 0.5, 0.6, np.nan],
            "log2fc_99": [0.7, 0.8, 0.9, np.nan],
        }
    )

    targets, tasks, meta = build_dose_response_latent_targets(
        df, n_components=2, min_observed=3, seed=42
    )

    assert tasks == ["drlatent_00", "drlatent_01"]
    assert targets.shape == (4, 2)
    assert np.isfinite(targets[:3]).all()
    assert np.isnan(targets[3]).all()
    assert "pxr_pec50" not in meta["assay_columns"]
    assert meta["fit_rows"] == 3


if __name__ == "__main__":
    test_build_target_matrix_preserves_missing_auxiliary_labels()
    test_standardize_targets_keeps_nan_and_returns_observed_counts()
    test_build_dose_response_latent_targets_excludes_sparse_rows_and_pec50()

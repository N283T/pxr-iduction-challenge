import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from track1_activity.scripts.build_dose_response_latent import (
    ASSAY_BASE_COLUMNS,
    build_assay_matrix,
    latent_training_mask,
)


def test_build_assay_matrix_excludes_pec50_and_adds_curve_shape_features():
    df = pd.DataFrame(
        {
            "pxr_pec50": [6.5, 7.0],
            "pxr_emax": [80.0, np.nan],
            "pxr_emax_vs_pos_ctrl": [0.7, np.nan],
            "counter_present": [1.0, 0.0],
            "counter_pec50": [5.1, np.nan],
            "counter_emax": [30.0, np.nan],
            "counter_emax_vs_pos_ctrl": [0.2, np.nan],
            "log2fc_8p25": [0.5, 0.2],
            "log2fc_33": [1.1, np.nan],
            "log2fc_99": [1.4, 0.4],
        }
    )

    matrix = build_assay_matrix(df)

    assert "pxr_pec50" not in matrix.columns
    assert ASSAY_BASE_COLUMNS[0] == "pxr_emax"
    assert np.isclose(matrix.loc[0, "log2fc_slope_8p25_33"], 0.6)
    assert np.isclose(matrix.loc[0, "log2fc_slope_33_99"], 0.3)
    assert matrix.loc[0, "log2fc_max"] == 1.4
    assert np.isclose(matrix.loc[0, "log2fc_auc"], 1.0)
    assert np.isnan(matrix.loc[1, "log2fc_slope_8p25_33"])
    assert matrix.loc[1, "log2fc_max"] == 0.4


def test_latent_training_mask_excludes_validation_ids_and_sparse_rows():
    compound_ids = np.array([10, 11, 12, 13])
    assay_counts = np.array([3, 1, 4, 2])

    mask = latent_training_mask(
        compound_ids=compound_ids,
        assay_counts=assay_counts,
        validation_ids={12},
        min_observed=2,
    )

    assert mask.tolist() == [True, False, False, True]


if __name__ == "__main__":
    test_build_assay_matrix_excludes_pec50_and_adds_curve_shape_features()
    test_latent_training_mask_excludes_validation_ids_and_sparse_rows()

"""Load Phase 2 HTChem Track 1 activity rows into PostgreSQL."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import psycopg2
from datasets import load_dataset
from psycopg2.extras import Json

DATA_DIR = Path(__file__).resolve().parent.parent.joinpath("data")
DB_PARAMS = {"dbname": "pxr_challenge", "host": "/tmp", "port": 5433}

SOURCES = {
    "crudes_htchem": {
        "source_type": "crude",
        "label": "Crude",
        "parquet": DATA_DIR.joinpath("crudes_htchem_train.parquet"),
        "note_col": "CAD Yield/Volatility Note",
    },
    "semi_pure_htchem": {
        "source_type": "semi_pure",
        "label": "Semi-Pure",
        "parquet": DATA_DIR.joinpath("semi_pure_htchem_train.parquet"),
        "note_col": None,
    },
}


def _none_if_na(value):
    return None if pd.isna(value) else value


def _float_or_none(value):
    if pd.isna(value):
        return None
    parsed = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(parsed) else float(parsed)


def _find_col(columns, *needles: str, exclude: tuple[str, ...] = ()) -> str:
    matches = [
        col
        for col in columns
        if all(needle.lower() in col.lower() for needle in needles)
        and not any(needle.lower() in col.lower() for needle in exclude)
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one column matching {needles}, found {matches}")
    return matches[0]


def _load_frame(config_name: str) -> pd.DataFrame:
    info = SOURCES[config_name]
    ds = load_dataset("openadmet/pxr-challenge-train-test", config_name)
    df = ds["train"].to_pandas()
    DATA_DIR.mkdir(exist_ok=True)
    df.to_parquet(info["parquet"], index=False)
    return df


def _row_values(row: pd.Series, columns, label: str, note_col: str | None) -> tuple:
    lower_label = label.lower()
    batch_col = _find_col(columns, label, "Batch ID")
    evap_col = _find_col(columns, label.split("-")[0], "EvapT")
    values = {
        "source_type": None,
        "ocnt_id": row["OCNT_ID"],
        "batch_id": row[batch_col],
        "ec50_um": row[_find_col(columns, label, "EC50s", exclude=("pEC50",))],
        "pec50": row[_find_col(columns, label, "pEC50s")],
        "emax_normalized": row[_find_col(columns, label, "Emax normalized")],
        "emax_raw": row[_find_col(columns, label, "Emax raw")],
        "corrected_ec50_um": row[
            _find_col(columns, "Corrected", label, "EC50", exclude=("pEC50",))
        ],
        "corrected_pec50": row[_find_col(columns, "Corrected", label, "pEC50 (log)")],
        "drc_pec50_se": row[_find_col(columns, label, "DRC pEC50 SE")],
        "corrected_pec50_se": row[
            _find_col(columns, "Corrected", label, "pEC50", "1 SE")
        ],
        "pec50_ci95": row[
            _find_col(columns, label, "pEC50", "95% CI", exclude=("Corrected",))
        ],
        "corrected_pec50_ci95": row[
            _find_col(columns, "Corrected", label, "pEC50", "95% CI")
        ],
        "volatility": row["Volatility"],
        "cad_yield_volatility_note": None if note_col is None else row.get(note_col),
        "evapt_c": row[evap_col],
        "theoretical_mass_on_column_ng": row[
            _find_col(columns, label, "Theoretical Mass-on-Column")
        ],
        "peak_area_pa_min": row[
            _find_col(columns, label, "Peak Area", exclude=("CV",))
        ],
        "actual_mass_on_column_ng": row[
            _find_col(columns, label, "Actual Mass-on-Column")
        ],
        "product_yield_percent": row[_find_col(columns, label, "Product Yield")],
        "correction_factor": row[_find_col(columns, label, "Correction Factor")],
        "cad_peak_area_cv_percent": row[_find_col(columns, label, "CAD Peak Area CV")],
        "cad_slope_cv_percent": row[_find_col(columns, label, "CAD Slope CV")],
        "cad_yield_se_log10": row[_find_col(columns, label, "CAD Yield SE")],
    }

    if lower_label == "semi-pure":
        values["evapt_c"] = row[_find_col(columns, "Semi-pure", "EvapT")]

    numeric_keys = [
        "ec50_um",
        "pec50",
        "emax_normalized",
        "emax_raw",
        "corrected_ec50_um",
        "corrected_pec50",
        "drc_pec50_se",
        "corrected_pec50_se",
        "pec50_ci95",
        "corrected_pec50_ci95",
        "evapt_c",
        "theoretical_mass_on_column_ng",
        "peak_area_pa_min",
        "actual_mass_on_column_ng",
        "product_yield_percent",
        "correction_factor",
        "cad_peak_area_cv_percent",
        "cad_slope_cv_percent",
        "cad_yield_se_log10",
    ]
    for key in numeric_keys:
        values[key] = _float_or_none(values[key])

    return (
        values["ocnt_id"],
        values["batch_id"],
        values["ec50_um"],
        values["pec50"],
        values["emax_normalized"],
        values["emax_raw"],
        values["corrected_ec50_um"],
        values["corrected_pec50"],
        values["drc_pec50_se"],
        values["corrected_pec50_se"],
        values["pec50_ci95"],
        values["corrected_pec50_ci95"],
        _none_if_na(values["volatility"]),
        _none_if_na(values["cad_yield_volatility_note"]),
        values["evapt_c"],
        values["theoretical_mass_on_column_ng"],
        values["peak_area_pa_min"],
        values["actual_mass_on_column_ng"],
        values["product_yield_percent"],
        values["correction_factor"],
        values["cad_peak_area_cv_percent"],
        values["cad_slope_cv_percent"],
        values["cad_yield_se_log10"],
    )


def _ensure_compounds(cur, df: pd.DataFrame) -> dict[str, int]:
    for _, row in df[["SMILES", "OCNT_ID"]].drop_duplicates("SMILES").iterrows():
        cur.execute(
            """
            INSERT INTO compounds (molecule_name, smiles)
            VALUES (%s, %s)
            ON CONFLICT (smiles) DO NOTHING
            """,
            (row["OCNT_ID"], row["SMILES"]),
        )
    cur.execute("SELECT smiles, id FROM compounds")
    return {smiles: compound_id for smiles, compound_id in cur.fetchall()}


def main() -> None:
    frames = {config: _load_frame(config) for config in SOURCES}
    for config, df in frames.items():
        print(f"Loaded {config}/train from HF: {len(df)} rows")
        print(f"Cached parquet: {SOURCES[config]['parquet']}")

    conn = psycopg2.connect(**DB_PARAMS)
    try:
        cur = conn.cursor()
        compound_map = _ensure_compounds(
            cur, pd.concat(frames.values(), ignore_index=True)
        )

        for config, df in frames.items():
            info = SOURCES[config]
            columns = list(df.columns)
            for _, row in df.iterrows():
                cur.execute(
                    """
                    INSERT INTO htchem_activity
                        (compound_id, source_type, ocnt_id, batch_id,
                         ec50_um, pec50, emax_normalized, emax_raw,
                         corrected_ec50_um, corrected_pec50, drc_pec50_se,
                         corrected_pec50_se, pec50_ci95, corrected_pec50_ci95,
                         volatility, cad_yield_volatility_note, evapt_c,
                         theoretical_mass_on_column_ng, peak_area_pa_min,
                         actual_mass_on_column_ng, product_yield_percent,
                         correction_factor, cad_peak_area_cv_percent,
                         cad_slope_cv_percent, cad_yield_se_log10, raw_record)
                    VALUES
                        (%s, %s, %s, %s,
                         %s, %s, %s, %s,
                         %s, %s, %s,
                         %s, %s, %s,
                         %s, %s, %s,
                         %s, %s,
                         %s, %s,
                         %s, %s,
                         %s, %s, %s)
                    ON CONFLICT (source_type, ocnt_id, batch_id) DO UPDATE SET
                        compound_id = EXCLUDED.compound_id,
                        ec50_um = EXCLUDED.ec50_um,
                        pec50 = EXCLUDED.pec50,
                        emax_normalized = EXCLUDED.emax_normalized,
                        emax_raw = EXCLUDED.emax_raw,
                        corrected_ec50_um = EXCLUDED.corrected_ec50_um,
                        corrected_pec50 = EXCLUDED.corrected_pec50,
                        drc_pec50_se = EXCLUDED.drc_pec50_se,
                        corrected_pec50_se = EXCLUDED.corrected_pec50_se,
                        pec50_ci95 = EXCLUDED.pec50_ci95,
                        corrected_pec50_ci95 = EXCLUDED.corrected_pec50_ci95,
                        volatility = EXCLUDED.volatility,
                        cad_yield_volatility_note = EXCLUDED.cad_yield_volatility_note,
                        evapt_c = EXCLUDED.evapt_c,
                        theoretical_mass_on_column_ng =
                            EXCLUDED.theoretical_mass_on_column_ng,
                        peak_area_pa_min = EXCLUDED.peak_area_pa_min,
                        actual_mass_on_column_ng =
                            EXCLUDED.actual_mass_on_column_ng,
                        product_yield_percent = EXCLUDED.product_yield_percent,
                        correction_factor = EXCLUDED.correction_factor,
                        cad_peak_area_cv_percent =
                            EXCLUDED.cad_peak_area_cv_percent,
                        cad_slope_cv_percent = EXCLUDED.cad_slope_cv_percent,
                        cad_yield_se_log10 = EXCLUDED.cad_yield_se_log10,
                        raw_record = EXCLUDED.raw_record,
                        loaded_at = now()
                    """,
                    (
                        compound_map[row["SMILES"]],
                        info["source_type"],
                        *_row_values(row, columns, info["label"], info["note_col"]),
                        Json(row.to_dict()),
                    ),
                )

        conn.commit()
        cur.execute(
            """
            SELECT source_type, count(*), count(pec50), count(corrected_pec50)
            FROM htchem_activity
            GROUP BY source_type
            ORDER BY source_type
            """
        )
        for source_type, total, pec50_count, corrected_count in cur.fetchall():
            print(
                f"{source_type}: rows={total}, "
                f"pec50={pec50_count}, corrected_pec50={corrected_count}"
            )
    finally:
        conn.close()


if __name__ == "__main__":
    main()

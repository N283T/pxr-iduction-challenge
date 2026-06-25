"""External rank-style features for Phase 2 classifier gate probes."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "track1_activity" / "src"))
sys.path.insert(
    0, str(REPO_ROOT / "track1_activity" / "analysis" / "chembl_pxr_probe")
)
sys.path.insert(
    0, str(REPO_ROOT / "track1_activity" / "analysis" / "phase2_htchem_pred_axis")
)

from chembl_pxr_activation_probe import tanimoto_matrix  # noqa: E402
from data import get_engine  # noqa: E402
from run_top500_tabpfn_htchem import pred_htchem_for_challenge  # noqa: E402
from splits import _morgan_fp_matrix  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "outputs" / "external_rank_features"
CHEMBL_URL = "postgresql+psycopg2:///chembl_36?host=/tmp&port=5433"
LOG2FC_PATH = (
    REPO_ROOT
    / "data"
    / "chemprop_pretrain_log2fc_predictions_optuna_trial10_seed5ens.parquet"
)

CHEMBL_FEATURE_COLS = [
    "chembl_rank_nn_sim",
    "chembl_rank_nn_pct",
    "chembl_rank_nn_high_frac",
    "chembl_rank_nn_pchembl",
    "chembl_rank_nn_n_assays",
    "chembl_rank_top5_pct",
    "chembl_rank_top5_high_frac",
    "chembl_rank_top5_pchembl",
    "chembl_rank_top5_sim_sum",
    "chembl_rank_covered_t025",
    "chembl_rank_covered_t030",
]
HTCHEM_RANK_COLS = [
    "htchem_rank_pct",
    "htchem_rank_z",
    "htchem_minus_lf_z_rank",
    "lf_rank_pct_for_htchem",
]
PAIRRANK_COLS = [
    "pairrank_chembl",
    "pairrank_htchem",
]
PAIRRANK_ALL_COLS = [
    "pairrank_all",
    "pairrank_chembl",
    "pairrank_htchem",
    "pairrank_single_conc",
]
PAIRRANK_DIR = (
    Path(__file__).resolve().parent
    / "outputs"
    / "pairwise_assay_rank"
    / "all_pxr_chembl_htchem_single_conc_mpa1500_top64"
)


def _filter_pxr_rank_rows(raw: pd.DataFrame) -> pd.DataFrame:
    desc = raw["description"].fillna("").str.lower()
    keep_terms = (
        desc.str.contains("activation")
        | desc.str.contains("agonist")
        | desc.str.contains("transactivation")
        | desc.str.contains("cyp3a4")
        | desc.str.contains("luciferase")
    )
    drop_terms = (
        desc.str.contains("antagonist")
        | desc.str.contains("inverse agonist")
        | desc.str.contains("inhibition")
        | desc.str.contains("binding")
        | desc.str.contains("displacement")
    )
    return raw[
        raw["assay_type"].eq("A")
        & raw["confidence_score"].ge(8)
        & raw["pchembl_value"].notna()
        & keep_terms
        & ~drop_terms
    ].copy()


def load_chembl_pxr_rank_molecules() -> pd.DataFrame:
    """Return ChEMBL PXR molecules with assay-local rank aggregates."""

    chembl = create_engine(CHEMBL_URL)
    raw = pd.read_sql(
        """
        SELECT
            s.standard_inchi_key AS inchikey,
            s.canonical_smiles AS smiles,
            m.chembl_id AS mol_chembl_id,
            a.chembl_id AS assay_chembl_id,
            a.assay_type,
            a.confidence_score,
            a.description,
            act.standard_type,
            act.pchembl_value
        FROM compound_structures s
        JOIN molecule_dictionary m ON m.molregno = s.molregno
        JOIN activities act ON act.molregno = m.molregno
        JOIN assays a ON a.assay_id = act.assay_id
        JOIN target_dictionary t ON t.tid = a.tid
        WHERE t.chembl_id = 'CHEMBL3401'
          AND s.canonical_smiles IS NOT NULL
          AND act.pchembl_value IS NOT NULL
        """,
        chembl,
    )
    rows = _filter_pxr_rank_rows(raw)
    if rows.empty:
        raise RuntimeError("No ChEMBL PXR rank rows after filtering.")

    per_assay_mol = (
        rows.groupby(
            ["assay_chembl_id", "inchikey", "smiles", "mol_chembl_id"],
            as_index=False,
        )
        .agg(pchembl=("pchembl_value", "median"), n_rows=("pchembl_value", "size"))
        .reset_index(drop=True)
    )
    assay_sizes = per_assay_mol.groupby("assay_chembl_id")["inchikey"].transform("size")
    per_assay_mol = per_assay_mol[assay_sizes >= 5].copy()
    if per_assay_mol.empty:
        raise RuntimeError("No ChEMBL PXR assays with >=5 molecules.")

    per_assay_mol["assay_rank_pct"] = per_assay_mol.groupby("assay_chembl_id")[
        "pchembl"
    ].rank(method="average", pct=True)
    per_assay_mol["assay_high_hit"] = (
        (per_assay_mol["assay_rank_pct"] >= 0.80)
        | (per_assay_mol["pchembl"] >= 6.0)
    ).astype(float)

    mols = (
        per_assay_mol.groupby(["inchikey", "smiles", "mol_chembl_id"], as_index=False)
        .agg(
            chembl_rank_pct=("assay_rank_pct", "median"),
            chembl_rank_pct_max=("assay_rank_pct", "max"),
            chembl_rank_high_frac=("assay_high_hit", "mean"),
            chembl_rank_pchembl=("pchembl", "median"),
            chembl_rank_n_assays=("assay_chembl_id", "nunique"),
            chembl_rank_n_rows=("n_rows", "sum"),
        )
        .reset_index(drop=True)
    )
    return mols


def _load_challenge_keys() -> set[str]:
    keys = pd.read_sql(
        """
        SELECT d.inchikey
        FROM compounds c
        LEFT JOIN compound_descriptors d ON d.compound_id = c.id
        WHERE c.id IN (
            SELECT compound_id FROM train_activity
            UNION
            SELECT compound_id FROM test_activity
        )
        """,
        get_engine(),
    )["inchikey"].dropna()
    return set(keys)


def _weighted_topk(sim: np.ndarray, values: np.ndarray, k: int = 5) -> np.ndarray:
    if sim.shape[1] == 0:
        return np.full(sim.shape[0], np.nan)
    k_eff = min(k, sim.shape[1])
    idx = np.argpartition(sim, kth=sim.shape[1] - k_eff, axis=1)[:, -k_eff:]
    row = np.arange(sim.shape[0])[:, None]
    top_sim = sim[row, idx].astype(np.float64)
    top_values = values[idx].astype(np.float64)
    denom = top_sim.sum(axis=1)
    weighted = (top_sim * top_values).sum(axis=1)
    return np.divide(weighted, denom, out=np.zeros(sim.shape[0]), where=denom > 0)


def _chembl_rank_features_for(compounds: pd.DataFrame, external: pd.DataFrame) -> pd.DataFrame:
    ext_fp = _morgan_fp_matrix(external["smiles"].tolist())
    query_fp = _morgan_fp_matrix(compounds["smiles"].tolist())
    sim = tanimoto_matrix(query_fp, ext_fp)
    nn_idx = np.argmax(sim, axis=1)
    nn_sim = sim[np.arange(len(compounds)), nn_idx]

    pct = external["chembl_rank_pct"].to_numpy(dtype=np.float32)
    high = external["chembl_rank_high_frac"].to_numpy(dtype=np.float32)
    pchembl = external["chembl_rank_pchembl"].to_numpy(dtype=np.float32)
    n_assays = external["chembl_rank_n_assays"].to_numpy(dtype=np.float32)

    return pd.DataFrame(
        {
            "molecule_name": compounds["molecule_name"].to_numpy(),
            "compound_id": compounds["compound_id"].to_numpy(dtype=int),
            "chembl_rank_nn_sim": nn_sim,
            "chembl_rank_nn_pct": pct[nn_idx],
            "chembl_rank_nn_high_frac": high[nn_idx],
            "chembl_rank_nn_pchembl": pchembl[nn_idx],
            "chembl_rank_nn_n_assays": n_assays[nn_idx],
            "chembl_rank_top5_pct": _weighted_topk(sim, pct),
            "chembl_rank_top5_high_frac": _weighted_topk(sim, high),
            "chembl_rank_top5_pchembl": _weighted_topk(sim, pchembl),
            "chembl_rank_top5_sim_sum": np.sort(sim, axis=1)[:, -5:].sum(axis=1),
            "chembl_rank_covered_t025": nn_sim >= 0.25,
            "chembl_rank_covered_t030": nn_sim >= 0.30,
        }
    )


def build_chembl_rank_feature_tables(force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    pool_path = OUT_DIR / "pool_chembl_rank_features.csv"
    test_path = OUT_DIR / "test_chembl_rank_features.csv"
    if not force and pool_path.exists() and test_path.exists():
        return pd.read_csv(pool_path), pd.read_csv(test_path)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    external = load_chembl_pxr_rank_molecules()
    challenge_keys = _load_challenge_keys()
    external = external[~external["inchikey"].isin(challenge_keys)].reset_index(drop=True)
    if external.empty:
        raise RuntimeError("All ChEMBL rank molecules are exact challenge overlaps.")

    engine = get_engine()
    pool = pd.read_csv(
        REPO_ROOT
        / "track1_activity"
        / "analysis"
        / "phase2_validation_matrix"
        / "outputs"
        / "phase2_labeled_pool_with_folds.csv"
    )
    test = pd.read_sql(
        """
        SELECT
            t.id AS test_id,
            c.id AS compound_id,
            c.molecule_name,
            c.std_smiles AS smiles
        FROM test_activity t
        JOIN compounds c ON c.id = t.compound_id
        ORDER BY t.id
        """,
        engine,
    )

    pool_features = _chembl_rank_features_for(pool, external)
    test_features = _chembl_rank_features_for(test, external)
    coverage = pd.DataFrame(
        [
            {
                "split": "pool",
                "n": len(pool_features),
                "nn_ge_0.25": int(pool_features["chembl_rank_covered_t025"].sum()),
                "nn_ge_0.30": int(pool_features["chembl_rank_covered_t030"].sum()),
                "nn_max": float(pool_features["chembl_rank_nn_sim"].max()),
                "nn_median": float(pool_features["chembl_rank_nn_sim"].median()),
                "external_molecules": len(external),
            },
            {
                "split": "test",
                "n": len(test_features),
                "nn_ge_0.25": int(test_features["chembl_rank_covered_t025"].sum()),
                "nn_ge_0.30": int(test_features["chembl_rank_covered_t030"].sum()),
                "nn_max": float(test_features["chembl_rank_nn_sim"].max()),
                "nn_median": float(test_features["chembl_rank_nn_sim"].median()),
                "external_molecules": len(external),
            },
        ]
    )
    pool_features.to_csv(pool_path, index=False)
    test_features.to_csv(test_path, index=False)
    coverage.to_csv(OUT_DIR / "chembl_rank_coverage.csv", index=False)
    external.to_csv(OUT_DIR / "chembl_rank_external_molecules.csv", index=False)
    return pool_features, test_features


def build_htchem_rank_feature_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    pred = pred_htchem_for_challenge()
    lf = pd.read_parquet(LOG2FC_PATH).copy()
    lf["lf_mean"] = 0.5 * (lf["log2fc_8p25_pred"] + lf["log2fc_33_pred"])
    frame = pd.DataFrame(
        {
            "compound_id": pred.index.astype(int).to_numpy(),
            "pred_htchem": pred.to_numpy(dtype=float),
        }
    )
    frame = frame.merge(lf.reset_index()[["compound_id", "lf_mean"]], on="compound_id")
    frame["htchem_rank_pct"] = frame["pred_htchem"].rank(pct=True)
    frame["lf_rank_pct_for_htchem"] = frame["lf_mean"].rank(pct=True)
    frame["htchem_rank_z"] = (
        frame["pred_htchem"] - frame["pred_htchem"].mean()
    ) / frame["pred_htchem"].std()
    frame["lf_rank_z"] = (frame["lf_mean"] - frame["lf_mean"].mean()) / frame[
        "lf_mean"
    ].std()
    frame["htchem_minus_lf_z_rank"] = frame["htchem_rank_z"] - frame["lf_rank_z"]

    pool = pd.read_csv(
        REPO_ROOT
        / "track1_activity"
        / "analysis"
        / "phase2_validation_matrix"
        / "outputs"
        / "phase2_labeled_pool_with_folds.csv"
    )[["compound_id", "molecule_name"]]
    test = pd.read_sql(
        """
        SELECT c.id AS compound_id, c.molecule_name
        FROM test_activity t
        JOIN compounds c ON c.id = t.compound_id
        ORDER BY t.id
        """,
        get_engine(),
    )
    pool_features = pool.merge(
        frame[["compound_id", *HTCHEM_RANK_COLS]], on="compound_id", how="left"
    )
    test_features = test.merge(
        frame[["compound_id", *HTCHEM_RANK_COLS]], on="compound_id", how="left"
    )
    return pool_features, test_features


def build_pairrank_feature_tables(mode: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Load cached all-PXR pairwise assay-rank scores as scalar features."""

    cols = PAIRRANK_ALL_COLS if mode == "pairrank_all" else PAIRRANK_COLS
    pool_path = PAIRRANK_DIR / "pool_pairrank_scores.csv"
    test_path = PAIRRANK_DIR / "test_pairrank_scores.csv"
    if not pool_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Missing pairrank score files under {PAIRRANK_DIR}. "
            "Run run_pairwise_assay_rank_probe.py first."
        )
    pool_features = pd.read_csv(pool_path)[["compound_id", *cols]]
    test_features = pd.read_csv(test_path)[["compound_id", *cols]]
    return pool_features, test_features, cols


def external_rank_feature_matrices(
    mode: str, pool: pd.DataFrame, test: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return pool/test external rank matrices aligned to classifier inputs."""

    if mode == "none":
        return (
            np.empty((len(pool), 0), dtype=np.float32),
            np.empty((len(test), 0), dtype=np.float32),
            [],
        )
    frames_pool: list[pd.DataFrame] = []
    frames_test: list[pd.DataFrame] = []
    cols: list[str] = []
    if mode in {"chembl", "chembl_htchem"}:
        pool_chembl, test_chembl = build_chembl_rank_feature_tables()
        frames_pool.append(pool_chembl)
        frames_test.append(test_chembl)
        cols.extend(CHEMBL_FEATURE_COLS)
    if mode in {"htchem", "chembl_htchem"}:
        pool_ht, test_ht = build_htchem_rank_feature_tables()
        frames_pool.append(pool_ht)
        frames_test.append(test_ht)
        cols.extend(HTCHEM_RANK_COLS)
    if mode in {"pairrank", "pairrank_all"}:
        pool_pair, test_pair, pair_cols = build_pairrank_feature_tables(mode)
        frames_pool.append(pool_pair)
        frames_test.append(test_pair)
        cols.extend(pair_cols)
    if not cols:
        raise ValueError(f"Unknown external rank mode: {mode}")

    pool_extra = pd.concat(
        [
            frame.set_index("compound_id")[feature_cols]
            for frame, feature_cols in zip(
                frames_pool,
                [
                    CHEMBL_FEATURE_COLS if "chembl_rank_nn_sim" in frame.columns
                    else PAIRRANK_ALL_COLS if "pairrank_all" in frame.columns
                    else PAIRRANK_COLS if "pairrank_chembl" in frame.columns
                    else HTCHEM_RANK_COLS
                    for frame in frames_pool
                ],
            )
        ],
        axis=1,
    ).reindex(pool["compound_id"].astype(int))
    test_extra = pd.concat(
        [
            frame.set_index("compound_id")[feature_cols]
            for frame, feature_cols in zip(
                frames_test,
                [
                    CHEMBL_FEATURE_COLS if "chembl_rank_nn_sim" in frame.columns
                    else PAIRRANK_ALL_COLS if "pairrank_all" in frame.columns
                    else PAIRRANK_COLS if "pairrank_chembl" in frame.columns
                    else HTCHEM_RANK_COLS
                    for frame in frames_test
                ],
            )
        ],
        axis=1,
    ).reindex(test["compound_id"].astype(int))

    return (
        pool_extra.astype(float).to_numpy(dtype=np.float32),
        test_extra.astype(float).to_numpy(dtype=np.float32),
        cols,
    )

"""EDA analysis for Track 1 Activity Prediction."""

from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

DATA_DIR = Path(__file__).resolve().parent.parent.parent.joinpath("data")


def load_data():
    train = pd.read_parquet(DATA_DIR.joinpath("default_train.parquet"))
    test = pd.read_parquet(DATA_DIR.joinpath("default_test.parquet"))
    counter = pd.read_parquet(DATA_DIR.joinpath("counter_assay_train.parquet"))
    single = pd.read_parquet(DATA_DIR.joinpath("single_concentration_train.parquet"))
    return train, test, counter, single


def compute_mol_descriptors(smiles_series: pd.Series) -> pd.DataFrame:
    """Compute basic molecular descriptors from SMILES."""
    records = []
    invalid_count = 0
    for smi in smiles_series:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            invalid_count += 1
            records.append({
                "SMILES": smi,
                "valid": False,
                "MW": np.nan,
                "LogP": np.nan,
                "HBA": np.nan,
                "HBD": np.nan,
                "TPSA": np.nan,
                "RotBonds": np.nan,
                "NumRings": np.nan,
                "NumAromaticRings": np.nan,
                "NumHeavyAtoms": np.nan,
                "FractionCSP3": np.nan,
            })
        else:
            records.append({
                "SMILES": smi,
                "valid": True,
                "MW": Descriptors.MolWt(mol),
                "LogP": Descriptors.MolLogP(mol),
                "HBA": Lipinski.NumHAcceptors(mol),
                "HBD": Lipinski.NumHDonors(mol),
                "TPSA": Descriptors.TPSA(mol),
                "RotBonds": Lipinski.NumRotatableBonds(mol),
                "NumRings": Descriptors.RingCount(mol),
                "NumAromaticRings": Descriptors.NumAromaticRings(mol),
                "NumHeavyAtoms": Descriptors.HeavyAtomCount(mol),
                "FractionCSP3": Descriptors.FractionCSP3(mol),
            })
    if invalid_count > 0:
        print(f"  WARNING: {invalid_count} invalid SMILES found")
    return pd.DataFrame(records)


def print_section(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def main():
    train, test, counter, single = load_data()

    # --- Dataset Overview ---
    print_section("Dataset Overview")
    for name, df in [("Train", train), ("Test", test), ("Counter-assay", counter), ("Single-conc", single)]:
        print(f"  {name}: {len(df):,} rows x {len(df.columns)} cols")

    # --- Train columns ---
    print_section("Train Columns & Types")
    for col in train.columns:
        print(f"  {col}: {train[col].dtype}")

    # --- Train basic stats ---
    print_section("Train Numeric Statistics")
    print(train.describe().to_string())

    # --- Missing values ---
    print_section("Missing Values (Train)")
    nulls = train.isnull().sum()
    for col, n in nulls.items():
        if n > 0:
            print(f"  {col}: {n} ({n / len(train) * 100:.1f}%)")
    if nulls.sum() == 0:
        print("  No missing values")

    # --- pEC50 distribution ---
    print_section("pEC50 Distribution (Train)")
    pec50 = train["pEC50"]
    print(f"  Count:  {len(pec50)}")
    print(f"  Mean:   {pec50.mean():.3f}")
    print(f"  Std:    {pec50.std():.3f}")
    print(f"  Min:    {pec50.min():.3f}")
    print(f"  25%:    {pec50.quantile(0.25):.3f}")
    print(f"  Median: {pec50.median():.3f}")
    print(f"  75%:    {pec50.quantile(0.75):.3f}")
    print(f"  Max:    {pec50.max():.3f}")

    # --- Test data overview ---
    print_section("Test Data Overview")
    print(f"  Columns: {list(test.columns)}")
    print(f"  Has pEC50: {'pEC50' in test.columns}")
    nulls_test = test.isnull().sum()
    for col, n in nulls_test.items():
        if n > 0:
            print(f"  {col}: {n} nulls ({n / len(test) * 100:.1f}%)")

    # --- SMILES overlap ---
    print_section("SMILES Overlap Analysis")
    train_smi = set(train["SMILES"])
    test_smi = set(test["SMILES"])
    counter_smi = set(counter["SMILES"])
    single_smi = set(single["SMILES"])

    print(f"  Train unique SMILES:  {len(train_smi):,}")
    print(f"  Test unique SMILES:   {len(test_smi):,}")
    print(f"  Counter unique SMILES: {len(counter_smi):,}")
    print(f"  Single unique SMILES: {len(single_smi):,}")
    print()
    print(f"  Train & Test overlap:    {len(train_smi & test_smi)}")
    print(f"  Counter & Train overlap: {len(counter_smi & train_smi)}")
    print(f"  Counter & Test overlap:  {len(counter_smi & test_smi)}")
    print(f"  Single & Train overlap:  {len(single_smi & train_smi)}")
    print(f"  Single & Test overlap:   {len(single_smi & test_smi)}")

    # --- Molecular descriptors ---
    print_section("Molecular Descriptors (Train)")
    print("  Computing descriptors...")
    train_desc = compute_mol_descriptors(train["SMILES"])
    print(f"  Valid molecules: {train_desc['valid'].sum()} / {len(train_desc)}")
    desc_cols = ["MW", "LogP", "HBA", "HBD", "TPSA", "RotBonds", "NumRings",
                 "NumAromaticRings", "NumHeavyAtoms", "FractionCSP3"]
    print(train_desc[desc_cols].describe().round(2).to_string())

    print_section("Molecular Descriptors (Test)")
    print("  Computing descriptors...")
    test_desc = compute_mol_descriptors(test["SMILES"])
    print(f"  Valid molecules: {test_desc['valid'].sum()} / {len(test_desc)}")
    print(test_desc[desc_cols].describe().round(2).to_string())

    # --- Train vs Test descriptor comparison ---
    print_section("Train vs Test Descriptor Comparison (Mean)")
    print(f"  {'Descriptor':<20} {'Train':>10} {'Test':>10} {'Diff':>10}")
    print(f"  {'-' * 50}")
    for col in desc_cols:
        t_mean = train_desc[col].mean()
        e_mean = test_desc[col].mean()
        print(f"  {col:<20} {t_mean:>10.2f} {e_mean:>10.2f} {e_mean - t_mean:>10.2f}")

    # --- pEC50 vs descriptors correlation ---
    print_section("pEC50 vs Descriptors Correlation (Train)")
    train_with_desc = pd.concat(
        [train[["pEC50"]].reset_index(drop=True), train_desc[desc_cols].reset_index(drop=True)],
        axis=1,
    )
    corr = train_with_desc.corr()["pEC50"].drop("pEC50").sort_values(key=abs, ascending=False)
    for col, val in corr.items():
        print(f"  {col:<20} {val:>8.3f}")

    # --- Single-concentration overview ---
    print_section("Single-Concentration Data Overview")
    print(f"  Columns: {list(single.columns)}")
    print(f"  Unique experiments: {single['experiment_name'].nunique()}")
    print(f"  Unique concentrations: {single['concentration_M'].nunique()}")
    print(f"  Concentration values: {sorted(single['concentration_M'].unique())}")
    print()
    print("  log2_fc_estimate stats:")
    print(f"    Mean:   {single['log2_fc_estimate'].mean():.3f}")
    print(f"    Std:    {single['log2_fc_estimate'].std():.3f}")
    print(f"    Median: {single['log2_fc_estimate'].median():.3f}")

    # --- Counter-assay overview ---
    print_section("Counter-Assay Data Overview")
    print(f"  Columns: {list(counter.columns)}")
    counter_desc = counter.describe()
    print(counter_desc.to_string())

    print_section("Analysis Complete")


if __name__ == "__main__":
    main()

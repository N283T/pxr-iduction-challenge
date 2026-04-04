import marimo

__generated_with = "0.22.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md("# Track 1: Activity Prediction EDA")
    return (mo,)


@app.cell
def _(mo):
    import pandas as pd
    from pathlib import Path

    DATA_DIR = Path(__file__).resolve().parent.parent.parent.joinpath("data")

    train = pd.read_parquet(DATA_DIR.joinpath("default_train.parquet"))
    test = pd.read_parquet(DATA_DIR.joinpath("default_test.parquet"))
    counter = pd.read_parquet(DATA_DIR.joinpath("counter_assay_train.parquet"))
    single = pd.read_parquet(DATA_DIR.joinpath("single_concentration_train.parquet"))

    mo.md(f"""
    ## Dataset Overview

    | Dataset | Rows | Columns |
    |---------|------|---------|
    | Train (default) | {len(train):,} | {len(train.columns)} |
    | Test (blinded) | {len(test):,} | {len(test.columns)} |
    | Counter-assay | {len(counter):,} | {len(counter.columns)} |
    | Single-concentration | {len(single):,} | {len(single.columns)} |
    """)
    return counter, single, test, train


@app.cell
def _(mo):
    mo.md("""
    ## Train Data - Columns & Types
    """)
    return


@app.cell
def _(train):
    train.dtypes
    return


@app.cell
def _(mo):
    mo.md("""
    ## Train Data - Basic Statistics
    """)
    return


@app.cell
def _(train):
    train.describe()
    return


@app.cell
def _(mo):
    mo.md("""
    ## Train Data - First Rows
    """)
    return


@app.cell
def _(train):
    train.head(10)
    return


@app.cell
def _(mo, train):
    mo.md(f"""
    ## Missing Values (Train)

    {train.isnull().sum().to_frame("nulls").to_markdown()}
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## pEC50 Distribution (Train)
    """)
    return


@app.cell
def _(train):
    import plotly.express as px

    fig_pec50 = px.histogram(
        train,
        x="pEC50",
        nbins=50,
        title="pEC50 Distribution (Train)",
        labels={"pEC50": "pEC50 (-log10(M))"},
    )
    fig_pec50
    return (px,)


@app.cell
def _(mo):
    mo.md("""
    ## Emax vs pEC50 (Train)
    """)
    return


@app.cell
def _(px, train):
    fig_emax = px.scatter(
        train,
        x="pEC50",
        y="Emax_estimate (log2FC vs. baseline)",
        title="Emax vs pEC50",
        labels={
            "pEC50": "pEC50 (-log10(M))",
            "Emax_estimate (log2FC vs. baseline)": "Emax (log2FC)",
        },
        opacity=0.5,
    )
    fig_emax
    return


@app.cell
def _(mo):
    mo.md("""
    ## Test Data - Columns
    """)
    return


@app.cell
def _(test):
    test.dtypes
    return


@app.cell
def _(test):
    test.head(10)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Train vs Test SMILES Overlap
    """)
    return


@app.cell
def _(mo, test, train):
    train_smiles = set(train["SMILES"])
    test_smiles = set(test["SMILES"])
    overlap = train_smiles & test_smiles

    mo.md(f"""
    - Train unique SMILES: **{len(train_smiles):,}**
    - Test unique SMILES: **{len(test_smiles):,}**
    - Overlap: **{len(overlap)}**
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Single-Concentration Data Overview
    """)
    return


@app.cell
def _(single):
    single.describe()
    return


@app.cell
def _(mo, single, test, train):
    single_smiles = set(single["SMILES"])
    train_smiles_2 = set(train["SMILES"])
    test_smiles_2 = set(test["SMILES"])

    mo.md(f"""
    ## SMILES Overlap: Single-Concentration vs Train/Test

    - Single-conc unique SMILES: **{len(single_smiles):,}**
    - Overlap with Train: **{len(single_smiles & train_smiles_2)}**
    - Overlap with Test: **{len(single_smiles & test_smiles_2)}**
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Counter-Assay Data Overview
    """)
    return


@app.cell
def _(counter):
    counter.describe()
    return


@app.cell
def _(counter, mo, test, train):
    counter_smiles = set(counter["SMILES"])

    mo.md(f"""
    ## SMILES Overlap: Counter-Assay vs Train/Test

    - Counter-assay unique SMILES: **{len(counter_smiles):,}**
    - Overlap with Train: **{len(counter_smiles & set(train['SMILES']))}**
    - Overlap with Test: **{len(counter_smiles & set(test['SMILES']))}**
    """)
    return


if __name__ == "__main__":
    app.run()

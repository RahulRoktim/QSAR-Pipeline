import pandas as pd

from config import RAW_DATASET, PROCESSED_DATASET


def preprocess():

    print("\nLoading raw dataset...")

    df = pd.read_csv(RAW_DATASET)

    print(f"Original records : {len(df)}")

    # Keep only the columns we need
    df = df[
        [
            "molecule_chembl_id",
            "canonical_smiles",
            "pchembl_value",
            "standard_units",
            "standard_type",
        ]
    ]

    # Remove missing values
    df = df.dropna(subset=["canonical_smiles", "pchembl_value"])

    df = df.rename(
        columns={
            "canonical_smiles": "SMILES",
            "pchembl_value": "Activity",
        }
    )

    # Aggregate replicate measurements per molecule using the MEDIAN pChEMBL
    # value instead of dropping to an arbitrary single record. This keeps one
    # robust row per unique molecule and reduces measurement noise (a molecule
    # with 10 IC50 readings should contribute one consolidated activity, not be
    # thrown away down to whichever row happened to be first).
    df["Activity"] = pd.to_numeric(df["Activity"], errors="coerce")
    df = df.dropna(subset=["Activity"])

    df = df.groupby("SMILES", as_index=False).agg(
        molecule_chembl_id=("molecule_chembl_id", "first"),
        Activity=("Activity", "median"),
        standard_units=("standard_units", "first"),
        standard_type=("standard_type", "first"),
    )

    df = df[
        [
            "molecule_chembl_id",
            "SMILES",
            "Activity",
            "standard_units",
            "standard_type",
        ]
    ]

    print(f"Processed records : {len(df)}")

    df.to_csv(PROCESSED_DATASET, index=False)

    print(f"\nSaved to: {PROCESSED_DATASET}")


if __name__ == "__main__":
    preprocess()
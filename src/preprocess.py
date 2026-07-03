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

    # Keep only unique molecules
    df = df.drop_duplicates(subset="canonical_smiles")

    # Rename columns
    df = df.rename(
        columns={
            "canonical_smiles": "SMILES",
            "pchembl_value": "Activity",
        }
    )

    # Keep only the important columns
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
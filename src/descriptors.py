import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors

from config import PROCESSED_DATASET, DESCRIPTOR_DATASET


def calculate_descriptors():

    print("\nLoading processed dataset...")

    df = pd.read_csv(PROCESSED_DATASET)

    descriptor_names = [name for name, _ in Descriptors._descList]

    results = []

    total = len(df)

    for i, row in df.iterrows():

        smiles = row["SMILES"]

        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            continue

        data = {}

        data["SMILES"] = smiles
        data["Activity"] = row["Activity"]

        for name, function in Descriptors._descList:

            try:
                data[name] = function(mol)

            except Exception:

                data[name] = None

        results.append(data)

        if (i + 1) % 25 == 0 or i + 1 == total:

            print(f"{i+1}/{total} molecules processed")

    descriptor_df = pd.DataFrame(results)

    descriptor_df.to_csv(DESCRIPTOR_DATASET, index=False)

    print("\nDescriptor calculation complete!")

    print(f"Molecules : {len(descriptor_df)}")

    print(f"Descriptors : {len(descriptor_df.columns)-2}")

    print(f"Saved to: {DESCRIPTOR_DATASET}")


if __name__ == "__main__":

    calculate_descriptors()
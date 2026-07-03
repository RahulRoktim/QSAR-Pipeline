import joblib
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors


def calculate_descriptors(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError("Invalid SMILES.")

    descriptor_functions = Descriptors._descList

    descriptors = {}

    for name, func in descriptor_functions:

        try:
            descriptors[name] = func(mol)

        except Exception:
            descriptors[name] = 0

    return pd.DataFrame([descriptors])


def predict():

    print("\n==============================")
    print("Predict New Molecule")
    print("==============================")

    model = joblib.load("models/best_model.pkl")

    smiles = input("Enter SMILES: ")

    X = calculate_descriptors(smiles)

    prediction = model.predict(X)

    print("\nPredicted Activity")
    print("------------------------------")
    print(f"Prediction : {prediction[0]:.4f}")


if __name__ == "__main__":
    predict()
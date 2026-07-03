import joblib

from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd

# Load trained model
model = joblib.load("models/random_forest.pkl")

# New molecule
smiles = input("Enter SMILES: ")

mol = Chem.MolFromSmiles(smiles)

if mol is None:
    print("Invalid SMILES!")
    exit()

# Calculate descriptors
data = pd.DataFrame([{
    "MW": Descriptors.MolWt(mol),
    "LogP": Descriptors.MolLogP(mol),
    "TPSA": Descriptors.TPSA(mol),
    "HBA": Descriptors.NumHAcceptors(mol),
    "HBD": Descriptors.NumHDonors(mol),
    "RotatableBonds": Descriptors.NumRotatableBonds(mol)
}])

prediction = model.predict(data)

print("\nPredicted Activity:", prediction[0])
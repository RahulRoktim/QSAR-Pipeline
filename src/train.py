import pandas as pd

from sklearn.model_selection import train_test_split

from src.trainer import train_all_models
from src.comparison import save_results

from config import (
    SELECTED_DATASET,
    TEST_SIZE,
    RANDOM_STATE,
)


def train_model():

    print("\n==============================")
    print("Training Model")
    print("==============================")

    # Load selected features
    df = pd.read_csv(SELECTED_DATASET)

    # Features
    X = df.drop(columns=["SMILES", "Activity"])

    # Target
    y = df["Activity"]

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    print(f"Training molecules : {len(X_train)}")
    print(f"Testing molecules  : {len(X_test)}")

    output = train_all_models(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    save_results(output["results"])

    print("\n==============================")
    print("Training Complete")
    print("==============================")
    print(f"Best Model  : {output['name']}")
    print(f"CV R²       : {output['cv_score']:.3f}")
    print(f"Test R²     : {output['test_score']:.3f}")

    return output["model"]


if __name__ == "__main__":
    train_model()
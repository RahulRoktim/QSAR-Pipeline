import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    root_mean_squared_error,
)


def evaluate_model():

    print("\n==============================")
    print("Evaluating Model")
    print("==============================")

    # Load model
    model = joblib.load("models/random_forest.pkl")

    # Load selected features
    df = pd.read_csv("outputs/selected_features.csv")

    # Features
    X = df.drop(columns=["SMILES", "Activity"])

    # Target
    y = df["Activity"]

    # Same split used during training
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    # Predict only on test set
    predictions = model.predict(X_test)

    # Metrics
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = root_mean_squared_error(y_test, predictions)

    print("\nEvaluation Results")
    print("------------------------------")
    print(f"R²   : {r2:.3f}")
    print(f"MAE  : {mae:.3f}")
    print(f"RMSE : {rmse:.3f}")

    # =====================================
    # Predicted vs Actual Plot
    # =====================================

    plt.figure(figsize=(6,6))

    plt.scatter(
        y_test,
        predictions,
        alpha=0.7
    )

    minimum = min(y_test.min(), predictions.min())
    maximum = max(y_test.max(), predictions.max())

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        "r--",
        linewidth=2
    )

    plt.xlabel("Actual Activity")
    plt.ylabel("Predicted Activity")
    plt.title("Predicted vs Actual")

    plt.tight_layout()

    plt.savefig(
        "outputs/predicted_vs_actual.png",
        dpi=300
    )

    plt.close()

    print("\nSaved:")
    print("outputs/predicted_vs_actual.png")

    return {
        "R2": r2,
        "MAE": mae,
        "RMSE": rmse,
    }


if __name__ == "__main__":
    evaluate_model()
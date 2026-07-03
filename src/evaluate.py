import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    root_mean_squared_error,
)

from config import (
    SELECTED_DATASET,
    BEST_MODEL_PATH,
    TEST_SIZE,
    RANDOM_STATE,
)


def evaluate_model():

    print("\n==============================")
    print("Evaluating Best Model")
    print("==============================")

    # Load best model
    model = joblib.load(BEST_MODEL_PATH)

    # Load dataset
    df = pd.read_csv(SELECTED_DATASET)

    X = df.drop(columns=["SMILES", "Activity"])
    y = df["Activity"]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    predictions = model.predict(X_test)

    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = root_mean_squared_error(y_test, predictions)

    print("\nEvaluation Results")
    print("------------------------------")
    print(f"R²   : {r2:.3f}")
    print(f"MAE  : {mae:.3f}")
    print(f"RMSE : {rmse:.3f}")

    # ===============================
    # Predicted vs Actual
    # ===============================

    plt.figure(figsize=(6, 6))

    plt.scatter(
        y_test,
        predictions,
        alpha=0.7,
    )

    low = min(y_test.min(), predictions.min())
    high = max(y_test.max(), predictions.max())

    plt.plot(
        [low, high],
        [low, high],
        "r--",
        linewidth=2,
    )

    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Predicted vs Actual")

    plt.tight_layout()

    plt.savefig(
        "outputs/predicted_vs_actual.png",
        dpi=300,
    )

    plt.close()

    # ===============================
    # Residual Plot
    # ===============================

    residuals = y_test - predictions

    plt.figure(figsize=(6, 6))

    plt.scatter(
        predictions,
        residuals,
        alpha=0.7,
    )

    plt.axhline(
        y=0,
        color="red",
        linestyle="--",
    )

    plt.xlabel("Predicted")
    plt.ylabel("Residual")

    plt.title("Residual Plot")

    plt.tight_layout()

    plt.savefig(
        "outputs/residual_plot.png",
        dpi=300,
    )

    plt.close()

    print("\nSaved:")
    print("outputs/predicted_vs_actual.png")
    print("outputs/residual_plot.png")

    return {
        "R2": r2,
        "MAE": mae,
        "RMSE": rmse,
    }


if __name__ == "__main__":
    evaluate_model()
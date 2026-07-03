import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline

from sklearn.model_selection import (
    train_test_split,
    KFold,
    cross_val_score,
)

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    root_mean_squared_error,
)

from src.feature_selector import FeatureSelector

def train_model():

    print("\n==============================")
    print("Training Model")
    print("==============================")

    # Load selected features
    df = pd.read_csv("outputs/selected_features.csv")

    # Features
    X = df.drop(columns=["SMILES", "Activity"])

    # Target
    y = df["Activity"]

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    print(f"Training molecules : {len(X_train)}")
    print(f"Testing molecules  : {len(X_test)}")

    pipeline = Pipeline([
    (
        "feature_selector",
        FeatureSelector()
    ),
    (
        "model",
        RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )
    )
])

    pipeline.fit(X_train, y_train)

    # ==============================
    # 5-Fold Cross Validation
    # ==============================

    print("\n==============================")
    print("5-Fold Cross Validation")
    print("==============================")

    kfold = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    cv_scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=kfold,
        scoring="r2"
    )

    for i, score in enumerate(cv_scores):
        print(f"Fold {i+1} R² : {score:.3f}")

    print("------------------------------")
    print(f"Mean R² : {cv_scores.mean():.3f}")
    print(f"Std  R² : {cv_scores.std():.3f}")

    # Predict
    predictions = pipeline.predict(X_test)

    # Metrics
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = root_mean_squared_error(y_test, predictions)

    print("\nModel Performance")
    print("------------------------------")
    print(f"R²   : {r2:.3f}")
    print(f"MAE  : {mae:.3f}")
    print(f"RMSE : {rmse:.3f}")

    # Save Model
    joblib.dump(
    pipeline,
    "models/random_forest.pkl"
)

    print("\nModel saved to models/random_forest.pkl")

    # ==========================================
    # Feature Importance
    # ==========================================

    selector = pipeline.named_steps["feature_selector"]
    rf = pipeline.named_steps["model"]

    selected_features = selector.get_feature_names_out(X.columns)

    importance = pd.DataFrame({
        "Descriptor": selected_features,
        "Importance": rf.feature_importances_
    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    importance.to_csv(
        "outputs/feature_importance.csv",
        index=False
    )

    print("\nTop 10 Most Important Descriptors")
    print("----------------------------------")
    print(importance.head(10))

    # Plot Top 20 Features

    top20 = importance.head(20)

    plt.figure(figsize=(10, 7))
    plt.barh(
        top20["Descriptor"],
        top20["Importance"]
    )
    plt.gca().invert_yaxis()
    plt.xlabel("Importance")
    plt.ylabel("Descriptor")
    plt.title("Top 20 Feature Importances")
    plt.tight_layout()

    plt.savefig(
        "outputs/feature_importance.png",
        dpi=300
    )

    plt.close()

    print("\nFeature importance saved:")
    print("outputs/feature_importance.csv")
    print("outputs/feature_importance.png")

    # Run Y-Randomization
    y_randomization_test(X, y)

    return pipeline

def y_randomization_test(X, y):

    print("\n==============================")
    print("Y-Randomization Test")
    print("==============================")

    kfold = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    random_scores = []

    for i in range(10):

        # Shuffle activity values
        shuffled_y = np.random.permutation(y)

        random_model = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )

        scores = cross_val_score(
            random_model,
            X,
            shuffled_y,
            cv=kfold,
            scoring="r2"
        )

        mean_score = scores.mean()
        random_scores.append(mean_score)

        print(f"Randomization {i+1:2d}: R² = {mean_score:.3f}")

    print("------------------------------")
    print(f"Average Random R² : {np.mean(random_scores):.3f}")
    print(f"Best Random R²    : {np.max(random_scores):.3f}")

if __name__ == "__main__":
    train_model()
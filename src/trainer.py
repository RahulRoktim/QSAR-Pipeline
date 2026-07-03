import joblib

from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    root_mean_squared_error,
)
from sklearn.model_selection import (
    KFold,
    cross_val_score,
)

from src.feature_selector import FeatureSelector
from src.models import get_models
from src.tuning import tune_model

from config import (
    HYPERPARAMETER_SEARCH,
    RANDOM_STATE,
    CV_FOLDS,
    BEST_MODEL_PATH,
)

PARAM_GRIDS = {

    "Random Forest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20],
    },

    "Extra Trees": {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20],
    },

    "Gradient Boosting": {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
    },

    "Support Vector Regression": {
        "C": [1, 10, 100],
        "gamma": ["scale", "auto"],
    },
}


def train_all_models(
    X_train,
    X_test,
    y_train,
    y_test,
):

    print("\n==============================")
    print("Training Multiple Models")
    print("==============================")

    models = get_models()

    results = []

    best_model = None
    best_name = None
    best_r2 = float("-inf")

    for name, model in models.items():

        print(f"\n{name}")

        if HYPERPARAMETER_SEARCH and name in PARAM_GRIDS:

            print("Running Hyperparameter Search...")

            model = tune_model(
                model,
                PARAM_GRIDS[name],
                X_train,
                y_train,
            )

        pipeline = Pipeline([
            (
                "feature_selector",
                FeatureSelector(),
            ),
            (
                "model",
                model,
            ),
        ])

        pipeline.fit(
            X_train,
            y_train,
        )

        predictions = pipeline.predict(X_test)

        r2 = r2_score(
            y_test,
            predictions,
        )

        mae = mean_absolute_error(
            y_test,
            predictions,
        )

        rmse = root_mean_squared_error(
            y_test,
            predictions,
        )

        cv = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=KFold(
                n_splits=CV_FOLDS,
                shuffle=True,
                random_state=RANDOM_STATE,
            ),
            scoring="r2",
        )

        print(f"R²   : {r2:.3f}")
        print(f"MAE  : {mae:.3f}")
        print(f"RMSE : {rmse:.3f}")
        print(f"CV   : {cv.mean():.3f}")

        results.append({

            "Model": name,
            "R2": r2,
            "MAE": mae,
            "RMSE": rmse,
            "CV_R2": cv.mean(),

        })

        if r2 > best_r2:

            best_r2 = r2
            best_model = pipeline
            best_name = name

    print("\n==============================")
    print("Best Model")
    print("==============================")

    print(best_name)
    print(f"R² : {best_r2:.3f}")

    joblib.dump(
        best_model,
        BEST_MODEL_PATH,
    )

    return {
        "model": best_model,
        "name": best_name,
        "results": results,
        "score": best_r2,
    }
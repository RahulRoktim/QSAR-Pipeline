import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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
    best_cv = float("-inf")
    best_test_r2 = float("-inf")

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
                # Standardize descriptors. Scale-sensitive models (SVR, and any
                # linear/kernel model) are crippled without this; fit inside the
                # pipeline so CV folds never see test-fold statistics.
                "scaler",
                StandardScaler(),
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

        cv_mean = cv.mean()

        print(f"Test R² : {r2:.3f}")
        print(f"MAE     : {mae:.3f}")
        print(f"RMSE    : {rmse:.3f}")
        print(f"CV R²   : {cv_mean:.3f} (+/- {cv.std():.3f})")

        results.append({

            "Model": name,
            "R2": r2,
            "MAE": mae,
            "RMSE": rmse,
            "CV_R2": cv_mean,
            "CV_R2_STD": cv.std(),

        })

        # Select by CROSS-VALIDATED R², not test-set R². Choosing the model by
        # its test score leaks the held-out set into model selection and reports
        # an optimistically biased number.
        if cv_mean > best_cv:

            best_cv = cv_mean
            best_test_r2 = r2
            best_model = pipeline
            best_name = name

    print("\n==============================")
    print("Best Model (selected by CV R²)")
    print("==============================")

    print(best_name)
    print(f"CV R²   : {best_cv:.3f}")
    print(f"Test R² : {best_test_r2:.3f}")

    if best_cv <= 0:
        print(
            "\nWARNING: best cross-validated R² is <= 0. The model has no "
            "predictive power on this dataset (typically caused by too few "
            "compounds relative to the number of descriptors). Treat "
            "predictions as unreliable until more training data is added."
        )

    joblib.dump(
        best_model,
        BEST_MODEL_PATH,
    )

    return {
        "model": best_model,
        "name": best_name,
        "results": results,
        "cv_score": best_cv,
        "test_score": best_test_r2,
    }
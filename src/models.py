from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
)

from sklearn.svm import SVR


def get_models():

    models = {}

    # ---------------------------------
    # Random Forest
    # ---------------------------------

    models["Random Forest"] = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )

    # ---------------------------------
    # Extra Trees
    # ---------------------------------

    models["Extra Trees"] = ExtraTreesRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )

    # ---------------------------------
    # Gradient Boosting
    # ---------------------------------

    models["Gradient Boosting"] = GradientBoostingRegressor(
        random_state=42,
    )

    # ---------------------------------
    # Support Vector Regression
    # ---------------------------------

    models["Support Vector Regression"] = SVR(
        kernel="rbf",
        C=10,
        gamma="scale",
    )

    # ---------------------------------
    # XGBoost (Optional)
    # ---------------------------------

    try:

        from xgboost import XGBRegressor

        models["XGBoost"] = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            objective="reg:squarederror",
        )

    except ImportError:

        print("XGBoost not installed. Skipping.")

    # ---------------------------------
    # LightGBM (Optional)
    # ---------------------------------

    try:

        from lightgbm import LGBMRegressor

        models["LightGBM"] = LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            random_state=42,
            verbose=-1,
        )

    except ImportError:

        print("LightGBM not installed. Skipping.")

    # ---------------------------------
    # CatBoost (Optional)
    # ---------------------------------

    try:

        from catboost import CatBoostRegressor

        models["CatBoost"] = CatBoostRegressor(
            iterations=300,
            learning_rate=0.05,
            verbose=False,
            random_state=42,
        )

    except ImportError:

        print("CatBoost not installed. Skipping.")

    return models
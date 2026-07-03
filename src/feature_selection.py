import pandas as pd
import numpy as np


def feature_selection():

    print("\n==============================")
    print("Feature Selection")
    print("==============================")

    # Load descriptor dataset
    df = pd.read_csv("outputs/descriptors.csv")

    print(f"Original shape: {df.shape}")

    # Save target column
    activity = df["Activity"]

    # Remove non-feature columns
    X = df.drop(columns=["SMILES", "Activity"])

    # ---------------------------------------
    # Remove constant descriptors
    # ---------------------------------------
    constant_cols = [
        col for col in X.columns
        if X[col].nunique() <= 1
    ]

    print(f"Constant descriptors removed: {len(constant_cols)}")

    X = X.drop(columns=constant_cols)

    # ---------------------------------------
    # Remove descriptors containing NaN
    # ---------------------------------------
    missing_cols = X.columns[X.isnull().any()]

    print(f"Descriptors with NaN removed: {len(missing_cols)}")

    X = X.drop(columns=missing_cols)

    # ---------------------------------------
    # Remove highly correlated descriptors
    # ---------------------------------------
    corr_matrix = X.corr().abs()

    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    to_drop = [
        column
        for column in upper.columns
        if any(upper[column] > 0.95)
    ]

    print(f"Highly correlated descriptors removed: {len(to_drop)}")

    X = X.drop(columns=to_drop)

    # ---------------------------------------
    # Reassemble dataset
    # ---------------------------------------
    final_df = pd.concat(
        [
            df["SMILES"],
            activity,
            X
        ],
        axis=1
    )

    print(f"Final shape: {final_df.shape}")

    output_file = "outputs/selected_features.csv"

    final_df.to_csv(output_file, index=False)

    print(f"Saved to: {output_file}")

    return final_df


if __name__ == "__main__":
    feature_selection()
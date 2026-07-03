import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin


class FeatureSelector(BaseEstimator, TransformerMixin):

    def __init__(self, correlation_threshold=0.95):
        self.correlation_threshold = correlation_threshold

    def fit(self, X, y=None):

        X = X.copy()

        # Remove constant descriptors
        self.constant_cols_ = [
            col for col in X.columns
            if X[col].nunique() <= 1
        ]

        X = X.drop(columns=self.constant_cols_)

        # Remove NaN descriptors
        self.nan_cols_ = X.columns[X.isnull().any()].tolist()

        X = X.drop(columns=self.nan_cols_)

        # Remove highly correlated descriptors
        corr_matrix = X.corr().abs()

        upper = corr_matrix.where(
            np.triu(
                np.ones(corr_matrix.shape),
                k=1
            ).astype(bool)
        )

        self.corr_cols_ = [
            column
            for column in upper.columns
            if any(upper[column] > self.correlation_threshold)
        ]

        return self

    def transform(self, X):

        X = X.copy()

        X = X.drop(
            columns=self.constant_cols_,
            errors="ignore"
        )

        X = X.drop(
            columns=self.nan_cols_,
            errors="ignore"
        )

        X = X.drop(
            columns=self.corr_cols_,
            errors="ignore"
        )

        return X

    def get_feature_names_out(self, input_features=None):

        if input_features is None:
            return None

        features = list(input_features)

        features = [
            f for f in features
            if f not in self.constant_cols_
        ]

        features = [
            f for f in features
            if f not in self.nan_cols_
        ]

        features = [
            f for f in features
            if f not in self.corr_cols_
        ]

        return np.array(features)
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Placeholder feature selector for Pipeline compatibility.

    Later versions can replace this with variance threshold,
    correlation filtering, Boruta, RFECV, SHAP, etc.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X
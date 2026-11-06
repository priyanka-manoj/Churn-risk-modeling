import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

class TotalChargesCleaner(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = X.copy()

        X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")

        X["TotalCharges"] = X["TotalCharges"].fillna(X["tenure"]*X["MonthlyCharges"].round(2))

        return X

# Logistic Regression 
def build_lr_preprocessor(num_features, cat_features):

    num_pipeline = Pipeline([
        ("scaler", StandardScaler())       
    ])

    cat_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_features),
        ("cat", cat_pipeline, cat_features)
    ])

    return preprocessor

# For ensemble techniques
def build_tree_preprocessor(num_features, cat_features):
    
    preprocessor = ColumnTransformer([
        ("num", "passthrough", num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
    ])

    return preprocessor

# Could have done this in the beginning! Adding a separate class now as existing pipelines work well. 
class CatBoostPreprocessor(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = X.copy()

        X["SeniorCitizen"] = X["SeniorCitizen"].astype(str)

        return X


class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # Does the effect of tenure change as customers become more established?
        X["tenure_squared"] = X["tenure"] ** 2
        # Does the relationship between monthly charges and churn depend on how long a customer has been subscribed?
        X["tenure_x_monthly"] = X["tenure"] * X["MonthlyCharges"]

        return X
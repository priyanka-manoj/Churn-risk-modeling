import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from steps.preprocessing import (
    TotalChargesCleaner,
    build_lr_preprocessor,
    ChurnFeatureEngineer
)


def build_lr_feature_pipeline(num_features, cat_features, class_weight=None):

    engineered_num_features = num_features + [
        "tenure_squared",
        "tenure_x_monthly"
    ]

    return Pipeline([
        ("cleaner", TotalChargesCleaner()),
        ("features", ChurnFeatureEngineer()),
        ("preprocessor", build_lr_preprocessor(
            engineered_num_features, cat_features
        )),
        ("model", LogisticRegression(max_iter=3000, random_state=42, class_weight=class_weight))
        
    ])
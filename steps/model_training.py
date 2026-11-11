import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

from steps.preprocessing import (
    TotalChargesCleaner,
    CatBoostPreprocessor,
    build_lr_preprocessor,
    build_tree_preprocessor,
    ChurnFeatureEngineer
)

# Logistic Regression
def build_lr_pipeline(
    num_features,
    cat_features,
    C=1.0,
    l1_ratio=0.0,
    class_weight=None,
    solver="lbfgs"
):

    
    return Pipeline([
        ("cleaner", TotalChargesCleaner()),
        ("preprocessor", build_lr_preprocessor(num_features, cat_features)),
        ("model", LogisticRegression(
            C=C,
            l1_ratio=l1_ratio,
            class_weight=class_weight,
            solver=solver,
            max_iter=3000,
            random_state=42
        ))
    ])

# Random Forest
def build_rf_pipeline(num_features, cat_features, class_weight = None):

    preprocessor = build_tree_preprocessor(num_features, cat_features)

    pipeline = Pipeline([
        ("cleaner", TotalChargesCleaner()),
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=400,
            random_state=42,
            class_weight=class_weight,
            n_jobs=-1))
    ])

    return pipeline

# Gradient Boosting
def build_gb_pipeline(num_features, cat_features):

    preprocessor = build_tree_preprocessor(num_features, cat_features)

    pipeline = Pipeline([
        ("cleaner", TotalChargesCleaner()),
        ("preprocessor", preprocessor),
        ("model", GradientBoostingClassifier(
            n_estimators = 300,
            learning_rate = 0.1,
            max_depth = 3,
            random_state=42
        ))
    ])

    return pipeline


# CatBoost
def build_catboost_pipeline(auto_class_weights=None):

    return Pipeline([
        ("cleaner", TotalChargesCleaner()),
        ("preprocessor", CatBoostPreprocessor()),
        ("model", CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.05,
            auto_class_weights=auto_class_weights,
            random_seed=42,
            verbose=False
        ))
    ])

# Support Vector Machine
def build_svm_pipeline(num_features, cat_features, kernel = "linear", class_weight=None):

    return Pipeline([
        ("cleaner", TotalChargesCleaner()),
        ("preprocessor", build_lr_preprocessor(num_features, cat_features)),
        ("model", SVC(kernel = kernel,
                      C = 1.0,
                      gamma = "scale",
                      class_weight = class_weight,
                      random_state=42
            
        ))
    ])
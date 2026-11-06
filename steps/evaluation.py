import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

def cross_validate_model(model, X_train, y_train, fit_params=None):

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scoring = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc"
    ]

    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        params = fit_params
    )

    return pd.DataFrame(scores).filter(like="test_").agg(
        ["mean", "std"]
    ).round(3)

def evaluate_model(model, X, y):

    y_pred = model.predict(X)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X)[:, 1]
    else:
        y_score = model.decision_function(X)

    metrics = {
        "Accuracy": accuracy_score(y, y_pred),
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1": f1_score(y, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y, y_score)
    }
    print("\nClassification Report:")
    print(
    classification_report(
        y,
        y_pred,
        target_names=["Stay", "Churn"],
        digits=3,
        zero_division=0
        )
    )
    return pd.Series(metrics).round(3)

def plot_confusion_matrix(model, X, y, labels=("Stay", "Churn")):

    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred, labels=[0, 1])

    # Percentage of all observations
    cm_pct = cm / cm.sum() * 100

    # Create annotations: count + percentage
    annotations = [
        [f"{count}\n({pct:.1f}%)" for count, pct in zip(row, pct_row)]
        for row, pct_row in zip(cm, cm_pct)
    ]

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=annotations,
        fmt="",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        annot_kws={"size": 12}
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()
import joblib
import pandas as pd

from config import MODEL_PATH
from steps.risk_stratification import assign_risk_tier


def predict_churn(customer_data):
    """
    Predict churn probability and assign risk tiers.

    Parameters
    ----------
    customer_data : pd.DataFrame
        Raw customer records containing the required model features.

    Returns
    -------
    pd.DataFrame
        Calibrated churn probabilities and corresponding risk tiers.
    """

    # Load the fitted calibrated pipeline
    model = joblib.load(MODEL_PATH)

    # Generate calibrated churn probabilities
    probabilities = model.predict_proba(customer_data)[:, 1]

    # Assign risk tiers
    risk_tiers = assign_risk_tier(probabilities)

    # Prepare prediction results
    results = pd.DataFrame({
        "churn_probability": probabilities,
        "risk_tier": risk_tiers
    }, index=customer_data.index)

    return results
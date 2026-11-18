import pandas as pd
import numpy as np

from config import RISK_THRESHOLDS, RISK_LABELS

def assign_risk_tier(probabilities):

    """
    Assigns risk tiers based on predicted probabilities.

    Parameters:
    - probabilities: A pandas Series or numpy array of predicted probabilities.

    Returns:
    - A pandas Series of risk tiers.
    """
    probabilities = pd.Series(probabilities, dtype=float)  # Ensure input is a pandas Series for consistency

    if probabilities.empty:
        raise ValueError("Input probabilities are empty. Please provide a non-empty array or Series.")
    
    if probabilities.isnull().any():
        raise ValueError("Input probabilities contain NaN values. Please handle missing values before assigning risk tiers.")
    
    if not ((probabilities >= 0) & (probabilities <= 1)).all():
        raise ValueError("Input probabilities must be in the range [0, 1].")

    thresholds = [0.0, *RISK_THRESHOLDS, np.inf]

    # Use pd.cut to assign risk tiers based on the thresholds
    risk_tiers = pd.cut(probabilities, bins=thresholds, labels=RISK_LABELS, right=False)

    return risk_tiers
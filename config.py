from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "calibrated_lr.joblib"

# Finalized churn-risk boundaries
RISK_THRESHOLDS = [0.30, 0.50, 0.70]

RISK_LABELS = [
    "Low Risk",
    "Medium Risk",
    "High Risk",
    "Very High Risk",
]
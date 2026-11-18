from pathlib import Path
import pandas as pd
import joblib
import numpy as np

from steps.risk_stratification import assign_risk_tier

from pipeline import predict_churn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT/ "models" / "calibrated_lr.joblib"

# Raw synthetic customer record with valid features
customer = pd.DataFrame([{
    "tenure": 6,
    "MonthlyCharges": 85.50,
    "TotalCharges": "513.00",
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check"
}])

def test_calibrated_pipeline_prediction():

    test_customer = customer.copy()

    # load the fitted model
    model = joblib.load(MODEL_PATH)

    probabilities = model.predict_proba(test_customer)[:,1]

    risk_tier = assign_risk_tier(probabilities)

    assert len(probabilities) == 1
    assert np.isfinite(probabilities)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    assert len(risk_tier) == 1
    assert pd.notna(risk_tier.iloc[0])
    

def test_prediction_output():
    
    test_customer = customer.copy()
    results = predict_churn(test_customer)

    assert len(results) == 1
    assert "churn_probability" in results.columns
    assert "risk_tier" in results.columns
    assert results["churn_probability"].between(0, 1).all()
    assert results["risk_tier"].notna().all()    
import pytest
import numpy as np

from steps.risk_stratification import assign_risk_tier


def test_assign_risk_tier_normal_cases():
    probabilities = [0.10, 0.40, 0.60, 0.80]

    expected = [
        "Low Risk",
        "Medium Risk",
        "High Risk",
        "Very High Risk"
    ]

    actual = assign_risk_tier(probabilities)

    assert actual.tolist() == expected

@pytest.mark.parametrize(
     "probability, expected",
     [
         (0.0, "Low Risk"),
        (0.2999, "Low Risk"),
        (0.30, "Medium Risk"),
        (0.50, "High Risk"),
        (0.70, "Very High Risk"),
        (1.0, "Very High Risk"),
     ]
 )   

def test_risk_tier_boundaries(probability, expected):
    actual = assign_risk_tier([probability])

    assert actual.iloc[0] == expected

@pytest.mark.parametrize(
        "probabilities",
        [ [-0.1], # Below valid range
         [1.2], # Above valid range
         [np.nan], # Missing value
         [np.inf], # positive infinity
         [-np.inf], # negative infinity
         [], # Empty input
         [0.3, np.nan] # Mixed values

        ]
)

def test_invalid_probabilities(probabilities):

    with pytest.raises(ValueError):
        assign_risk_tier(probabilities)

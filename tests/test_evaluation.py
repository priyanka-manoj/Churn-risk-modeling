import pandas as pd
import numpy as np

from steps.evaluation import calculate_ece

def test_ece_perfect_predictions():

    y_true = [0, 0, 1, 1]
    y_prob = [0.0, 0.0, 1.0, 1.0]

    ece, summary = calculate_ece(y_true, y_prob)

    assert np.isclose(ece, 0.0) 
    assert summary["customers"].sum() == 4

def test_ece_known_value():

    y_true = [0, 0, 1, 1]
    y_prob = [0.20, 0.20, 0.80, 0.80]

    ece, summary = calculate_ece(y_true, y_prob)
    # cal gap for non-churner bin 0.20, cal gap for churner bin 0.20, ece = 0.5*0.20 + 0.5*0.20
    assert np.isclose(ece, 0.20)
    assert summary["customers"].sum() == 4

# tests whether ece function works well when some prob bins do not contain any customers
def test_ece_empty_bins():
    y_true = [0, 1]
    y_prob = [0.05, 0.95]

    ece, summary = calculate_ece(
        y_true, y_prob, n_bins=10
    )

    assert np.isclose(ece, 0.05)
    assert len(summary) == 2
    assert summary["customers"].sum() == 2
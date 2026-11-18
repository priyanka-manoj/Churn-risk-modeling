import pytest
import pandas as pd
import numpy as np
from steps.preprocessing import TotalChargesCleaner


def test_total_charges_imputation():
    df = pd.DataFrame(
        {
            "tenure" : [0.0, 5.0, 24],
            "MonthlyCharges" : [60, 70, 50],
            "TotalCharges" : [0.0, " ", 1200]
        }


    )

    cleaner = TotalChargesCleaner()
    result = cleaner.fit_transform(df)

    expected = [0.0, 350.0, 1200.0]

    assert result["TotalCharges"].to_list() == expected

import numpy as np
import pandas as pd
import pytest

from steps.preprocessing import TotalChargesCleaner


@pytest.mark.parametrize(
    "total_charges, expected",
    [
        ("350", 350.0),
        ("350.50", 350.50),
        (350, 350.0),
        (350.50, 350.50),
        (" 350.50 ", 350.50),
    ]
)
def test_total_charges_numeric_conversion(total_charges, expected):

    df = pd.DataFrame({
        "tenure": [5],
        "MonthlyCharges": [70.0],
        "TotalCharges": [total_charges]
    })

    result = TotalChargesCleaner().fit_transform(df)

    assert result["TotalCharges"].iloc[0] == pytest.approx(expected)

@pytest.mark.parametrize(
    "total_charges",
    [
        "",
        " ",
        np.nan,
        None,
    ]
)
def test_total_charges_missing_values(total_charges):

    df = pd.DataFrame({
        "tenure": [5],
        "MonthlyCharges": [70.0],
        "TotalCharges": [total_charges]
    })

    result = TotalChargesCleaner().fit_transform(df)

    assert result["TotalCharges"].iloc[0] == pytest.approx(350.0)

def test_total_charges_does_not_modify_input():

    df = pd.DataFrame({
        "tenure": [0, 5],
        "MonthlyCharges": [50.0, 70.0],
        "TotalCharges": [" ", "350"]
    })

    original = df.copy(deep=True)

    TotalChargesCleaner().fit_transform(df)

    pd.testing.assert_frame_equal(df, original)

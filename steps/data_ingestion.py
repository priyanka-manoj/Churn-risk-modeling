import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split_data(filepath, random_state=10):
    
    df = pd.read_csv(filepath)

    X = df.drop(columns=["customerID", "Churn"])
    y = df["Churn"].map({"No": 0, "Yes": 1})

    # First split 70 % train, 30% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size = 0.3,
        random_state = random_state,
        stratify = y
    )

    # Second split, divide temp equally

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size = 0.5,
        random_state = random_state,
        stratify = y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
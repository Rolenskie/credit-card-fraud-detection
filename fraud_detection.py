"""
Credit card fraud detection pipeline.

Loads the Kaggle MLG-ULB credit card dataset, handles the severe class
imbalance, trains and compares models, and reports the metrics that
matter for a rare-event problem.

Usage:
    place creditcard.csv in ./data/, then: python fraud_detection.py
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

DATA_PATH = Path("data") / "creditcard.csv"
RANDOM_STATE = 42


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Download creditcard.csv from Kaggle "
            "(mlg-ulb/creditcardfraud) and place it in the data/ folder."
        )
    return pd.read_csv(path)

def describe_imbalance(df: pd.DataFrame) -> None:
    counts = df["Class"].value_counts()
    fraud_rate = counts.get(1, 0) / len(df)

    print("\n=== Class balance ===")
    print(f"Legitimate  : {counts.get(0, 0):,}")
    print(f"Fraud       : {counts.get(1, 0):,}")
    print(f"Fraud rate  : {fraud_rate:.4%}")
    print(
        f"Accuracy of a model that always predicts 'legitimate': "
        f"{1 - fraud_rate:.4%} (and it catches zero fraud)"
    )

def preprocess(df: pd.DataFrame):
    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    cols = ["Amount", "Time"]
    X_train = X_train.copy()
    X_test = X_test.copy()
    scaler = RobustScaler()
    X_train[cols] = scaler.fit_transform(X_train[cols])
    X_test[cols] = scaler.transform(X_test[cols])

    return X_train, X_test, y_train, y_test

def main() -> None:
    df = load_data(DATA_PATH)
    print(f"Loaded {len(df):,} rows and {df.shape[1]} columns")
    describe_imbalance(df)
    X_train, X_test, y_train, y_test = preprocess(df)
    print("\n=== Train/test split ===")
    print(f"Training rows   : {len(X_train):,} (fraud: {y_train.sum()})") # every legit row = 0, so .sum() adds all the 1's (frauds), and the legits don't get any focus
    print(f"Test rows       : {len(X_test):,} (fraud: {y_test.sum()})")


if __name__ == "__main__":
    main()
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

DATA_PATH = Path("data") / "creditcard.csv"


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

def main() -> None:
    df = load_data(DATA_PATH)
    print(f"Loaded {len(df):,} rows and {df.shape[1]} columns")
    describe_imbalance(df)


if __name__ == "__main__":
    main()
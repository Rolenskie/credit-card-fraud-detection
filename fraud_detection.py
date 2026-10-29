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


def main() -> None:
    df = load_data(DATA_PATH)
    print(f"Loaded {len(df):,} rows and {df.shape[1]} columns")


if __name__ == "__main__":
    main()
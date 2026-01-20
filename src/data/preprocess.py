import pandas as pd
from typing import Tuple

RAW_PATH = "data/raw/raw_data.csv"
PROCESSED_PATH = "data/processed/cleaned_data.csv"


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply full preprocessing pipeline.
    """

    df = df.copy()

    df.columns = df.columns.str.strip()

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Drop identifier
    df.drop(columns=["customerID"], inplace=True)

    # Convert churn to binary
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Handle missing values
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Convert categorical Yes/No columns to consistent format
    yes_no_cols = [
        "Partner", "Dependents", "PhoneService", "PaperlessBilling"
    ]

    for col in yes_no_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0})

    return df


def run_preprocessing(
    raw_path: str = RAW_PATH,
    processed_path: str = PROCESSED_PATH
) -> None:
    """
    Preprocessing the data.
    """
    df = pd.read_csv(raw_path)
    df_clean = preprocess_data(df)
    df_clean.to_csv(processed_path, index=False)


if __name__ == "__main__":
    run_preprocessing()
    print("Cleaned data saved.")

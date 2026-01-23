import pandas as pd
from sklearn.model_selection import train_test_split

PROCESSED_PATH = "data/processed/cleaned_data.csv"
FEATURE_PATH = "data/processed/features.csv"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    y = df["Churn"]
    X = df.drop(columns=["Churn"])

    # One-hot encode categorical features
    X_encoded = pd.get_dummies(X, drop_first=True)

    X_encoded["Churn"] = y.values

    return X_encoded


def run_feature_pipeline():
    df = pd.read_csv(PROCESSED_PATH)
    df_features = build_features(df)
    df_features.to_csv(FEATURE_PATH, index=False)
    print("Feature engineering complete.")


if __name__ == "__main__":
    run_feature_pipeline()

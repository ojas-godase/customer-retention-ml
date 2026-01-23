import os
import pandas as pd
import mlflow
import matplotlib.pyplot as plt

from sklearn.inspection import PartialDependenceDisplay

FEATURE_PATH = "data/processed/features.csv"
XGB_RUN_ID = "8ce72d9bab764bfbb7f59497215f4a39"

mlflow.set_tracking_uri("file:./mlruns")

OUTPUT_DIR = "reports"
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 12,
    "figure.dpi": 300
})

# LOAD DATA
df = pd.read_csv(FEATURE_PATH)
X = df.drop(columns=["Churn"])

# LOAD MODEL
model_uri = f"runs:/{XGB_RUN_ID}/model"
model = mlflow.sklearn.load_model(model_uri)

print(f"Loaded model from run: {XGB_RUN_ID}")

# PDP CONFIGURATION
pdp_features = [
    ("tenure", "Partial Dependence of Churn on Tenure"),
    ("MonthlyCharges", "Partial Dependence of Churn on Monthly Charges"),
    ("Contract_Two year", "Partial Dependence of Churn on Two-Year Contract"),
    ("InternetService_Fiber optic", "Partial Dependence of Churn on Fiber Optic Internet"),
    ("PaymentMethod_Electronic check", "Partial Dependence of Churn on Electronic Check Payment"),
]

# GENERATE PDPs
for feature, title in pdp_features:
    fig, ax = plt.subplots(figsize=(8, 6))

    PartialDependenceDisplay.from_estimator(
        model,
        X,
        features=[feature],
        grid_resolution=50,
        ax=ax
    )

    ax.set_title(title)
    ax.set_ylabel("Predicted Churn Probability")

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, f"pdp_{feature.replace(' ', '_')}.png"),
        bbox_inches="tight"
    )
    plt.close()

print("PDP analysis completed.")
print("Plots saved to reports/")
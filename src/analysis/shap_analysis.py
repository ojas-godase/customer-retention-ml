import os
import pandas as pd
import mlflow
import shap
import matplotlib.pyplot as plt


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

# LOAD MODEL FROM MLFLOW
model_uri = f"runs:/{XGB_RUN_ID}/model"
model = mlflow.sklearn.load_model(model_uri)

print(f"Loaded XGBoost model from MLflow run: {XGB_RUN_ID}")

# SHAP 
explainer = shap.Explainer(model)
shap_values = explainer(X)

# SHAP SUMMARY 
plt.figure(figsize=(14, 10))
shap.summary_plot(
    shap_values,
    X,
    show=False
)
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "shap_summary.png"),
    bbox_inches="tight"
)
plt.close()

# SHAP FEATURE IMPORTANCE 
plt.figure(figsize=(10, 10))
shap.summary_plot(
    shap_values,
    X,
    plot_type="bar",
    show=False
)
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "shap_feature_importance.png"),
    bbox_inches="tight"
)
plt.close()

print("SHAP analysis completed successfully.")
print(f"Plots saved to: {OUTPUT_DIR}/")
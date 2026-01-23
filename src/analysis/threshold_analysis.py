import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import precision_recall_curve, roc_auc_score
import mlflow

# Load data
df = pd.read_csv("data/processed/features.csv")
X = df.drop(columns=["Churn"])
y = df["Churn"].values

# Load models from MLflow
mlflow.set_tracking_uri("file:./mlruns")

LOGREG_RUN_ID = "9d462f0b6efa4c809adef464c8879b23"
XGB_RUN_ID = "8ce72d9bab764bfbb7f59497215f4a39"

logreg_model = mlflow.sklearn.load_model(f"runs:/{LOGREG_RUN_ID}/model")
xgb_model = mlflow.sklearn.load_model(f"runs:/{XGB_RUN_ID}/model")

# Probabilities
logreg_probs = logreg_model.predict_proba(X)[:, 1]
xgb_probs = xgb_model.predict_proba(X)[:, 1]

# Precision–Recall curves
logreg_p, logreg_r, logreg_t = precision_recall_curve(y, logreg_probs)
xgb_p, xgb_r, xgb_t = precision_recall_curve(y, xgb_probs)

print("Logistic Regression ROC AUC:", roc_auc_score(y, logreg_probs))
print("XGBoost ROC AUC:", roc_auc_score(y, xgb_probs))


def evaluate_at_threshold(probs, y, threshold):
    preds = (probs >= threshold).astype(int)

    tp = ((preds == 1) & (y == 1)).sum()
    fp = ((preds == 1) & (y == 0)).sum()
    fn = ((preds == 0) & (y == 1)).sum()

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)

    return precision, recall


thresholds = [0.2, 0.3, 0.4, 0.5]

print("\n=== Logistic Regression ===")
for t in thresholds:
    p, r = evaluate_at_threshold(logreg_probs, y, t)
    print(f"Threshold {t:.1f} → Precision={p:.3f}, Recall={r:.3f}")

print("\n=== XGBoost ===")
for t in thresholds:
    p, r = evaluate_at_threshold(xgb_probs, y, t)
    print(f"Threshold {t:.1f} → Precision={p:.3f}, Recall={r:.3f}")

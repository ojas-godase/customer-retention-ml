import numpy as np
import pandas as pd
import mlflow
from sklearn.metrics import confusion_matrix


CONTACT_COST = 100     
CHURN_LOSS = 2000      

LOGREG_RUN_ID = "9d462f0b6efa4c809adef464c8879b23"
XGB_RUN_ID = "8ce72d9bab764bfbb7f59497215f4a39"

# LOAD DATA
df = pd.read_csv("data/processed/features.csv")
X = df.drop(columns=["Churn"])
y = df["Churn"].values

mlflow.set_tracking_uri("file:./mlruns")

logreg = mlflow.sklearn.load_model(f"runs:/{LOGREG_RUN_ID}/model")
xgb = mlflow.sklearn.load_model(f"runs:/{XGB_RUN_ID}/model")

logreg_probs = logreg.predict_proba(X)[:, 1]
xgb_probs = xgb.predict_proba(X)[:, 1]


def profit_at_threshold(probs, y, threshold):
    preds = (probs >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y, preds).ravel()

    contacted = tp + fp
    profit = (tp * CHURN_LOSS) - (contacted * CONTACT_COST)

    return profit, tp, contacted


thresholds = np.linspace(0.05, 0.6, 30)

print("\n=== Logistic Regression Profit ===")
for t in thresholds:
    profit, tp, contacted = profit_at_threshold(logreg_probs, y, t)
    print(f"t={t:.2f} | profit=₹{profit:,} | churn_saved={tp} | contacted={contacted}")

print("\n=== XGBoost Profit ===")
for t in thresholds:
    profit, tp, contacted = profit_at_threshold(xgb_probs, y, t)
    print(f"t={t:.2f} | profit=₹{profit:,} | churn_saved={tp} | contacted={contacted}")

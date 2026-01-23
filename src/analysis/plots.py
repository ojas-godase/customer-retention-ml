import numpy as np
import pandas as pd
import mlflow
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score

LOGREG_RUN_ID = "9d462f0b6efa4c809adef464c8879b23"
XGB_RUN_ID = "8ce72d9bab764bfbb7f59497215f4a39"

FEATURE_PATH = "data/processed/features.csv"

mlflow.set_tracking_uri("file:./mlruns")

# LOAD DATA
df = pd.read_csv(FEATURE_PATH)
X = df.drop(columns=["Churn"])
y = df["Churn"].values

logreg = mlflow.sklearn.load_model(f"runs:/{LOGREG_RUN_ID}/model")
xgb = mlflow.sklearn.load_model(f"runs:/{XGB_RUN_ID}/model")

logreg_probs = logreg.predict_proba(X)[:, 1]
xgb_probs = xgb.predict_proba(X)[:, 1]

thresholds = np.linspace(0.05, 0.6, 30)

logreg_precision, logreg_recall = [], []
xgb_precision, xgb_recall = [], []

for t in thresholds:
    logreg_preds = (logreg_probs >= t).astype(int)
    xgb_preds = (xgb_probs >= t).astype(int)

    logreg_precision.append(precision_score(y, logreg_preds))
    logreg_recall.append(recall_score(y, logreg_preds))

    xgb_precision.append(precision_score(y, xgb_preds))
    xgb_recall.append(recall_score(y, xgb_preds))

# PLOT
plt.figure(figsize=(10, 6))

plt.plot(thresholds, logreg_recall, label="LogReg Recall")
plt.plot(thresholds, logreg_precision, label="LogReg Precision")

plt.plot(thresholds, xgb_recall, label="XGBoost Recall")
plt.plot(thresholds, xgb_precision, label="XGBoost Precision")

plt.xlabel("Decision Threshold")
plt.ylabel("Score")
plt.title("Precision and Recall vs Threshold")
plt.legend()
plt.grid(True)

plt.show()




# PROFIT VS THRESHOLD
CONTACT_COST = 100
CHURN_LOSS = 2000

def compute_profit(probs, y, threshold):
    preds = (probs >= threshold).astype(int)

    tp = ((preds == 1) & (y == 1)).sum()
    fp = ((preds == 1) & (y == 0)).sum()

    contacted = tp + fp
    profit = (tp * CHURN_LOSS) - (contacted * CONTACT_COST)

    return profit

logreg_profit = []
xgb_profit = []

for t in thresholds:
    logreg_profit.append(compute_profit(logreg_probs, y, t))
    xgb_profit.append(compute_profit(xgb_probs, y, t))

plt.figure(figsize=(10, 6))

plt.plot(thresholds, logreg_profit, label="Logistic Regression")
plt.plot(thresholds, xgb_profit, label="XGBoost")

plt.xlabel("Decision Threshold")
plt.ylabel("Profit (₹)")
plt.title("Profit vs Threshold")
plt.legend()
plt.grid(True)

plt.show()



# PROFIT + CONTACTED CUSTOMERS (DUAL AXIS)

def compute_profit_and_contacts(probs, y, threshold):
    preds = (probs >= threshold).astype(int)

    tp = ((preds == 1) & (y == 1)).sum()
    fp = ((preds == 1) & (y == 0)).sum()

    contacted = tp + fp
    money_saved = tp * CHURN_LOSS
    money_spent = contacted * CONTACT_COST
    profit = money_saved - money_spent

    return profit, contacted, money_spent


logreg_profit, logreg_contacts = [], []
xgb_profit, xgb_contacts = [], []

for t in thresholds:
    p, c, _ = compute_profit_and_contacts(logreg_probs, y, t)
    logreg_profit.append(p)
    logreg_contacts.append(c)

    p, c, _ = compute_profit_and_contacts(xgb_probs, y, t)
    xgb_profit.append(p)
    xgb_contacts.append(c)

fig, ax1 = plt.subplots(figsize=(10, 6))

# Profit curves
ax1.plot(thresholds, logreg_profit, label="LogReg Profit", color="blue")
ax1.plot(thresholds, xgb_profit, label="XGBoost Profit", color="orange")
ax1.set_xlabel("Decision Threshold")
ax1.set_ylabel("Profit (₹)")
ax1.grid(True)

# Second axis: customers contacted
ax2 = ax1.twinx()
ax2.plot(thresholds, logreg_contacts, linestyle="--", color="blue", alpha=0.5, label="LogReg Contacts")
ax2.plot(thresholds, xgb_contacts, linestyle="--", color="orange", alpha=0.5, label="XGBoost Contacts")
ax2.set_ylabel("Customers Contacted")

# Legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

plt.title("Profit and Customers Contacted vs Threshold")
plt.show()

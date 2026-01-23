import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

FEATURE_PATH = "data/processed/features.csv"

st.set_page_config(
    page_title="Customer Churn Decision Dashboard",
    layout="wide"
)

st.title("Customer Churn Prediction & Decision Dashboard")

# LOAD DATA
@st.cache_data
def load_data():
    df = pd.read_csv(FEATURE_PATH)
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    return df, X, y

df, X, y = load_data()

# LOAD MODELS
@st.cache_resource
def load_models():
    xgb = joblib.load("models/xgboost.pkl")
    logreg = joblib.load("models/logreg.pkl")
    return xgb, logreg

xgb_model, logreg_model = load_models()

st.sidebar.header("Model & Decision Settings")

model_choice = st.sidebar.selectbox(
    "Select model",
    ["XGBoost", "Logistic Regression"]
)

attention_threshold = st.sidebar.slider(
    "Minimum churn probability to consider action",
    0.05, 0.90, 0.20, 0.01
)

decision_threshold = st.sidebar.slider(
    "Decision threshold (policy)",
    0.05, 0.90, 0.20, 0.01
)

save_rate = st.sidebar.slider(
    "Retention success rate after contact",
    0.05, 1.00, 0.35, 0.05
)

contact_cost = st.sidebar.number_input(
    "Cost to contact customer (₹)",
    min_value=100,
    value=800,
    step=100
)

customer_value = st.sidebar.number_input(
    "Value of customer if retained (₹)",
    min_value=1000,
    value=2000,
    step=500
)


mode = st.sidebar.radio(
    "Prediction mode",
    ["Existing customer", "New customer (manual input)"]
)

model = xgb_model if model_choice == "XGBoost" else logreg_model

FRIENDLY_LABELS = {
    "SeniorCitizen": "Senior Citizen",
    "Partner": "Has Partner",
    "Dependents": "Has Dependents",
    "PhoneService": "Phone Service",
    "PaperlessBilling": "Paperless Billing"
}

# CUSTOMER INPUT
if mode == "Existing customer":
    idx = st.sidebar.number_input(
        "Customer index", 0, len(X) - 1, 0
    )
    customer = X.iloc[[idx]]

else:
    st.sidebar.subheader("Customer details")
    manual = {}

    for col in X.columns:
        label = FRIENDLY_LABELS.get(col, col)

        if X[col].nunique() == 2:
            choice = st.sidebar.selectbox(label, ["No", "Yes"])
            manual[col] = 1 if choice == "Yes" else 0
        else:
            manual[col] = st.sidebar.number_input(
                label,
                float(X[col].min()),
                float(X[col].max()),
                float(X[col].median())
            )

    customer = pd.DataFrame([manual])[X.columns]

prob = model.predict_proba(customer)[0][1]

risk = (
    "Low" if prob < 0.3 else
    "Medium" if prob < 0.6 else
    "High"
)

# TOP METRICS
col1, col2, col3 = st.columns(3)
col1.metric("Churn Probability", f"{prob:.2%}")
col2.metric("Risk Level", risk)
col3.metric(
    "Decision",
    "CONTACT" if prob >= decision_threshold else "DO NOT CONTACT"
)

st.divider()

# DECISION & PROFIT LOGIC
expected_saved_value = prob * save_rate * customer_value
expected_profit = expected_saved_value - contact_cost
expected_loss_no_action = prob * customer_value

if prob < attention_threshold:
    st.info(
        f"""
        **No action taken**

        Churn probability ({prob:.2%}) is below the attention threshold
        ({attention_threshold:.2%}).
        """
    )
else:
    st.subheader("Business Impact")

    if prob >= decision_threshold:
        st.success(
            f"""
            **Customer will be contacted**

            • Cost to contact: ₹{contact_cost:,.0f}  
            • Retention success rate: {save_rate:.0%}  
            • Expected value saved: ₹{expected_saved_value:,.0f}  

            **Expected profit from intervention: ₹{expected_profit:,.0f}**
            """
        )
    else:
        st.error(
            f"""
            **Customer will NOT be contacted**

            • Expected loss due to churn: ₹{expected_loss_no_action:,.0f}  
            • Missed retention opportunity: ₹{expected_saved_value:,.0f}
            """
        )

    st.divider()

    # MODEL EXPLANATION
    st.subheader("Why this prediction?")

    if model_choice == "XGBoost":
        explainer = shap.Explainer(model)
        shap_values = explainer(customer)

        fig, ax = plt.subplots(figsize=(8, 4))
        shap.plots.bar(shap_values[0], max_display=10, show=False)
        st.pyplot(fig)

    else:
        lr = model.named_steps["model"]
        coefs = lr.coef_[0]

        coef_df = (
            pd.DataFrame({
                "Feature": customer.columns,
                "Coefficient": coefs
            })
            .assign(abs_coef=lambda d: d["Coefficient"].abs())
            .sort_values("abs_coef", ascending=False)
            .head(10)
        )

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(coef_df["Feature"], coef_df["Coefficient"])
        ax.invert_yaxis()
        st.pyplot(fig)

with st.expander("View customer feature values"):
    st.dataframe(customer.T, use_container_width=True)

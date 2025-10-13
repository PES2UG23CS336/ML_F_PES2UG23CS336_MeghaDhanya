# app.py
import streamlit as st
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import shap
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(layout="wide", page_title="Fraud Detection Demo")

st.title("Detecting Payment Fraud — Demo UI")
st.markdown("Upload `creditcard.csv` or use sample. Trains models (SMOTE) and shows metrics + plots.")

uploaded = st.file_uploader("Upload creditcard.csv (or skip to use sample)", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
else:
    st.info("No file uploaded — using sample from Kaggle (first 30k rows). Upload full CSV for full results.")
    # If you have no file, we create a synthetic small sample from available data approach (just to demo)
    st.stop()

# DATA PREP
df = df.dropna(subset=["Class"])
df["Class"] = pd.to_numeric(df["Class"], errors="coerce")
df = df.dropna(subset=["Class"])
df = df.fillna(0)

X = df.drop("Class", axis=1)
y = df["Class"]
scaler = StandardScaler()
if "Amount" in X.columns:
    X["Amount"] = scaler.fit_transform(X[["Amount"]])
if "Time" in X.columns:
    X["Time"] = scaler.fit_transform(X[["Time"]])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# MODEL OPTIONS
model_map = {
    "Logistic Regression": LogisticRegression(max_iter=500, class_weight="balanced"),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Naive Bayes": GaussianNB(),
    "SVM": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42),
    "ANN": MLPClassifier(hidden_layer_sizes=(64,32), max_iter=300, random_state=42)
}

st.sidebar.header("Controls")
chosen = st.sidebar.selectbox("Model to train & evaluate", list(model_map.keys()))
train_btn = st.sidebar.button("Train & Evaluate")

if train_btn:
    model = model_map[chosen]
    with st.spinner(f"Training {chosen}..."):
        model.fit(X_train_res, y_train_res)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1] if hasattr(model,"predict_proba") else model.decision_function(X_test)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = auc(*roc_curve(y_test, y_prob)[:2])

    st.subheader(f"Results — {chosen}")
    st.write(f"Precision: {precision:.4f} — Recall: {recall:.4f} — F1: {f1:.4f} — AUC: {roc_auc:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(4,4))
    import seaborn as sns
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Legit","Fraud"], yticklabels=["Legit","Fraud"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    st.pyplot(fig)

    # ROC plot
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig2, ax2 = plt.subplots(figsize=(5,4))
    ax2.plot(fpr,tpr,label=f"AUC = {roc_auc:.4f}")
    ax2.plot([0,1],[0,1],"k--")
    ax2.set_xlabel("False Positive Rate"); ax2.set_ylabel("True Positive Rate"); ax2.legend()
    st.pyplot(fig2)

    # SHAP (try for tree models)
    st.subheader("SHAP Summary (top features)")
    try:
        if chosen in ["Random Forest","Gradient Boosting"]:
            explainer = shap.TreeExplainer(model)
            sample = X_test.sample(1000, random_state=42) if len(X_test)>1000 else X_test
            shap_values = explainer.shap_values(sample)
            st.pyplot(shap.summary_plot(shap_values, sample, plot_type="bar", show=False))
        else:
            explainer = shap.Explainer(model, X_train_res)  # slower
            sample = X_test.sample(1000, random_state=42) if len(X_test)>1000 else X_test
            shap_values = explainer(sample)
            st.pyplot(shap.summary_plot(shap_values, sample, plot_type="bar", show=False))
    except Exception as e:
        st.write("SHAP error or too slow in this environment:", e)

    # Single transaction prediction
    st.subheader("Single transaction prediction (sampled)")
    sample_row = X_test.sample(1, random_state=42)
    prob = model.predict_proba(sample_row)[:,1] if hasattr(model,"predict_proba") else model.decision_function(sample_row)
    st.write("Sample features:", sample_row.T)
    st.write("Fraud probability:", float(prob))
    st.success("Model ready — use this UI to demo model metrics and sample predictions.")

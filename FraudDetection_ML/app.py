import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Payment Fraud Detection", layout="wide")
st.title("Detecting Payment Fraud Using Machine Learning")
st.markdown("By Megha Dhanya (PES2UG23CS336 - Section F)")
st.info("Upload the creditcard.csv dataset and choose a model to train and evaluate.")

uploaded = st.file_uploader("Upload Dataset (creditcard.csv)", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded)
    df = df.dropna(subset=["Class"])
    df["Class"] = pd.to_numeric(df["Class"], errors="coerce")
    df = df.dropna(subset=["Class"])
    df = df.fillna(0)
    st.write("Dataset shape:", df.shape)
    st.write("Class distribution:")
    st.bar_chart(df["Class"].value_counts())
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
    st.write("After SMOTE balancing:", np.bincount(y_train_res))
    models = {
        "Logistic Regression": LogisticRegression(max_iter=500, class_weight='balanced'),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Naive Bayes": GaussianNB(),
        "SVM": SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42),
        "ANN": MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=300, random_state=42)
    }
    st.sidebar.header("Controls")
    chosen_model = st.sidebar.selectbox("Select a model to train:", list(models.keys()))
    run_button = st.sidebar.button("Train and Evaluate")
    if run_button:
        model = models[chosen_model]
        st.write(f"Training {chosen_model}...")
        model.fit(X_train_res, y_train_res)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = auc(*roc_curve(y_test, y_prob)[:2])
        st.write(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f} | AUC: {roc_auc:.4f}")
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig2, ax2 = plt.subplots()
        ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
        ax2.plot([0, 1], [0, 1], 'k--')
        ax2.set_xlabel("False Positive Rate")
        ax2.set_ylabel("True Positive Rate")
        ax2.legend(loc="lower right")
        ax2.set_title(f"ROC Curve - {chosen_model}")
        st.pyplot(fig2)
        st.subheader("SHAP Feature Importance (if supported)")
        try:
            if chosen_model in ["Random Forest", "Gradient Boosting", "Decision Tree", "Logistic Regression"]:
                explainer = shap.Explainer(model, X_train_res)
                sample = X_test.sample(min(500, len(X_test)), random_state=42)
                shap_values = explainer(sample)
                st.pyplot(shap.summary_plot(shap_values, sample, plot_type="bar", show=False))
            else:
                st.info(f"SHAP not supported for {chosen_model}.")
        except Exception as e:
            st.warning(f"Could not compute SHAP for {chosen_model}: {e}")
        st.subheader("Predict a Random Transaction")
        sample_txn = X_test.sample(1, random_state=42)
        prob = model.predict_proba(sample_txn)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(sample_txn)
        st.write(sample_txn.T)
        st.write(f"Predicted Fraud Probability: {float(prob):.4f}")
else:
    st.warning("Please upload the dataset to start.")

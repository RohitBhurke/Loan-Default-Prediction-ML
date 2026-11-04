
import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score,
    f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

st.set_page_config(
    page_title="Loan Prediction System",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Loan Prediction & Model Comparison")
st.write("Evaluate five machine learning models using loan application test data.")

# Model selection
model_files = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "SVM": "svm.pkl"
}

selected_model = st.selectbox(
    "Select Machine Learning Model",
    list(model_files.keys())
)

# CSV upload
uploaded_file = st.file_uploader(
    "Upload labelled test data (CSV)",
    type=["csv"]
)

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    if "loan_status" not in data.columns:
        st.error("CSV must contain the loan_status target column.")
        st.stop()

    X = data.drop("loan_status", axis=1)
    y = data["loan_status"]

    model_path = os.path.join(
        "model", model_files[selected_model]
    )

    model = joblib.load(model_path)

    y_pred = model.predict(X)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X)[:, 1]
    else:
        y_score = model.decision_function(X)

    # Evaluation metrics
    st.subheader("Model Evaluation Metrics")

    metrics = {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC Score": roc_auc_score(y, y_score),
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1 Score": f1_score(y, y_pred, zero_division=0),
        "MCC Score": matthews_corrcoef(y, y_pred)
    }

    cols = st.columns(3)

    for i, (name, value) in enumerate(metrics.items()):
        cols[i % 3].metric(name, f"{value:.4f}")

    # Confusion matrix
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Default (0)", "Default (1)"],
        yticklabels=["No Default (0)", "Default (1)"],
        ax=ax
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)
    plt.close(fig)

    # Classification report
    st.subheader("Classification Report")
    st.text(classification_report(y, y_pred, zero_division=0))

    st.success("Model evaluation completed successfully!")

else:
    st.info("Upload your labelled test_data.csv file to evaluate the selected model.")

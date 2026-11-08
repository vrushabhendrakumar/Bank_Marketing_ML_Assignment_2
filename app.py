"""Streamlit dashboard for comparing trained Bank Marketing classifiers."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef, confusion_matrix,
    classification_report, ConfusionMatrixDisplay)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model" / "saved_models"
DEFAULT_TEST = ROOT / "test_data.csv"
TARGET = "y"

st.set_page_config(page_title="BankReach | ML Model Lab", page_icon="🏦",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.block-container {padding-top: 1.7rem; padding-bottom: 2.5rem;}
.hero {padding: 1.25rem 1.5rem; border-radius: 16px;
       background: linear-gradient(120deg,#102a43,#176b87); color: white;}
.hero h1 {color:white; margin:0;}
.hero p {color:#e2f1f7; margin:.35rem 0 0 0;}
div[data-testid="stMetric"] {background:#f5f8fb; padding:14px; border-radius:12px;}
</style>
""", unsafe_allow_html=True)

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "k-nearest_neighbors.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib"
}
# Match the actual saved filenames produced by train_models.py.
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "k-nearest_neighbors.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib"
}

@st.cache_resource
def load_models():
    loaded = {}
    for name, filename in MODEL_FILES.items():
        path = MODEL_DIR / filename
        if path.exists():
            loaded[name] = joblib.load(path)
    return loaded

def score_positive(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)

st.markdown("""
<div class="hero">
  <h1>🏦 BankReach | Classification Model Lab</h1>
  <p>Compare five machine-learning classifiers for term-deposit subscription prediction.</p>
</div>
""", unsafe_allow_html=True)
st.write("")
st.caption("BITS Pilani WILP · M.Tech (DSE) · Machine Learning · Assignment 2")

models = load_models()
if not models:
    st.error("Trained model files were not found. First run `python data/download_dataset.py` and `python model/train_models.py`, then commit the generated model files.")
    st.stop()

with st.sidebar:
    st.header("Experiment controls")
    selected_model = st.selectbox("Choose classifier", list(models.keys()))
    st.markdown("---")
    st.markdown("**Test data**")
    uploaded = st.file_uploader("Upload labelled test CSV", type=["csv"],
                                help="Include all predictor columns and the target column y (yes/no) to calculate metrics.")
    st.caption("The uploaded CSV is used only for evaluation. Do not upload the full training dataset.")

if uploaded is not None:
    test_df = pd.read_csv(uploaded)
elif DEFAULT_TEST.exists() and DEFAULT_TEST.stat().st_size > 20:
    test_df = pd.read_csv(DEFAULT_TEST)
    st.info("Showing the saved held-out test set. Upload another labelled CSV from the sidebar to evaluate a different sample.")
else:
    st.warning("No test CSV is available yet. Upload a labelled test CSV, or run the training script to generate `test_data.csv`.")
    st.stop()

if TARGET not in test_df.columns:
    st.error("The CSV must include a target column named `y` with values `yes` or `no` for evaluation.")
    st.stop()

y_raw = test_df[TARGET].astype(str).str.lower().str.strip()
valid = y_raw.isin(["yes", "no"])
if not valid.all():
    st.error("Target column `y` must contain only `yes` and `no` values.")
    st.stop()
y_true = y_raw.map({"no": 0, "yes": 1}).to_numpy()
X_test = test_df.drop(columns=[TARGET])
if "duration" in X_test.columns:
    X_test = X_test.drop(columns=["duration"])
expected = list(models[selected_model].feature_names_in_) if hasattr(models[selected_model], "feature_names_in_") else None
if expected:
    missing = sorted(set(expected) - set(X_test.columns))
    if missing:
        st.error("Missing required predictor columns: " + ", ".join(missing))
        st.stop()
    X_test = X_test[expected]

model = models[selected_model]
try:
    y_pred = model.predict(X_test)
    y_score = score_positive(model, X_test)
except Exception as exc:
    st.exception(exc)
    st.stop()

accuracy = accuracy_score(y_true, y_pred)
auc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) == 2 else float("nan")
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
mcc = matthews_corrcoef(y_true, y_pred)

st.subheader("Evaluation snapshot")
st.caption(f"Model: {selected_model} · Test observations: {len(test_df):,}")
c1, c2, c3 = st.columns(3)
c1.metric("Accuracy", f"{accuracy:.4f}")
c2.metric("ROC-AUC", f"{auc:.4f}" if np.isfinite(auc) else "N/A")
c3.metric("Precision", f"{precision:.4f}")
c4, c5, c6 = st.columns(3)
c4.metric("Recall", f"{recall:.4f}")
c5.metric("F1 score", f"{f1:.4f}")
c6.metric("MCC", f"{mcc:.4f}")

left, right = st.columns([1, 1])
with left:
    st.subheader("Confusion matrix")
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["No", "Yes"], cmap="Blues",
        colorbar=False, ax=ax
    )
    ax.set_title(selected_model)
    st.pyplot(fig, clear_figure=True)
with right:
    st.subheader("Classification report")
    report = classification_report(
        y_true, y_pred, target_names=["No subscription", "Subscription"],
        output_dict=True, zero_division=0
    )
    st.dataframe(pd.DataFrame(report).T.round(4), use_container_width=True)

st.subheader("Prediction sample")
preview = X_test.head(15).copy()
preview["Actual"] = np.where(y_true[:len(preview)] == 1, "Yes", "No")
preview["Predicted"] = np.where(y_pred[:len(preview)] == 1, "Yes", "No")
st.dataframe(preview, use_container_width=True)

st.markdown("---")
st.caption("Academic demonstration only. Predictions are not financial advice. The duration feature is intentionally excluded to avoid post-call information leakage.")

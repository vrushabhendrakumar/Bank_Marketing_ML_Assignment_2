"""Train and evaluate five classifiers on the UCI Bank Marketing dataset."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef, classification_report,
    confusion_matrix)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "bank-full.csv"
MODEL_DIR = ROOT / "model" / "saved_models"
RESULT_DIR = ROOT / "results"
TEST_PATH = ROOT / "test_data.csv"
TARGET = "y"
RANDOM_STATE = 42
TEST_SIZE = 0.20

def build_preprocessor(X: pd.DataFrame):
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    transformer = ColumnTransformer([
        ("num", numeric_pipe, numeric),
        ("cat", categorical_pipe, categorical)
    ])
    return transformer

def get_models():
    # LinearSVC is used instead of kernel SVC to keep training practical
    # on the full 45k-row dataset; decision_function supports ROC-AUC.
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1500, class_weight="balanced", solver="lbfgs",
            random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=12, min_samples_leaf=10, class_weight="balanced",
            random_state=RANDOM_STATE
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=7, weights="distance", n_jobs=-1
        ),
        "Gaussian Naive Bayes": GaussianNB(),
        "Support Vector Machine": LinearSVC(
            class_weight="balanced", random_state=RANDOM_STATE,
            max_iter=10000
        )
    }

def positive_scores(estimator, X):
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    return estimator.decision_function(X)

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Run: python data/download_dataset.py"
        )
    df = pd.read_csv(DATA_PATH)
    if TARGET not in df.columns:
        raise ValueError("Expected target column 'y' in the dataset.")
    df = df.dropna(subset=[TARGET]).copy()
    df[TARGET] = df[TARGET].astype(str).str.lower().str.strip()
    df = df[df[TARGET].isin(["yes", "no"])].copy()

    # Exclude duration: it is only known after the call and causes target leakage.
    excluded = [TARGET]
    if "duration" in df.columns:
        excluded.append("duration")
    X = df.drop(columns=excluded)
    y = df[TARGET].map({"no": 0, "yes": 1}).astype(int)

    if X.shape[1] < 12:
        raise ValueError(f"Only {X.shape[1]} predictors found; assignment requires >=12.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    # Persist the held-out labelled test rows for Streamlit evaluation.
    test_frame = X_test.copy()
    test_frame[TARGET] = y_test.map({0: "no", 1: "yes"}).values
    test_frame.to_csv(TEST_PATH, index=False)

    rows = []
    detailed = {}
    for name, classifier in get_models().items():
        pipeline = Pipeline([
            ("preprocess", build_preprocessor(X_train)),
            ("classifier", classifier)
        ])
        print(f"Training {name} ...")
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        scores = positive_scores(pipeline, X_test)
        metrics = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "AUC": roc_auc_score(y_test, scores),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1": f1_score(y_test, pred, zero_division=0),
            "MCC": matthews_corrcoef(y_test, pred)
        }
        rows.append(metrics)
        detailed[name] = {
            "classification_report": classification_report(
                y_test, pred, target_names=["no", "yes"],
                output_dict=True, zero_division=0
            ),
            "confusion_matrix": confusion_matrix(y_test, pred).tolist()
        }
        joblib.dump(pipeline, MODEL_DIR / (name.lower().replace(" ", "_") + ".joblib"))
        print("  " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items() if k != "Model"))

    results = pd.DataFrame(rows).sort_values("F1", ascending=False)
    results.to_csv(RESULT_DIR / "model_comparison.csv", index=False)
    with open(RESULT_DIR / "evaluation_details.json", "w", encoding="utf-8") as f:
        json.dump(detailed, f, indent=2)
    metadata = {
        "target": TARGET,
        "excluded_features": ["duration"],
        "predictor_count": int(X.shape[1]),
        "rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "positive_class": "yes"
    }
    with open(RESULT_DIR / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print("\nModel comparison:")
    print(results.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nSaved models to {MODEL_DIR}")
    print(f"Saved held-out test CSV to {TEST_PATH}")

if __name__ == "__main__":
    main()

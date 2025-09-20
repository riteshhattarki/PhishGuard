# src/train.py
# The purpose of this file is to train both the Logistic Regression and the LinearSVC model
# The program print metrics, confusion matrix, and decide and save the best model to use
from pathlib import Path
import os, json, joblib, numpy as np
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

# Load processed data --> loading from Enron dataset
X_train, y_train = joblib.load(PROC / "train.pkl")
X_test,  y_test  = joblib.load(PROC / "test.pkl")
vectorizer = joblib.load(PROC / "tfidf_vectorizer.pkl")

def safe_auc(y_true, scores):
    try:
        return float(roc_auc_score(y_true, scores))
    except Exception:
        return None

def evaluate(name, clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Continuous scores for AUC
    scores = None
    if hasattr(clf, "predict_proba"):
        scores = clf.predict_proba(X_test)[:, 1]
    elif hasattr(clf, "decision_function"):
        scores = clf.decision_function(X_test)

    # Calculates metrics such as auc, confusion matrix, precision, recall, and f1 scores
    # Outputs the classification report
    auc = safe_auc(y_test, scores) if scores is not None else None
    cm = confusion_matrix(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
    report = classification_report(y_test, y_pred, digits=3)

    print(f"\n== {name} ==")
    print(cm)
    print(report)
    if auc is not None:
        print(f"AUC: {auc:.3f}")

    return clf, {"name": name, "precision": float(precision), "recall": float(recall), "f1": float(f1), "auc": auc, "cm": cm.tolist()}

# Candidates
# Setting up both Logistic Regression and LinearSVC model
logreg = LogisticRegression(solver="liblinear", max_iter=2000, class_weight="balanced")
linsvc = LinearSVC(class_weight="balanced")

candidates = [("LogisticRegression (TF-IDF)", logreg), ("LinearSVC (TF-IDF)", linsvc)]
best = None
best_key = -1.0
all_metrics = []

for name, model in candidates:
    clf, m = evaluate(name, model)
    all_metrics.append(m)
    key = (m["auc"] if m["auc"] is not None else 0.0) * 1000.0 + m["f1"]  # prefer AUC, then F1
    if key > best_key:
        best_key = key
        best = (name, clf, m)

best_name, best_model, best_metrics = best

# Save artifacts
joblib.dump(best_model, MODELS / "model.joblib")
joblib.dump(vectorizer,  MODELS / "vectorizer.joblib")

meta = {
    "timestamp": datetime.utcnow().isoformat() + "Z",
    "model_name": best_name,
    "metrics": best_metrics,
    "train_shape": list(X_train.shape),
    "test_shape": list(X_test.shape),
    "vocab_size": int(getattr(vectorizer, "max_features", 0) or len(vectorizer.vocabulary_)),
}
with open(MODELS / "model-meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"\nSaved model → {MODELS/'model.joblib'}")
print(f"Saved vectorizer → {MODELS/'vectorizer.joblib'}")
print(f"Saved metadata → {MODELS/'model-meta.json'}")

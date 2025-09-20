# src/eval.py
# Runs the evaluation script
# Basically runs an evalaution (prints metrics --> covered in Module 4)
# Also shows explainability (why the model did what it did)
from pathlib import Path
import os, json, joblib, numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)

ROOT    = Path(__file__).resolve().parents[1]
PROC    = ROOT / "data" / "processed"
MODELS  = ROOT / "models"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

# ---- Load artifacts ---- from Enron dataset
X_train, y_train = joblib.load(PROC / "train.pkl")
X_test,  y_test  = joblib.load(PROC / "test.pkl")
model            = joblib.load(MODELS / "model.joblib")
vectorizer       = joblib.load(MODELS / "vectorizer.joblib")

# ---- Predictions / scores ----
# y_pred: labels at default threshold
y_pred = model.predict(X_test)

# Continuous scores for curves & threshold tuning:
# (LogReg: predict_proba; LinearSVC: decision_function)
scores = None
if hasattr(model, "predict_proba"):
    scores = model.predict_proba(X_test)[:, 1]
elif hasattr(model, "decision_function"):
    scores = model.decision_function(X_test)

# ---- Metrics (text) ----
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, digits=3)
print("== Confusion Matrix ==")
print(cm)
print("\n== Classification Report ==")
print(report)

# Save metrics JSON
metrics = {
    "timestamp": datetime.utcnow().isoformat() + "Z",
    "confusion_matrix": cm.tolist(),
    "report": report,
}
with open(REPORTS / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# ---- Confusion matrix plot ----
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["ham (0)", "spam (1)"])
fig, ax = plt.subplots(figsize=(5, 5))
disp.plot(ax=ax, colorbar=False)
ax.set_title("Confusion Matrix")
plt.tight_layout()
plt.savefig(REPORTS / "confusion_matrix.png", dpi=160)
plt.close(fig)

# ---- ROC & PR curves (if scores available) ----
if scores is not None:
    # ROC
    fpr, tpr, roc_thresh = roc_curve(y_test, scores)
    auc = roc_auc_score(y_test, scores)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr, tpr, lw=2, label=f"ROC AUC = {auc:.3f}")
    ax.plot([0,1],[0,1],"--", lw=1, label="random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(REPORTS / "roc_curve.png", dpi=160)
    plt.close(fig)

    # Precision-Recall
    prec, rec, pr_thresh = precision_recall_curve(y_test, scores)
    ap = average_precision_score(y_test, scores)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(rec, prec, lw=2, label=f"AP = {ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curve")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(REPORTS / "pr_curve.png", dpi=160)
    plt.close(fig)

    # ---- Threshold sweep (optional): pick a threshold for a target FPR or Precision ----
    # Example: choose threshold that achieves at most 1% false positive rate.
    # (Only meaningful for probabilistic models; for margins it still works as a score cutoff.)
    target_fpr = 0.01
    # use ROC thresholds to find closest FPR
    idx = int(np.argmin(np.abs(fpr - target_fpr)))
    chosen = {
        "target_fpr": target_fpr,
        "threshold": float(roc_thresh[idx]),
        "fpr": float(fpr[idx]),
        "tpr": float(tpr[idx]),
        "auc": float(auc),
    }
    with open(REPORTS / "threshold_choice.json", "w") as f:
        json.dump(chosen, f, indent=2)
    print(f"\n[Threshold] aiming FPR≈{target_fpr:.2%} -> "
          f"threshold={chosen['threshold']:.4f}, actual FPR={chosen['fpr']:.3f}, TPR={chosen['tpr']:.3f}")

# ---- Explainability: top spam/ham n-grams (for linear models) ----
def top_terms_from_linear(model, vectorizer, k=25):
    coef = getattr(model, "coef_", None)
    if coef is None:
        return None
    coef = coef.ravel()
    # Build reverse vocab list where index -> term
    vocab = vectorizer.vocabulary_
    rev = [None] * len(vocab)
    for term, idx in vocab.items():
        rev[idx] = term
    terms = np.array(rev, dtype=object)

    # Largest positive coefficients → push toward spam(1)
    top_spam_idx = np.argsort(coef)[-k:][::-1]
    top_ham_idx  = np.argsort(coef)[:k]

    top_spam = [(terms[i], float(coef[i])) for i in top_spam_idx if terms[i]]
    top_ham  = [(terms[i], float(coef[i])) for i in top_ham_idx  if terms[i]]
    return {"spam_terms": top_spam, "ham_terms": top_ham}

explain = top_terms_from_linear(model, vectorizer, k=25)
if explain is not None:
    with open(REPORTS / "top_terms.json", "w") as f:
        json.dump(explain, f, indent=2)
    print("\nSaved top contributing terms to reports/top_terms.json")
else:
    print("\nExplainability: model has no linear coefficients (skip).")

print("\nSaved:")
for p in ["metrics.json", "confusion_matrix.png", "roc_curve.png", "pr_curve.png", "top_terms.json", "threshold_choice.json"]:
    pp = REPORTS / p
    if pp.exists():
        print(" -", pp)

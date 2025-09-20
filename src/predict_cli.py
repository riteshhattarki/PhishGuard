# src/predict.py
# Will load saved model + vectorizer
# Will accept input via JSON file or CLI arguments
# Will return the label (phish or safe) and a confidence rating
# Will return a confidence rating
import argparse, json, sys, re
from pathlib import Path
import numpy as np
import joblib

# Optional HTML cleaner: BeautifulSoup; fallback to regex if not installed
try:
    from bs4 import BeautifulSoup
    def strip_html(text: str) -> str:
        return BeautifulSoup(text or "", "html.parser").get_text(" ")
except Exception:
    def strip_html(text: str) -> str:
        return re.sub(r"<[^>]+>", " ", text or "")

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"

def load_artifacts(prefer_calibrated: bool = True):
    """
    Load (calibrated) model if present, else fallback to model.joblib.
    Always load vectorizer.joblib.
    """
    model_path = None
    if prefer_calibrated:
        p = MODELS_DIR / "model_calibrated.joblib"
        if p.exists():
            model_path = p
    if model_path is None:
        p = MODELS_DIR / "model.joblib"
        if not p.exists():
            raise FileNotFoundError("No model found at models/model_calibrated.joblib or models/model.joblib")
        model_path = p
    vec_path = MODELS_DIR / "vectorizer.joblib"
    if not vec_path.exists():
        raise FileNotFoundError("Vectorizer missing at models/vectorizer.joblib")

    model = joblib.load(model_path)
    vec   = joblib.load(vec_path)
    return model_path, model, vec

def clean_text(subject: str, message: str, max_len: int = 20000) -> str:
    subject = subject or ""
    message = message or ""
    text = (subject + " " + message).strip()
    text = strip_html(text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_len:
        text = text[:max_len]
    return text

def predict_with_confidence(model, X):
    """
    Returns (label_int, confidence_float, score_float, confidence_type)
    - If model has predict_proba -> probability of spam (class 1)
    - Else if decision_function -> margin; convert to [0,1] via logistic for a pseudo-prob
    """
    label = int(model.predict(X)[0])
    confidence_type = "probability"
    score = None
    conf = None

    if hasattr(model, "predict_proba"):
        conf = float(model.predict_proba(X)[0, 1])
        score = conf
        confidence_type = "probability"
    elif hasattr(model, "decision_function"):
        margin = float(model.decision_function(X)[0])
        # map margin to [0,1] with a logistic for a readable confidence
        conf = float(1.0 / (1.0 + np.exp(-margin)))
        score = margin
        confidence_type = "margin_logistic"
    else:
        # last resort: no score available
        conf = None
        score = None
        confidence_type = "none"

    return label, conf, score, confidence_type

def top_contributing_terms_linear(model, vec, Xrow, k=5):
    """
    For linear models (LogReg / LinearSVC), show the top phrases that pushed the decision.
    Returns dict {spam_terms: [(term, contrib), ...], ham_terms: [...]}
    """
    coef = getattr(model, "coef_", None)
    if coef is None:
        return None
    coef = coef.ravel()

    # Build reverse vocabulary (index -> term) once and cache on the vectorizer
    if not hasattr(vec, "_terms_"):
        rev = [None] * len(vec.vocabulary_)
        for term, idx in vec.vocabulary_.items():
            rev[idx] = term
        vec._terms_ = np.array(rev, dtype=object)

    row = Xrow.tocoo()
    contribs = []
    for j, v in zip(row.col, row.data):
        contribs.append((vec._terms_[j], float(v * coef[j])))

    # Sort by contribution
    contribs.sort(key=lambda x: x[1], reverse=True)
    top_spam = [(t, c) for t, c in contribs[:k] if t]
    top_ham  = [(t, c) for t, c in contribs[-k:] if t]
    return {"spam_terms": top_spam, "ham_terms": top_ham}

def main():
    ap = argparse.ArgumentParser(description="PhishGuard predictor")
    ap.add_argument("--input", "-i", type=str, help="Path to JSON file with {'subject':..., 'message':...}")
    ap.add_argument("--subject", type=str, default="", help="Subject text (alternative to --input)")
    ap.add_argument("--message", type=str, default="", help="Message/body text (alternative to --input)")
    ap.add_argument("--threshold", type=float, default=0.5, help="Spam threshold in [0,1] for probability or logistic-mapped margin.")
    ap.add_argument("--topk", type=int, default=5, help="Number of top contributing phrases to display for each class.")
    ap.add_argument("--no_calibrated", action="store_true", help="Do not prefer calibrated model even if present.")
    args = ap.parse_args()

    # Load model + vectorizer
    model_path, model, vec = load_artifacts(prefer_calibrated=not args.no_calibrated)

    # Read input
    if args.input:
        with open(args.input, "r") as f:
            payload = json.load(f)
        subject = payload.get("subject", "") or ""
        message = payload.get("message", "") or ""
    else:
        subject = args.subject or ""
        message = args.message or ""

    text = clean_text(subject, message)
    if not text:
        print(json.dumps({"error": "Empty subject/message"}))
        sys.exit(1)

    X = vec.transform([text])

    # Predict
    label, conf, score, conf_type = predict_with_confidence(model, X)

    # Apply threshold only if we have a confidence in [0,1]
    decision = label
    used_threshold = None
    if conf is not None:
        decision = int(conf >= args.threshold)
        used_threshold = args.threshold

    # Explain (linear only)
    explain = top_contributing_terms_linear(model, vec, X, k=args.topk)

    result = {
        "model_path": str(model_path),
        "label": "spam" if decision == 1 else "ham",
        "raw_label": int(label),
        "confidence": conf,              # probability if available, else logistic(margin), else null
        "confidence_type": conf_type,    # "probability", "margin_logistic", or "none"
        "threshold_used": used_threshold,
        "score": score,                  # raw decision score or probability (see confidence_type)
        "top_terms": explain,            # may be None if model is not linear
    }
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()

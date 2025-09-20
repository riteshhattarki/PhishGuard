# src/api.py
# This is the FastAPI integration that allows the API to talk to the backend
# Allows the API access to the model to retreive accurate results
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
import joblib, numpy as np, re

try:
    from bs4 import BeautifulSoup
    def strip_html(t: str) -> str:
        return BeautifulSoup(t or "", "html.parser").get_text(" ")
except Exception:
    def strip_html(t: str) -> str:
        return re.sub(r"<[^>]+>", " ", t or "")

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
model_path = MODELS / "model_calibrated.joblib"
if not model_path.exists():
    model_path = MODELS / "model.joblib"
vec_path = MODELS / "vectorizer.joblib"

model = joblib.load(model_path)
vec   = joblib.load(vec_path)

app = FastAPI(title="PhishGuard API")

class EmailIn(BaseModel):
    subject: str = ""
    message: str = ""
    threshold: float = 0.5
    topk: int = 5

def clean_text(sub: str, msg: str) -> str:
    t = f"{sub or ''} {msg or ''}".strip()
    t = strip_html(t)
    return re.sub(r"\s+", " ", t).strip()[:20000]

@app.post("/predict")
def predict(email: EmailIn):
    text = clean_text(email.subject, email.message)
    X = vec.transform([text])
    raw = int(model.predict(X)[0])

    conf, score, ctype = None, None, "none"
    if hasattr(model, "predict_proba"):
        conf = float(model.predict_proba(X)[0, 1]); score, ctype = conf, "probability"
    elif hasattr(model, "decision_function"):
        margin = float(model.decision_function(X)[0])
        conf = float(1.0 / (1.0 + np.exp(-margin))); score, ctype = margin, "margin_logistic"

    decision = raw if conf is None else int(conf >= email.threshold)

    explain = None
    coef = getattr(model, "coef_", None)
    if coef is not None:
        coef = coef.ravel()
        if not hasattr(vec, "_terms_"):
            rev = [None] * len(vec.vocabulary_)
            for t, i in vec.vocabulary_.items():
                rev[i] = t
            import numpy as np
            vec._terms_ = np.array(rev, dtype=object)
        row = X.tocoo()
        contribs = [(vec._terms_[j], float(v * coef[j])) for j, v in zip(row.col, row.data)]
        contribs.sort(key=lambda x: x[1], reverse=True)
        explain = {
            "spam_terms": [(t, c) for t, c in contribs[:email.topk] if t],
            "ham_terms":  [(t, c) for t, c in contribs[-email.topk:] if t],
        }

    return {
        "model": model_path.name,
        "label": "spam" if decision == 1 else "ham",
        "raw_label": raw,
        "confidence": conf,
        "confidence_type": ctype,
        "threshold_used": (email.threshold if conf is not None else None),
        "score": score,
        "top_terms": explain
    }

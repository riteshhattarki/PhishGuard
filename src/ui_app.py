# src/ui_app.py
# Will set up the Streamlit UI
import numpy as np, re, joblib
from pathlib import Path
import streamlit as st

try:
    from bs4 import BeautifulSoup
    def strip_html(t: str) -> str:
        return BeautifulSoup(t or "", "html.parser").get_text(" ")
except Exception:
    def strip_html(t: str) -> str:
        return re.sub(r"<[^>]+>", " ", t or "")

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

@st.cache_resource
def load_artifacts():
    model_path = MODELS / "model_calibrated.joblib"
    if not model_path.exists():
        model_path = MODELS / "model.joblib"
    vec_path = MODELS / "vectorizer.joblib"
    if not model_path.exists() or not vec_path.exists():
        raise FileNotFoundError("Missing model/vectorizer in models/.")
    model = joblib.load(model_path)
    vec   = joblib.load(vec_path)
    return model, vec, model_path

def clean_text(subject: str, message: str, max_len: int = 20000) -> str:
    text = f"{subject or ''} {message or ''}".strip()
    text = strip_html(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_len]

def predict(model, vec, text: str):
    X = vec.transform([text])
    raw = int(model.predict(X)[0])

    conf, score, ctype = None, None, "none"
    if hasattr(model, "predict_proba"):
        conf = float(model.predict_proba(X)[0, 1]); score, ctype = conf, "probability"
    elif hasattr(model, "decision_function"):
        margin = float(model.decision_function(X)[0])
        conf = float(1.0 / (1.0 + np.exp(-margin)))   # map margin → 0..1 for display
        score, ctype = margin, "margin_logistic"

    explain = None
    coef = getattr(model, "coef_", None)
    if coef is not None:
        coef = coef.ravel()
        if not hasattr(vec, "_terms_"):
            rev = [None] * len(vec.vocabulary_)
            for t, i in vec.vocabulary_.items():
                rev[i] = t
            vec._terms_ = np.array(rev, dtype=object)
        row = X.tocoo()
        contribs = [(vec._terms_[j], float(v * coef[j])) for j, v in zip(row.col, row.data)]
        contribs.sort(key=lambda x: x[1], reverse=True)
        topk = 8
        explain = {
            "spam_terms": [(t, c) for t, c in contribs[:topk] if t],
            "ham_terms":  [(t, c) for t, c in contribs[-topk:] if t],
        }
    return raw, conf, score, ctype, explain

# ---------- UI ----------
st.set_page_config(page_title="PhishGuard", layout="centered")
st.title("📧 PhishGuard — Email Spam/Phish Detector")

model, vec, model_path = load_artifacts()
st.caption(f"Model: `{model_path.name}` • Vocabulary size: {len(vec.vocabulary_):,}")

with st.form("email_form"):
    c1, c2 = st.columns(2)
    subject = c1.text_input("Subject", "")
    threshold = c2.slider("Spam threshold", 0.0, 1.0, 0.50, 0.01,
                          help="Decide spam if confidence ≥ threshold")
    message = st.text_area("Message (paste the raw body)", height=240,
                           placeholder="Paste the email body here…")
    submitted = st.form_submit_button("Analyze")

if submitted:
    text = clean_text(subject, message)
    if not text:
        st.warning("Please paste a subject or message."); st.stop()

    raw, conf, score, ctype, explain = predict(model, vec, text)
    decision = raw if conf is None else int(conf >= threshold)
    label = "SPAM" if decision == 1 else "HAM"
    st.subheader(f"Result: **{label}**")

    if conf is not None:
        st.progress(min(max(conf, 0.0), 1.0),
                    text=f"Spam confidence: {conf:.2%} ({ctype})")
        st.caption(f"Threshold: {threshold:.2f}")
    else:
        st.caption("Model does not expose probabilities; using label only.")

    if explain:
        s1, s2 = st.columns(2)
        s1.markdown("**Top phrases pushing SPAM**")
        s1.write([f"{t} ({c:.3f})" for t, c in explain["spam_terms"][:6]])
        s2.markdown("**Top phrases pushing HAM**")
        s2.write([f"{t} ({c:.3f})" for t, c in explain["ham_terms"][:6]])

st.markdown("---")
st.caption("Note: For sensitive emails, verify manually even if marked as ham.")

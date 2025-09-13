"""
Text preprocessing utilities for PhishGuard.
- Normalizes URLs/emails/numbers
- Collapses whitespace
"""
import re

URL_RE = re.compile(r"http\S+")
EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.[A-Za-z]{2,}\b")
NUM_RE = re.compile(r"\d+")
WS_RE = re.compile(r"\s+")

def clean(text: str) -> str:
    if text is None:
        return ""
    text = text.lower()
    text = URL_RE.sub(" __URL__ ", text)
    text = EMAIL_RE.sub(" __EMAIL__ ", text)
    text = NUM_RE.sub(" __NUM__ ", text)
    text = WS_RE.sub(" ", text).strip()
    return text

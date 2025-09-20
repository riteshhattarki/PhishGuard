# src/prepare_data.py
from typing import Optional
from pathlib import Path
import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# ------------------------------
# Path setup (robust to CWD)
# ------------------------------
# This file lives in phishguard/src/ -> project root is parent of 'src'
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
os.makedirs(PROC_DIR, exist_ok=True)

def _find_csv() -> Optional[Path]:
    """Try common locations for the labeled Enron CSV."""
    candidates = [
        RAW_DIR / "enron" / "enron_spam_data.csv",
        RAW_DIR / "enron_spam_data.csv",
        *RAW_DIR.glob("enron_spam_data*/enron_spam_data.csv"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None

def _load_from_csv(csv_path: Path) -> pd.DataFrame:
    print(f"→ Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    # Normalize columns that may vary by source
    if "Subject" not in df.columns and "subject" in df.columns:
        df.rename(columns={"subject": "Subject"}, inplace=True)
    if "Message" not in df.columns and "message" in df.columns:
        df.rename(columns={"message": "Message"}, inplace=True)
    if "Spam/Ham" not in df.columns:
        # Try other common labels
        for cand in ("Label", "label", "target", "is_spam", "spam"):
            if cand in df.columns:
                df.rename(columns={cand: "Spam/Ham"}, inplace=True)
                break

    # Map labels to "Ham"/"Spam" for consistency if needed
    if df["Spam/Ham"].dtype != object:
        # assume 1=spam, 0=ham
        df["Spam/Ham"] = df["Spam/Ham"].map({0: "Ham", 1: "Spam"})

    return df

def _load_from_folders(ham_dir: Path, spam_dir: Path) -> pd.DataFrame:
    """Fallback: build a dataframe from raw ham/spam text files."""
    print(f"→ Building dataframe from folders:\n   ham:  {ham_dir}\n   spam: {spam_dir}")
    rows: list[tuple[str, str, str]] = []  # (Subject, Message, Spam/Ham)

    def parse_text_file(p: Path) -> tuple[str, str]:
        text = p.read_text(encoding="latin_1", errors="strict")
        # If the first line starts with "Subject: ", split; else treat whole file as message
        if "\n" in text:
            first, rest = text.split("\n", 1)
        else:
            first, rest = text, ""
        subject_prefix = "Subject: "
        subject = first[len(subject_prefix):] if first.startswith(subject_prefix) else first
        message = rest
        return subject, message

    if ham_dir.exists():
        for f in sorted(ham_dir.glob("*.txt")):
            subj, msg = parse_text_file(f)
            rows.append((subj, msg, "Ham"))
    if spam_dir.exists():
        for f in sorted(spam_dir.glob("*.txt")):
            subj, msg = parse_text_file(f)
            rows.append((subj, msg, "Spam"))

    if not rows:
        raise FileNotFoundError(
            "No ham/spam text files found. Expected at least one of:\n"
            f" - {ham_dir}/*.txt\n - {spam_dir}/*.txt"
        )

    df = pd.DataFrame(rows, columns=["Subject", "Message", "Spam/Ham"])
    return df

def load_dataframe() -> pd.DataFrame:
    csv_path = _find_csv()
    if csv_path:
        df = _load_from_csv(csv_path)
    else:
        # Fallback to folder structure
        df = _load_from_folders(RAW_DIR / "ham", RAW_DIR / "spam")

    # Ensure required columns exist
    for col in ("Subject", "Message", "Spam/Ham"):
        if col not in df.columns:
            raise ValueError(f"Missing expected column '{col}' in dataframe.")

    # Clean/normalize
    df["Subject"] = df["Subject"].fillna("")
    df["Message"] = df["Message"].fillna("")
    df["text"] = (df["Subject"] + " " + df["Message"]).str.strip()

    # Binary labels: 0=ham, 1=spam
    df["label"] = df["Spam/Ham"].astype(str).str.strip().str.capitalize().map({"Ham": 0, "Spam": 1})
    if df["label"].isna().any():
        bad = df.loc[df["label"].isna(), "Spam/Ham"].value_counts()
        raise ValueError(f"Unrecognized labels found: {bad.to_dict()}")

    # Drop empty texts (rare but safe)
    df = df[~df["text"].str.fullmatch(r"\s*")]
    df = df[["text", "label"]].reset_index(drop=True)
    print(f"→ Loaded {len(df):,} emails "
          f"({(df['label']==0).sum():,} ham, {(df['label']==1).sum():,} spam)")
    return df

def main():
    df = load_dataframe()

    # Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"],
        test_size=0.20, random_state=42, stratify=df["label"]
    )

    # TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        stop_words="english",
        lowercase=True,
        max_features=10000,     # tune later
        ngram_range=(1, 2),     # unigrams + bigrams help catch phishing-y phrases
        min_df=2                # ignore ultra-rare tokens
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Save artifacts
    joblib.dump(vectorizer, PROC_DIR / "tfidf_vectorizer.pkl")
    joblib.dump((X_train_tfidf, y_train.reset_index(drop=True)), PROC_DIR / "train.pkl")
    joblib.dump((X_test_tfidf, y_test.reset_index(drop=True)), PROC_DIR / "test.pkl")

    print("\nSaved artifacts:")
    print(f" - {PROC_DIR / 'tfidf_vectorizer.pkl'}")
    print(f" - {PROC_DIR / 'train.pkl'}")
    print(f" - {PROC_DIR / 'test.pkl'}")
    print(f"\nShapes: train {X_train_tfidf.shape}, test {X_test_tfidf.shape}")

if __name__ == "__main__":
    main()

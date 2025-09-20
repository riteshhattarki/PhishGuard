"""
Module 2: Data Ingestion & Processing
- Load raw ham and spam/phish emails from directories
- Strip headers, normalize text via preprocess.clean
- Deduplicate by hash of cleaned text
- (Optional) balance classes by downsampling the majority
- Save combined CSV and (optional) stratified train/val/test splits
"""
import argparse, os, sys, re, hashlib
from pathlib import Path
from typing import Tuple
import pandas as pd

# --- dual import guard: works in both module and script execution ---
try:
    from .preprocess import clean  # when run as: python -m src.make_dataset
except Exception:
    from src.preprocess import clean  # when run as: python src/make_dataset.py

HEADER_BODY_SPLIT_RE = re.compile(r"\r?\n\r?\n", re.MULTILINE)

def read_text_file(path: Path) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""

def strip_headers(raw: str) -> str:
    parts = HEADER_BODY_SPLIT_RE.split(raw, maxsplit=1)
    return parts[1] if len(parts) > 1 else parts[0]

def html_to_text(s: str) -> str:
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.I)
    s = re.sub(r"<[^>]+>", " ", s)  # strip tags
    return s

def load_dir(dir_path: str, label: str, min_chars: int = 10) -> pd.DataFrame:
    rows = []
    p = Path(dir_path)
    if not p.exists():
        return pd.DataFrame(columns=["text","label"])
    for fp in p.rglob("*"):
        if fp.is_file():
            raw = read_text_file(fp)
            if not raw:
                continue
            body = strip_headers(raw)
            body = html_to_text(body)
            txt = clean(body)
            if len(txt) < min_chars:
                continue
            rows.append({"text": txt, "label": label, "source": str(fp)})
    return pd.DataFrame(rows)

def dedupe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text_hash"] = df["text"].map(lambda s: hashlib.sha1(s.encode("utf-8")).hexdigest())
    before = len(df)
    df = df.drop_duplicates(subset=["text_hash"]).drop(columns=["text_hash"])
    after = len(df)
    print(f"[INFO] Dedupe: {before} -> {after} rows")
    return df

def balance_downsample(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    counts = df["label"].value_counts()
    if len(counts) < 2:
        return df
    min_class = counts.min()
    parts = []
    for label, grp in df.groupby("label"):
        parts.append(grp.sample(n=min_class, random_state=seed) if len(grp) > min_class else grp)
    return pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=seed)

def split_stratified(df: pd.DataFrame, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from sklearn.model_selection import train_test_split
    y = (df["label"].str.lower()=="phish").astype(int)
    train_df, temp_df = train_test_split(df, test_size=0.30, stratify=y, random_state=seed)  # 70%
    y_temp = (temp_df["label"].str.lower()=="phish").astype(int)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=y_temp, random_state=seed)  # 15/15
    return train_df, val_df, test_df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ham_dir", required=True, help="Directory with ham emails (.txt/.eml).")
    ap.add_argument("--spam_dir", required=True, help="Directory with spam/phish emails.")
    ap.add_argument("--out_csv", default="data/processed/emails.csv")
    ap.add_argument("--make_splits", action="store_true")
    ap.add_argument("--balance", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_chars", type=int, default=10)
    args = ap.parse_args()

    print(f"[INFO] Loading ham from {args.ham_dir} and spam/phish from {args.spam_dir}")
    ham = load_dir(args.ham_dir, "ham", min_chars=args.min_chars)
    spam = load_dir(args.spam_dir, "phish", min_chars=args.min_chars)
    print(f"[INFO] Loaded counts: ham={len(ham)} phish={len(spam)}")

    df = pd.concat([ham, spam], ignore_index=True)
    if df.empty:
        print("[ERROR] No data found. Check your directories and file contents.")
        sys.exit(1)

    df = dedupe(df)
    if args.balance:
        df = balance_downsample(df, seed=args.seed)
        print(f"[INFO] After balancing: {df['label'].value_counts().to_dict()}")

    out_dir = os.path.dirname(args.out_csv)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    df[["text","label"]].to_csv(args.out_csv, index=False)
    print(f"[OK] Wrote combined CSV -> {args.out_csv} ({len(df)} rows)")

    if args.make_splits:
        try:
            train_df, val_df, test_df = split_stratified(df, seed=args.seed)
            train_df[["text","label"]].to_csv("data/processed/train.csv", index=False)
            val_df[["text","label"]].to_csv("data/processed/val.csv", index=False)
            test_df[["text","label"]].to_csv("data/processed/test.csv", index=False)
            print("[OK] Wrote splits -> data/processed/train.csv, val.csv, test.csv")
        except Exception as e:
            print(f"[WARN] Split failed (likely too small per class): {e}")
            print("[INFO] You can still train using the combined CSV.")

if __name__ == "__main__":
    main()

# PhishGuard

A minimal, production-style pipeline to classify emails as **phish** vs **ham**.

## Quickstart

```bash
# 1) Create and activate a virtual env (example: Python 3.10+)
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) (Optional) Inspect/edit config
cat configs/baseline.yaml

# 4) Train baseline (uses sample data)
python -m src.train_baseline --data data/processed/emails.csv --outdir models

# 5) Predict on a sample JSON
python -m src.predict --model_dir models --input sample_email.json
```

## Repo layout
```
phishguard/
  data/
    raw/
    processed/
  models/
  reports/
  src/
  api/
  notebooks/
  configs/
```

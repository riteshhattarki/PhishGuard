import argparse, json, os, joblib, yaml
from .preprocess import clean

def load_config(path:str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main(args):
    pipe = joblib.load(os.path.join(args.model_dir, "baseline_pipe.joblib"))
    # Calibrator optional
    cal_path = os.path.join(args.model_dir, "calibrator.joblib")
    calibrator = joblib.load(cal_path) if os.path.exists(cal_path) else None

    with open(args.input, "r") as f:
        payload = json.load(f)
    subject = payload.get("subject", "")
    body = payload.get("body", "")
    text = clean(f"{subject}\n\n{body}")

    # Predict label and probability
    label_idx = pipe.predict([text])[0]
    label = "phish" if label_idx == 1 else "ham"
    prob = None
    if calibrator is not None:
        X = pipe.named_steps["tfidf"].transform([text])
        prob = float(calibrator.predict_proba(X)[:,1][0])

    out = {"label": label, "prob_phish": prob}
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--input", type=str, default="sample_email.json")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    args = parser.parse_args()
    main(args)

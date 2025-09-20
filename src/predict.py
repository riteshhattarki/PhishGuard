import argparse, json, os, joblib, yaml
from .preprocess import clean

def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main(args):
    # Load pipeline and optional calibrator
    pipe = joblib.load(os.path.join(args.model_dir, "baseline_pipe.joblib"))
    cal_path = os.path.join(args.model_dir, "calibrator.joblib")
    calibrator = joblib.load(cal_path) if os.path.exists(cal_path) else None

    # Load thresholds from config
    cfg = load_config(args.config) if os.path.exists(args.config) else {}
    thr_prob = cfg.get("thresholding", {}).get("threshold", 0.5)
    thr_margin = cfg.get("thresholding", {}).get("margin_threshold", 0.0)

    # Read and clean input
    with open(args.input, "r") as f:
        payload = json.load(f)
    subject = payload.get("subject", "")
    body = payload.get("body", "")
    text = clean(f"{subject}\n\n{body}")

    # --- Decide using probability if calibrator exists; otherwise use SVM margin ---
    prob = None
    if calibrator is not None:
        X = pipe.named_steps["tfidf"].transform([text])
        prob = float(calibrator.predict_proba(X)[:, 1][0])
        label_idx = 1 if prob >= thr_prob else 0
    else:
        # LinearSVC decision function: >0 means "phish-ish", <0 means "ham-ish"
        margin = float(pipe.decision_function([text])[0])
        label_idx = 1 if margin >= thr_margin else 0

    label = "phish" if label_idx == 1 else "ham"

    # Include margin in output if no calibrator (helps you see confidence)
    out = {"label": label, "prob_phish": prob}
    if calibrator is None:
        out["margin"] = float(pipe.decision_function([text])[0])

    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--input", type=str, default="sample_email.json")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    args = parser.parse_args()
    main(args)

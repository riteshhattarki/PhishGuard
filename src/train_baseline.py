import argparse, json, os, joblib, yaml # Starting by importing all packages/dependencies
import pandas as pd # Pandas deals with tabular data --> use pandas to load emails.csv
from sklearn.model_selection import train_test_split # Splits your data into train vs. test set
from sklearn.feature_extraction.text import TfidfVectorizer # Turns text into numbers (how important each word/phrase is)
from sklearn.svm import LinearSVC # The actual classfiier (SVM) --> this will classify phish vs. safe
from sklearn.pipeline import Pipeline # Chains multiple steps (TF-IDF -> classifier) to train in one go
from sklearn.metrics import classification_report, roc_auc_score # Deals with precision and probabilties
from sklearn.calibration import CalibratedClassifierCV # Wraps classifier models (such as LinearSVC) to produce the probability score
from .preprocess import clean # Cleans up user input


# This function opens the YAML and returns a Python dict of everything in the baseline.yaml file
# Basically, return everything in baseline.yaml as a dictionary
def load_config(path:str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# cfg takes in the returned dict from above
def build_pipeline(cfg):
    # Takes everything from "vectorizer" ([1, 2], 1, 1.0, true)
    # Passed into TF-IDF vectorizer
    # Turns the text into numbers (for word importance)
    # Generating a vectorizer that has numbers representing word importance
    vec = TfidfVectorizer(
        ngram_range=tuple(cfg["vectorizer"]["ngram_range"]),
        min_df=cfg["vectorizer"]["min_df"],
        max_df=cfg["vectorizer"]["max_df"],
        sublinear_tf=cfg["vectorizer"]["sublinear_tf"],
    )
    # Sets up the classifier model (to detect phish vs. safe) by taking more values out of the cfg dict
    clf = LinearSVC(C=cfg["model"]["C"], random_state=cfg["seed"])
    
    # The pipeline combines the result of the vector and the classified model
    pipe = Pipeline([("tfidf", vec), ("clf", clf)])
    # Return this combination
    return pipe

def main(args):
    cfg = load_config(args.config) # Call load_config to return dictionary of baseline.yaml file
    os.makedirs(args.outdir, exist_ok=True)

    # Reads emails.csv, asserts text and label columns, cleans data, and maps labels
    # 1 for phish, everything else 0
    df = pd.read_csv(args.data)
    assert {"text","label"}.issubset(df.columns), "CSV must have columns: text,label"
    df["text"] = df["text"].astype(str).map(clean)
    df["y"] = (df["label"].str.lower()=="phish").astype(int)

    # Ensures that both phish and safe (2 classes) are present
    # Dataset must have enough rows to split without errors
    n = len(df)
    n_classes = df["y"].nunique()
    
    if n_classes < 2:
        raise ValueError(
            f"Dataset has only {n_classes} class after preprocessing. "
            "Make sure both 'phish' and 'ham' exist in the CSV."
        )
    
    if n < 2 * n_classes:
        raise ValueError(
            f"Dataset too small for stratified split (n={n}, classes={n_classes}). "
            "Add a few more rows or reduce classes."
        )

    
    # Decides on the test size for the data (how much data should the model take in)
    min_test_frac = max(0.2, n_classes / n)
    test_size = min(0.5, min_test_frac)

    # Split the dataset into train and test pieces
    # X_train (training email texts)
    # X_test (testing email texts --> evaluation)
    # y_train (training for labels phish vs. safe
    # y_test (testing labels)
    # df stands for DataFrame and takes in the text column in emails.csv
    # df["y"] takes in the labels
    # Takes in the test size (calculated as 0.2)
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["y"], test_size=test_size, stratify=df["y"], random_state=cfg["seed"]
    )

    pipe = build_pipeline(cfg) # Builds the pipeline by calling the build_pipeline function above (includes the vectorizer and the classifier model)
    pipe.fit(X_train, y_train) # Trains the model

    # Save vectorizer + base classifier
    joblib.dump(pipe, os.path.join(args.outdir, "baseline_pipe.joblib"))

    # Optionally calibrate for probabilities --> only run if calibration enabled is true
    if cfg["calibration"]["enabled"]:
        cal = CalibratedClassifierCV(
            pipe.named_steps["clf"],
            method=cfg["calibration"]["method"],
            cv=cfg["calibration"]["cv"]
        )
        # Fit calibration on training data transformed by vectorizer
        Xtr = pipe.named_steps["tfidf"].transform(X_train)
        cal.fit(Xtr, y_train)
        joblib.dump(cal, os.path.join(args.outdir, "calibrator.joblib"))
    else:
        cal = None

    # Evaluate by converting test emails into TF-IDF features and output phish/safe labels
    Xt = pipe.named_steps["tfidf"].transform(X_test)
    pred = pipe.predict(X_test)
    print(classification_report(y_test, pred, target_names=["ham","phish"]))

    if cal is not None:
        proba = cal.predict_proba(Xt)[:,1]
        try:
            print("ROC AUC:", roc_auc_score(y_test, proba))
        except Exception as e:
            print("ROC AUC not available:", e)

if __name__ == "__main__":
    # 1.) Loads settings
    # 2.) Reads and cleans the dataset
    # 3.) Splits into train/test safely
    # 4.) Builds + trains the pipeline
    # 5.) Saves the trained model (and calibrator if applicable)
    # 6.) Tests the model and prints evaluation results
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/processed/emails.csv")
    parser.add_argument("--outdir", type=str, default="models")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    args = parser.parse_args()
    main(args)

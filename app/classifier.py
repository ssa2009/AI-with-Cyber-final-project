"""
Trains a binary text classifier: does this input contain PII or not?
 
Approach: TF-IDF (word + char n-grams, to catch structured patterns like
emails/phone formats as well as vocabulary) + Logistic Regression.
LinearSVC is compared as well since it's a strong baseline for sparse
high-dimensional text features.
"""
 
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "pii_dataset.jsonl")
MODEL_DIR = os.path.join(BASE_DIR, "models")
 
 
def load_data():
    df = pd.read_json(DATA_PATH, lines=True)
    return df
 
 
def main():
    df = load_data()
    X = df["source_text"]
    y = df["has_pii"]
 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
 
    # Word-level TF-IDF for vocabulary signal + char n-gram TF-IDF for
    # structural patterns (emails/phone numbers share character shapes
    # even with unseen names/domains).
    word_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)
 
    from scipy.sparse import hstack
 
    def featurize(fit, texts):
        if fit:
            wv = word_vec.fit_transform(texts)
            cv = char_vec.fit_transform(texts)
        else:
            wv = word_vec.transform(texts)
            cv = char_vec.transform(texts)
        return hstack([wv, cv])
 
    X_train_feats = featurize(True, X_train)
    X_test_feats = featurize(False, X_test)
 
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "linear_svc": LinearSVC(class_weight="balanced"),
    }
 
    results = {}
    for name, clf in models.items():
        clf.fit(X_train_feats, y_train)
        preds = clf.predict(X_test_feats)
        acc = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds, target_names=["no_pii", "has_pii"])
        cm = confusion_matrix(y_test, preds)
        results[name] = {"accuracy": acc, "report": report, "cm": cm, "model": clf}
        print(f"\n=== {name} ===")
        print(f"Accuracy: {acc:.4f}")
        print(report)
        print("Confusion matrix [[TN, FP], [FN, TP]]:")
        print(cm)
 
    # Pick the better model by F1 on the positive (has_pii) class -- recall
    # on has_pii matters most here, since a missed PII leak is worse than a
    # false-positive warning.
    best_name = max(results, key=lambda n: results[n]["accuracy"])
    best_model = results[best_name]["model"]
    print(f"\nSelected model: {best_name}")
 
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_model, f"{MODEL_DIR}/classifier.joblib")
    joblib.dump(word_vec, f"{MODEL_DIR}/word_vectorizer.joblib")
    joblib.dump(char_vec, f"{MODEL_DIR}/char_vectorizer.joblib")
    print(f"Saved model + vectorizers to {MODEL_DIR}/")
 
 
if __name__ == "__main__":
    main()

"""
PIIGuard: the inference-time wrapper a real app would call before enabling
the submit button.
 
Design: two complementary layers, since a single sklearn text classifier
can tell you "this looks risky" but can't point at *what* is risky.
 
  1. Regex layer -- deterministic, near-perfect precision/recall for
     structured PII with fixed formats: email, phone, SSN, credit card,
     IPv4, and dates-as-DOB. Used to produce exact highlighted spans.
 
  2. ML layer -- the trained TF-IDF + Logistic Regression classifier from
     train_classifier.py. Catches free-text PII the regex can't (names,
     street addresses, "my old apartment was on Oak Street") and acts as
     a second opinion / overall confidence score.
 
The verdict combines both: if EITHER layer flags the input, the submit
button stays disabled until the user acknowledges. Regex hits are always
surfaced as exact spans; the ML layer's vote is used mainly to catch the
things regex structurally cannot (names, addresses in prose).
"""
 
import os
import re
import joblib
from scipy.sparse import hstack
 
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
 
REGEX_PATTERNS = {
    "EMAIL": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "PHONE": re.compile(r"(\+?\d{1,2}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}"),
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "CREDITCARDNUMBER": re.compile(r"\b(?:\d{4}[\s-]){3}\d{4}\b"),
    "DOB": re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b|\bDOB:\s?\d{1,2}/\d{1,2}/\d{2,4}\b"),
}
 
# IPs are deliberately excluded from the "definitely block" regex set --
# they're extremely common in benign technical text (see hard_test_set.py
# false positives) and aren't PII on their own without more context.
 
 
class PIIGuard:
    def __init__(self, model_dir=MODEL_DIR, ml_threshold=0.5):
        self.clf = joblib.load(f"{model_dir}/classifier.joblib")
        self.word_vec = joblib.load(f"{model_dir}/word_vectorizer.joblib")
        self.char_vec = joblib.load(f"{model_dir}/char_vectorizer.joblib")
        self.ml_threshold = ml_threshold
        self.has_proba = hasattr(self.clf, "predict_proba")
 
    def _regex_spans(self, text):
        spans = []
        for label, pattern in REGEX_PATTERNS.items():
            for m in pattern.finditer(text):
                spans.append({"start": m.start(), "end": m.end(),
                               "label": label, "text": m.group()})
        spans.sort(key=lambda s: s["start"])
        return spans
 
    def _ml_verdict(self, text):
        feats = hstack([self.word_vec.transform([text]), self.char_vec.transform([text])])
        pred = int(self.clf.predict(feats)[0])
        if self.has_proba:
            confidence = float(self.clf.predict_proba(feats)[0][pred])
        else:
            # LinearSVC has no predict_proba; use decision_function distance
            # as a rough confidence proxy instead.
            score = float(self.clf.decision_function(feats)[0])
            confidence = 1 / (1 + pow(2.718281828, -abs(score)))
        return pred, confidence
 
    def check(self, text: str) -> dict:
        """
        Returns a verdict dict:
          should_block : bool  -- whether the submit button should stay disabled
          spans        : list  -- exact structured PII matches to highlight
          ml_flagged   : bool  -- whether the free-text classifier also flagged it
          ml_confidence: float
          message      : str   -- user-facing warning text
        """
        regex_spans = self._regex_spans(text)
        ml_pred, ml_conf = self._ml_verdict(text)
 
        should_block = bool(regex_spans) or bool(ml_pred)
 
        if regex_spans:
            kinds = sorted(set(s["label"] for s in regex_spans))
            message = f"This looks like it contains {', '.join(kinds)}. Please review before submitting."
        elif ml_pred:
            message = "This may contain personal information (e.g. a name or address). Please review before submitting."
        else:
            message = "No personal information detected."
 
        return {
            "should_block": should_block,
            "spans": regex_spans,
            "ml_flagged": bool(ml_pred),
            "ml_confidence": round(ml_conf, 3),
            "message": message,
        }
 
 
if __name__ == "__main__":
    guard = PIIGuard()
    demo_inputs = [
        "Summarize this article about renewable energy for me.",
        "Please email jane.doe@gmail.com and cc her at (713) 555-0192.",
        "My old apartment was on Oak Street, does that help find comps?",
        "Ping 10.0.4.221, that's the staging server having issues today.",
    ]
    for text in demo_inputs:
        result = guard.check(text)
        print("-" * 70)
        print("INPUT:", text)
        print("VERDICT:", result)

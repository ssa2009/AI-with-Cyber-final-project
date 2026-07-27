import re
from pathlib import Path
from typing import Any

import joblib


MODEL_PATH = Path("model/pii_classifier.joblib")


PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "EMAIL": re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    ),
    "PHONE_NUMBER": re.compile(
        r"(?<!\w)(?:\+?1[-.\s]?)?"
        r"(?:\(?\d{3}\)?[-.\s]?)"
        r"\d{3}[-.\s]?\d{4}(?!\w)"
    ),
    "IP_ADDRESS": re.compile(
        r"\b(?:"
        r"(?:25[0-5]|2[0-4]\d|1?\d?\d)\."
        r"){3}"
        r"(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"
    ),
    "US_SSN": re.compile(
        r"\b\d{3}-\d{2}-\d{4}\b"
    ),
    "CREDIT_CARD_LIKE_NUMBER": re.compile(
        r"\b(?:\d[ -]*?){13,19}\b"
    ),
    "ZIP_CODE": re.compile(
        r"\b\d{5}(?:-\d{4})?\b"
    ),
}


class PIIDetector:
    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model was not found at {model_path}. "
                "Run 'python train_model.py' first."
            )

        self.model = joblib.load(model_path)

    def find_pattern_matches(
        self,
        text: str,
    ) -> list[dict[str, Any]]:
        matches: list[dict[str, Any]] = []

        for pii_type, pattern in PII_PATTERNS.items():
            for match in pattern.finditer(text):
                matches.append(
                    {
                        "type": pii_type,
                        "value": match.group(),
                        "start": match.start(),
                        "end": match.end(),
                    }
                )

        matches.sort(key=lambda item: item["start"])
        return matches

    def predict(self, text: str) -> dict[str, Any]:
        cleaned_text = text.strip()

        if not cleaned_text:
            return {
                "contains_pii": False,
                "confidence": 0.0,
                "pattern_matches": [],
                "reason": "No text was entered.",
            }

        probabilities = self.model.predict_proba([cleaned_text])[0]
        pii_probability = float(probabilities[1])

        pattern_matches = self.find_pattern_matches(cleaned_text)

        # A lower threshold increases recall.
        ml_detected = pii_probability >= 0.60
        pattern_detected = len(pattern_matches) > 0

        contains_pii = ml_detected or pattern_detected

        if pattern_detected:
            reason = "A known PII pattern was detected."
        elif ml_detected:
            reason = "The machine-learning model detected possible PII."
        else:
            reason = "No likely PII was detected."

        return {
            "contains_pii": contains_pii,
            "confidence": round(pii_probability, 4),
            "pattern_matches": pattern_matches,
            "reason": reason,
        }
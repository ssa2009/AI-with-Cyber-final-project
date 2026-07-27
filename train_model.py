import random
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


MODEL_PATH = Path("model/pii_classifier.joblib")
RANDOM_STATE = 42


# Sentence templates. "{pii}" gets filled with a realistic piece of PII
# for the positive class, or a generic non-identifying word/phrase for
# the negative class. Same templates are used for both classes, so the
# only real difference between the two classes is the PII content
# itself -- not sentence style, length, or formatting.
TEMPLATES = [
    "Hi, my name is {pii}, nice to meet you.",
    "You can reach me at {pii} if you have questions.",
    "Please send the invoice to {pii}.",
    "My phone number is {pii}, call anytime.",
    "I live at {pii}, near the park.",
    "Can you confirm my appointment on {pii}?",
    "The account belongs to {pii}.",
    "I was born on {pii}.",
    "Please update my address to {pii}.",
    "Contact {pii} for more details.",
    "I work at {pii} in the marketing department.",
    "My social security number is {pii}.",
    "I recently moved to {pii}.",
    "You can find me on {pii}.",
    "My credit card number is {pii}.",
    "I like pizza and hiking on weekends.",
    "The weather has been really nice lately.",
    "Let's schedule the meeting for next week.",
    "I really enjoyed the movie we watched.",
    "Can you help me debug this code?",
    "The project is due at the end of the month.",
    "I think we should order tacos for lunch.",
    "This book was recommended by a friend.",
    "The traffic this morning was terrible.",
    "I'm learning to play the guitar.",
]

PII_VALUES = [
    "John Smith",
    "Sarah Johnson",
    "michael.brown@gmail.com",
    "jane.doe@yahoo.com",
    "555-123-4567",
    "(212) 555-0199",
    "742 Evergreen Terrace, Springfield",
    "1600 Pennsylvania Ave, Washington DC",
    "March 3, 1990",
    "01/15/1985",
    "123-45-6789",
    "4111 1111 1111 1111",
    "linkedin.com/in/janedoe123",
    "Acme Corporation",
    "Emily Davis",
    "robert.wilson@outlook.com",
    "555-987-6543",
    "88 Main Street, Boston",
    "June 12, 1978",
    "987-65-4321",
]

NON_PII_VALUES = [
    "the office",
    "next Tuesday",
    "a coworker",
    "the usual place",
    "our team",
    "sometime soon",
    "the store downtown",
    "a friend of mine",
    "the community center",
    "last weekend",
    "the local gym",
    "a colleague",
    "the new cafe",
    "another department",
    "the library",
    "a nearby park",
    "the conference room",
    "an old classmate",
    "the neighborhood",
    "the usual time",
]


def build_synthetic_dataset(examples_per_class: int = 1500) -> pd.DataFrame:
    """
    Build a small, self-contained synthetic dataset. Both classes use
    the exact same sentence templates, so the model can't cheat by
    learning template/style differences -- it has to learn to
    recognize actual PII content (names, emails, phone numbers,
    addresses, dates, SSNs, card numbers) versus generic filler.
    """
    rng = random.Random(RANDOM_STATE)

    rows = []

    for _ in range(examples_per_class):
        template = rng.choice(TEMPLATES)
        pii_value = rng.choice(PII_VALUES)
        text = template.format(pii=pii_value) if "{pii}" in template else template
        rows.append({"text": text, "label": 1})

    for _ in range(examples_per_class):
        template = rng.choice(TEMPLATES)
        filler_value = rng.choice(NON_PII_VALUES)
        text = template.format(pii=filler_value) if "{pii}" in template else template
        rows.append({"text": text, "label": 0})

    data = pd.DataFrame(rows).drop_duplicates(subset="text")

    return data.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


def train_model() -> None:
    data = build_synthetic_dataset()

    print(f"Total training examples: {len(data):,}")
    print(data["label"].value_counts())

    x_train, x_test, y_train, y_test = train_test_split(
        data["text"],
        data["label"],
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=data["label"],
    )

    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(2, 5),
                    min_df=2,
                    max_features=100_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1_000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    print("Training model...")
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["No PII", "Contains PII"],
        )
    )

    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train_model()
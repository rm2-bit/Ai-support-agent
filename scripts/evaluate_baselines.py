import json
import csv
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

TRAIN_FILE = Path("data/processed/train_labeled.jsonl")
GOLDEN_FILE = Path("data/golden/golden_set.csv")


def load_training_data():
    texts = []
    labels = []

    with open(TRAIN_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)

            texts.append(row["text"])
            labels.append(row["intent"])

    return texts, labels


def load_golden_data():
    texts = []
    labels = []

    with open(GOLDEN_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            text = row["customer_messages"].strip()
            intent = row["intent"].strip()

            if text and intent:
                texts.append(text)
                labels.append(intent)

    return texts, labels


def evaluate():
    print("=" * 70)
    print("BASELINE 2 — TF-IDF + LOGISTIC REGRESSION")
    print("=" * 70)

    print("\nLoading training data...")
    train_texts, train_labels = load_training_data()

    print(f"Training examples: {len(train_texts)}")

    print("\nLoading golden set...")
    test_texts, test_labels = load_golden_data()

    print(f"Golden examples: {len(test_texts)}")

    # Convert text into TF-IDF features.
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_features=50000,
        sublinear_tf=True,
    )

    print("\nBuilding TF-IDF features...")
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    print(f"TF-IDF train shape: {X_train.shape}")
    print(f"TF-IDF test shape:  {X_test.shape}")

    # Train classifier.
    print("\nTraining Logistic Regression...")

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )

    model.fit(X_train, train_labels)

    # Predict golden set.
    predictions = model.predict(X_test)

    accuracy = accuracy_score(test_labels, predictions)
    macro_f1 = f1_score(
        test_labels,
        predictions,
        average="macro",
        zero_division=0,
    )
    weighted_f1 = f1_score(
        test_labels,
        predictions,
        average="weighted",
        zero_division=0,
    )

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\nAccuracy:    {accuracy:.4f}")
    print(f"Macro-F1:    {macro_f1:.4f}")
    print(f"Weighted-F1: {weighted_f1:.4f}")

    print("\nPer-intent results:")
    print(
        classification_report(
            test_labels,
            predictions,
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    labels = sorted(set(test_labels))

    matrix = confusion_matrix(
        test_labels,
        predictions,
        labels=labels,
    )

    print("Labels:")
    print(labels)
    print(matrix)


if __name__ == "__main__":
    evaluate()
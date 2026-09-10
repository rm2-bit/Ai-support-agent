import json
import re
import random
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans


# ============================================================
# CONFIG
# ============================================================

INPUT_PATH = Path("data/processed/amazonhelp_threads.jsonl")

SAMPLE_SIZE = 15000
N_CLUSTERS = 20

RANDOM_SEED = 42


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    """
    Clean Twitter-specific noise while keeping the customer's
    actual issue understandable.
    """

    if not isinstance(text, str):
        return ""

    text = text.strip()

    # Remove URLs
    text = re.sub(r"https?://\S+", " ", text)

    # Remove @mentions
    text = re.sub(r"@\w+", " ", text)

    # Remove common Twitter artifacts
    text = re.sub(r"\bRT\b", " ", text)

    # Decode some HTML entities
    text = text.replace("&gt;", ">")
    text = text.replace("&lt;", "<")
    text = text.replace("&amp;", "&")

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def looks_like_support_request(text):
    """
    Keep messages that appear to contain a customer-support issue.

    This is only for intent discovery. It is NOT our final classifier.
    """

    text_lower = text.lower()

    support_terms = [
        # orders / delivery
        "order",
        "package",
        "parcel",
        "delivery",
        "delivered",
        "shipping",
        "shipment",
        "tracking",
        "arrive",
        "arrived",
        "delivery date",

        # payment
        "payment",
        "paid",
        "charge",
        "charged",
        "refund",
        "money",
        "billing",
        "credit",
        "debit",

        # account
        "account",
        "password",
        "login",
        "locked",
        "sign in",

        # returns
        "return",
        "replacement",
        "replace",
        "exchange",

        # amazon services
        "prime",
        "video",
        "music",
        "kindle",
        "alexa",
        "fire tv",
        "firetv",

        # technical
        "error",
        "not working",
        "doesn't work",
        "doesnt work",
        "can't",
        "cannot",
        "problem",
        "issue",
        "broken",

        # subscription
        "subscription",
        "cancel",
        "membership",

        # points / promotions
        "points",
        "coupon",
        "promo",
        "promotion",
        "discount"
    ]

    return any(
        term in text_lower
        for term in support_terms
    )


# ============================================================
# LOAD CUSTOMER MESSAGES
# ============================================================

def load_customer_messages():
    print("Loading conversations...")

    messages = []

    with open(INPUT_PATH, "r", encoding="utf-8") as f:

        for line in f:

            conversation = json.loads(line)

            for message in conversation["messages"]:

                # Only customer messages
                if message.get("speaker") != "customer":
                    continue

                text = clean_text(
                    message.get("text", "")
                )

                # Remove extremely short messages
                if len(text) < 20:
                    continue

                # Keep likely support-related messages
                if not looks_like_support_request(text):
                    continue

                messages.append({
                    "conversation_id": conversation["conversation_id"],
                    "text": text
                })

    print(
        f"Support-related customer messages available: "
        f"{len(messages):,}"
    )

    return messages


# ============================================================
# SAMPLE DATA
# ============================================================

def sample_messages(messages):

    random.seed(RANDOM_SEED)

    if len(messages) <= SAMPLE_SIZE:
        return messages

    return random.sample(messages, SAMPLE_SIZE)


# ============================================================
# CLUSTERING
# ============================================================

def cluster_messages(messages):

    texts = [m["text"] for m in messages]

    print("\nCreating TF-IDF representation...")

    vectorizer = TfidfVectorizer(
        max_features=15000,
        min_df=3,
        max_df=0.95,
        ngram_range=(1, 2),
        sublinear_tf=True
    )

    X = vectorizer.fit_transform(texts)

    print(f"TF-IDF matrix shape: {X.shape}")

    print(f"\nClustering into {N_CLUSTERS} groups...")

    model = MiniBatchKMeans(
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_SEED,
        batch_size=512,
        n_init=10
    )

    labels = model.fit_predict(X)

    return vectorizer, model, labels, X


# ============================================================
# SHOW CLUSTER INFORMATION
# ============================================================

def show_clusters(messages, vectorizer, model, labels, X):

    feature_names = vectorizer.get_feature_names_out()

    print("\n")
    print("=" * 80)
    print("DISCOVERED INTENT CLUSTERS")
    print("=" * 80)

    for cluster_id in range(N_CLUSTERS):

        indices = np.where(labels == cluster_id)[0]

        print("\n")
        print("-" * 80)
        print(f"CLUSTER {cluster_id}")
        print(f"Messages: {len(indices):,}")

        # --------------------------------------------
        # Top TF-IDF terms
        # --------------------------------------------

        center = model.cluster_centers_[cluster_id]

        top_indices = center.argsort()[-12:][::-1]

        top_terms = [
            feature_names[i]
            for i in top_indices
        ]

        print("\nTop terms:")
        print(", ".join(top_terms))

        # --------------------------------------------
        # Representative examples
        # --------------------------------------------

        cluster_matrix = X[indices]

        distances = model.transform(cluster_matrix)[:, cluster_id]

        closest = distances.argsort()[:5]

        print("\nRepresentative examples:")

        for position in closest:

            original_index = indices[position]

            example = messages[original_index]

            print(
                f"\n[{example['conversation_id']}] "
                f"{example['text']}"
            )


# ============================================================
# SAVE DISCOVERY RESULTS
# ============================================================

def save_results(messages, labels):

    output_path = Path(
        "data/processed/intent_discovery.jsonl"
    )

    with open(output_path, "w", encoding="utf-8") as f:

        for message, label in zip(messages, labels):

            result = {
                "conversation_id": message["conversation_id"],
                "text": message["text"],
                "cluster": int(label)
            }

            f.write(
                json.dumps(
                    result,
                    ensure_ascii=False
                ) + "\n"
            )

    print(
        f"\nSaved clustering results to: "
        f"{output_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    messages = load_customer_messages()

    if not messages:
        print("No customer messages found.")
        return

    messages = sample_messages(messages)

    print(
        f"Using {len(messages):,} messages "
        f"for intent discovery."
    )

    vectorizer, model, labels, X = cluster_messages(
        messages
    )

    show_clusters(
        messages,
        vectorizer,
        model,
        labels,
        X
    )

    save_results(
        messages,
        labels
    )


if __name__ == "__main__":
    main()
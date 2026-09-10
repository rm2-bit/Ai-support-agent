import pandas as pd
import json
from pathlib import Path


DATA_PATH = Path("data/raw/twcs.csv")
OUTPUT_PATH = Path("data/processed/amazonhelp_conversations.jsonl")

BRAND = "AmazonHelp"


def load_data():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    print(f"Loaded {len(df):,} tweets")

    return df


def build_lookup(df):
    """
    Create tweet_id -> tweet lookup.
    """

    return df.set_index("tweet_id").to_dict("index")


def get_brand_tweets(df):
    """
    Return tweets written by AmazonHelp.
    """

    return df[
        (df["author_id"] == BRAND) &
        (df["inbound"] == False)
    ].copy()


def reconstruct_conversations(df):
    lookup = build_lookup(df)

    brand_tweets = get_brand_tweets(df)

    conversations = []

    for _, brand_tweet in brand_tweets.iterrows():

        parent_id = brand_tweet["in_response_to_tweet_id"]

        if pd.isna(parent_id):
            continue

        try:
            parent_id = int(parent_id)
        except (ValueError, TypeError):
            continue

        if parent_id not in lookup:
            continue

        parent = lookup[parent_id]

        # We only want AmazonHelp responding to a customer.
        if parent["inbound"] != True:
            continue

        conversation = {
            "conversation_id": int(parent_id),
            "brand": BRAND,
            "customer_message": str(parent["text"]),
            "brand_reply": str(brand_tweet["text"]),
            "customer_tweet_id": int(parent_id),
            "brand_tweet_id": int(brand_tweet["tweet_id"]),
        }

        conversations.append(conversation)

    return conversations


def save_conversations(conversations):

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        for conversation in conversations:
            f.write(
                json.dumps(
                    conversation,
                    ensure_ascii=False
                ) + "\n"
            )

    print(
        f"Saved {len(conversations):,} conversations "
        f"to {OUTPUT_PATH}"
    )


def main():

    df = load_data()

    conversations = reconstruct_conversations(df)

    save_conversations(conversations)

    print("\n=== SAMPLE CONVERSATIONS ===")

    for conversation in conversations[:5]:

        print("\nCustomer:")
        print(conversation["customer_message"])

        print("\nAmazonHelp:")
        print(conversation["brand_reply"])

        print("-" * 70)


if __name__ == "__main__":
    main()
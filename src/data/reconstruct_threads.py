import pandas as pd
import json
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

DATA_PATH = Path("data/raw/twcs.csv")
OUTPUT_PATH = Path("data/processed/amazonhelp_threads.jsonl")

BRAND = "AmazonHelp"

# Prevent pathological conversations from dominating the dataset.
MAX_MESSAGES = 20


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    # tweet_id is an integer in the source dataset.
    df["tweet_id"] = df["tweet_id"].astype(int)

    print(f"Loaded {len(df):,} tweets")

    return df


# ============================================================
# LOOKUP
# ============================================================

def build_lookup(df):
    """
    Create:

        tweet_id -> tweet information

    This makes parent/child lookup fast.
    """

    return df.set_index("tweet_id").to_dict("index")


# ============================================================
# RESPONSE IDS
# ============================================================

def get_children(row, lookup):
    """
    Convert response_tweet_id into valid tweet IDs.

    The dataset may contain multiple response IDs.
    """

    value = row["response_tweet_id"]

    if pd.isna(value):
        return []

    ids = []

    for x in str(value).split(","):

        x = x.strip()

        if not x:
            continue

        try:
            tweet_id = int(float(x))
        except (ValueError, TypeError):
            continue

        if tweet_id in lookup:
            ids.append(tweet_id)

    return ids


# ============================================================
# FIND ROOT
# ============================================================

def find_root(tweet_id, lookup):
    """
    Follow in_response_to_tweet_id backwards
    until the beginning of the conversation.

    Example:

        tweet 10
          ↑
        tweet 9
          ↑
        tweet 7

    Root = tweet 7
    """

    visited = set()
    current_id = tweet_id

    while current_id in lookup:

        if current_id in visited:
            break

        visited.add(current_id)

        parent_id = lookup[current_id][
            "in_response_to_tweet_id"
        ]

        if pd.isna(parent_id):
            break

        try:
            parent_id = int(parent_id)
        except (ValueError, TypeError):
            break

        if parent_id not in lookup:
            break

        current_id = parent_id

    return current_id


# ============================================================
# RECONSTRUCT THREAD
# ============================================================

def reconstruct_thread(start_id, lookup):
    """
    Follow the conversation forward.

    Currently we follow the first valid response branch.
    We cap conversations at MAX_MESSAGES.
    """

    messages = []

    current_id = start_id

    visited = set()

    while current_id is not None:

        # Protect against huge/pathological threads.
        if len(messages) >= MAX_MESSAGES:
            break

        # Protect against cycles.
        if current_id in visited:
            break

        visited.add(current_id)

        # Missing tweet.
        if current_id not in lookup:
            break

        tweet = lookup[current_id]

        speaker = (
            "customer"
            if bool(tweet["inbound"])
            else "brand"
        )

        messages.append({
            "tweet_id": int(current_id),
            "speaker": speaker,
            "author_id": str(tweet["author_id"]),
            "created_at": str(tweet["created_at"]),
            "text": str(tweet["text"])
        })

        children = get_children(
            tweet,
            lookup
        )

        if not children:
            break

        # For now, follow the first response.
        current_id = children[0]

    return messages


# ============================================================
# MEANINGFUL CUSTOMER MESSAGE
# ============================================================

def is_meaningful_customer_message(text):
    """
    Remove very short acknowledgements and empty messages.

    We don't want the evaluation set dominated by:

        "thanks"
        "okay"
        "got it"
    """

    if not text:
        return False

    text = str(text).strip()

    if len(text) < 10:
        return False

    low_information = {
        "thanks",
        "thank you",
        "thanks!",
        "thank you!",
        "okay",
        "ok",
        "got it",
        "you're welcome",
        "thx",
    }

    if text.lower() in low_information:
        return False

    return True


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_data()

    lookup = build_lookup(df)

    # --------------------------------------------------------
    # AmazonHelp tweets
    # --------------------------------------------------------

    amazon_tweets = df[
        (df["author_id"] == BRAND) &
        (df["inbound"] == False)
    ]

    print(
        f"AmazonHelp outbound tweets: "
        f"{len(amazon_tweets):,}"
    )

    # --------------------------------------------------------
    # Find conversation roots
    # --------------------------------------------------------

    roots = set()

    for _, row in amazon_tweets.iterrows():

        parent_id = row["in_response_to_tweet_id"]

        if pd.isna(parent_id):
            continue

        try:
            parent_id = int(parent_id)
        except (ValueError, TypeError):
            continue

        if parent_id not in lookup:
            continue

        parent = lookup[parent_id]

        # AmazonHelp should be replying to a customer.
        if parent["inbound"] != True:
            continue

        root_id = find_root(
            parent_id,
            lookup
        )

        if root_id not in lookup:
            continue

        # We want conversations whose root
        # is a customer message.
        if lookup[root_id]["inbound"] == True:
            roots.add(root_id)

    starts = sorted(roots)

    print(
        f"Unique root conversations: "
        f"{len(starts):,}"
    )

    # --------------------------------------------------------
    # Reconstruct conversations
    # --------------------------------------------------------

    threads = []

    for start_id in starts:

        messages = reconstruct_thread(
            start_id,
            lookup
        )

        if len(messages) < 2:
            continue

        # Confirm AmazonHelp appears.
        has_brand = any(
            message["speaker"] == "brand"
            and message["author_id"] == BRAND
            for message in messages
        )

        if not has_brand:
            continue

        # ----------------------------------------------------
        # Meaningful customer messages
        # ----------------------------------------------------

        customer_messages = [
            message
            for message in messages
            if message["speaker"] == "customer"
        ]

        meaningful_customer_messages = [
            message
            for message in customer_messages
            if is_meaningful_customer_message(
                message["text"]
            )
        ]

        # Skip threads containing no useful customer request.
        if not meaningful_customer_messages:
            continue

        # ----------------------------------------------------
        # Conversation metadata
        # ----------------------------------------------------

        brand_messages = [
            message
            for message in messages
            if message["speaker"] == "brand"
        ]

        thread = {
            "conversation_id": int(start_id),
            "brand": BRAND,
            "messages": messages,
            "message_count": len(messages),
            "customer_message_count": len(
                customer_messages
            ),
            "brand_message_count": len(
                brand_messages
            )
        }

        threads.append(thread)

    # --------------------------------------------------------
    # Save JSONL
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        for thread in threads:

            f.write(
                json.dumps(
                    thread,
                    ensure_ascii=False
                ) + "\n"
            )

    print(
        f"\nSaved {len(threads):,} threads to:"
    )

    print(OUTPUT_PATH)

    # ========================================================
    # STATISTICS
    # ========================================================

    if not threads:
        print("\nNo threads found.")
        return

    lengths = [
        thread["message_count"]
        for thread in threads
    ]

    customer_counts = [
        thread["customer_message_count"]
        for thread in threads
    ]

    brand_counts = [
        thread["brand_message_count"]
        for thread in threads
    ]

    print("\n" + "=" * 70)
    print("THREAD STATISTICS")
    print("=" * 70)

    print(
        f"Total threads       : {len(threads):,}"
    )

    print(
        f"Average length      : "
        f"{sum(lengths) / len(lengths):.2f}"
    )

    print(
        f"Median length       : "
        f"{pd.Series(lengths).median():.0f}"
    )

    print(
        f"Maximum length      : {max(lengths)}"
    )

    print(
        f"1-2 messages        : "
        f"{sum(x <= 2 for x in lengths):,}"
    )

    print(
        f"3-4 messages        : "
        f"{sum(3 <= x <= 4 for x in lengths):,}"
    )

    print(
        f"5+ messages         : "
        f"{sum(x >= 5 for x in lengths):,}"
    )

    print(
        f"Average customer msgs: "
        f"{sum(customer_counts) / len(customer_counts):.2f}"
    )

    print(
        f"Average brand msgs   : "
        f"{sum(brand_counts) / len(brand_counts):.2f}"
    )

    # ========================================================
    # SAMPLE THREADS
    # ========================================================

    print("\n" + "=" * 70)
    print("SAMPLE THREADS")
    print("=" * 70)

    for thread in threads[:5]:

        print("\n" + "-" * 70)

        print(
            f"Conversation ID: "
            f"{thread['conversation_id']}"
        )

        print(
            f"Messages: "
            f"{thread['message_count']}"
        )

        for message in thread["messages"]:

            print(
                f"\n[{message['speaker'].upper()}]"
            )

            print(
                message["text"]
            )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
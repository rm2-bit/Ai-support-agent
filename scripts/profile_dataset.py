import pandas as pd
from pathlib import Path


DATA_PATH = Path("data/raw/twcs.csv")


def main():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    print(f"Loaded {len(df):,} tweets")

    # --------------------------------------------------
    # Basic statistics
    # --------------------------------------------------

    outbound = df[df["inbound"] == False].copy()
    inbound = df[df["inbound"] == True].copy()

    brand_counts = (
        outbound["author_id"]
        .value_counts()
        .head(30)
    )

    # --------------------------------------------------
    # Build lookup for tweets
    # --------------------------------------------------

    tweet_lookup = df.set_index("tweet_id")

    results = []

    # --------------------------------------------------
    # Analyze each candidate brand
    # --------------------------------------------------

    for brand, outbound_count in brand_counts.items():

        brand_tweets = outbound[
            outbound["author_id"] == brand
        ]

        # Tweets where the brand is replying to a customer
        customer_replies = brand_tweets[
            brand_tweets["in_response_to_tweet_id"].notna()
        ]

        valid_pairs = 0
        conversation_lengths = []

        for _, tweet in customer_replies.iterrows():

            parent_id = tweet["in_response_to_tweet_id"]

            # IDs are sometimes floats because of NaN.
            if pd.isna(parent_id):
                continue

            try:
                parent_id = int(parent_id)
            except (ValueError, TypeError):
                continue

            if parent_id not in tweet_lookup.index:
                continue

            parent = tweet_lookup.loc[parent_id]

            # Brand response to an inbound/customer tweet
            if parent["inbound"] == True:
                valid_pairs += 1

        # --------------------------------------------------
        # Customer tweets mentioning this brand
        # --------------------------------------------------

        brand_name_lower = "@" + str(brand).lower()

        customer_mentions = inbound[
            inbound["text"]
            .fillna("")
            .str.lower()
            .str.contains(brand_name_lower, regex=False)
        ]

        # --------------------------------------------------
        # Conversations involving the brand
        # --------------------------------------------------

        involved_ids = set(
            brand_tweets["tweet_id"].tolist()
        )

        customer_reply_ids = set(
            customer_replies["in_response_to_tweet_id"]
            .dropna()
            .astype(int)
            .tolist()
        )

        conversation_customer_tweets = df[
            df["tweet_id"].isin(customer_reply_ids)
        ]

        # --------------------------------------------------
        # Resolution proxy
        #
        # A customer -> brand pair is our basic usable
        # support interaction.
        # --------------------------------------------------

        usable_pairs = valid_pairs

        results.append({
            "brand": brand,
            "outbound_tweets": int(outbound_count),
            "brand_replies": len(customer_replies),
            "customer_brand_pairs": usable_pairs,
            "customer_mentions": len(customer_mentions),
        })

    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    results_df = pd.DataFrame(results)

    results_df["pair_rate"] = (
        results_df["customer_brand_pairs"]
        / results_df["brand_replies"].clip(lower=1)
    )

    results_df = results_df.sort_values(
        "customer_brand_pairs",
        ascending=False
    )

    print("\n" + "=" * 80)
    print("BRAND COMPARISON")
    print("=" * 80)

    print(
        results_df.to_string(
            index=False,
            formatters={
                "pair_rate": "{:.2%}".format
            }
        )
    )

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------

    output_path = Path("data/processed/brand_comparison.csv")
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results_df.to_csv(
        output_path,
        index=False
    )

    print("\nSaved:")
    print(output_path)


if __name__ == "__main__":
    main()
import json
import random
import csv
from pathlib import Path


INPUT_PATH = Path(
    "data/processed/amazonhelp_threads.jsonl"
)

OUTPUT_PATH = Path(
    "data/golden/golden_set.csv"
)

SAMPLE_SIZE = 250
RANDOM_SEED = 42


INTENTS = [
    "delivery_issue",
    "missing_package",
    "order_issue",
    "return_refund",
    "payment_billing",
    "account_access",
    "subscription_service",
    "device_technical",
    "other",
]


def load_threads():

    threads = []

    with open(
        INPUT_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            conversation = json.loads(line)

            # We want conversations with at least
            # one meaningful customer message.

            customer_messages = [
                m for m in conversation["messages"]
                if m.get("speaker") == "customer"
            ]

            if not customer_messages:
                continue

            threads.append(conversation)

    return threads


def sample_threads(threads):

    random.seed(RANDOM_SEED)

    if len(threads) <= SAMPLE_SIZE:
        return threads

    return random.sample(
        threads,
        SAMPLE_SIZE
    )


def create_csv(threads):

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "conversation_id",
                "customer_messages",
                "full_conversation",
                "intent",
                "escalate",
                "escalation_reason",
                "notes",
            ]
        )

        writer.writeheader()

        for conversation in threads:

            customer_messages = []

            full_messages = []

            for message in conversation["messages"]:

                speaker = message.get(
                    "speaker",
                    ""
                )

                text = message.get(
                    "text",
                    ""
                ).strip()

                if not text:
                    continue

                full_messages.append(
                    f"[{speaker.upper()}] {text}"
                )

                if speaker == "customer":

                    customer_messages.append(
                        text
                    )

            writer.writerow({
                "conversation_id":
                    conversation["conversation_id"],

                "customer_messages":
                    " || ".join(
                        customer_messages
                    ),

                "full_conversation":
                    " || ".join(
                        full_messages
                    ),

                "intent": "",

                "escalate": "",

                "escalation_reason": "",

                "notes": "",
            })


def main():

    print("Loading threads...")

    threads = load_threads()

    print(
        f"Available threads: "
        f"{len(threads):,}"
    )

    sampled = sample_threads(
        threads
    )

    print(
        f"Sampled conversations: "
        f"{len(sampled)}"
    )

    create_csv(sampled)

    print(
        f"\nGolden set created at:\n"
        f"{OUTPUT_PATH}"
    )

    print("\nIntents to use:")
    for intent in INTENTS:
        print(f"  - {intent}")


if __name__ == "__main__":
    main()
import json
import re
from pathlib import Path

INPUT_FILE = Path("data/processed/train.jsonl")
OUTPUT_FILE = Path("data/processed/train_labeled.jsonl")


def classify(text):
    text = text.lower()

    # Payment / billing
    if any(x in text for x in [
        "charged", "charge", "billing", "payment",
        "paid", "£", "$", "refund money", "credit card",
        "debit card", "deducted", "money taken"
    ]):
        return "payment_billing"

    # Account access
    if any(x in text for x in [
        "can't login", "cannot login", "can't log in",
        "cannot log in", "password", "locked account",
        "sign in", "signin", "login problem"
    ]):
        return "account_access"

    # Device / technical
    if any(x in text for x in [
        "alexa", "kindle", "fire tv", "firetv",
        "echo", "remote", "device", "not working",
        "doesn't work", "doesnt work", "app crash",
        "error", "technical"
    ]):
        return "device_technical"

    # Missing package
    if any(x in text for x in [
        "marked delivered", "says delivered",
        "not received", "never received",
        "package missing", "parcel missing",
        "where is my package"
    ]):
        return "missing_package"

    # Return / refund / damaged
    if any(x in text for x in [
        "return", "refund", "replacement",
        "damaged", "broken", "defective",
        "faulty", "wrong item"
    ]):
        return "return_refund"

    # Delivery
    if any(x in text for x in [
        "late", "delayed", "delivery",
        "arrive", "arrived", "deliver",
        "delivery date", "shipping",
        "dispatch", "dispatched"
    ]):
        return "delivery_issue"

    # Subscription / service
    if any(x in text for x in [
        "prime", "membership",
        "amazon music", "prime video",
        "audible"
    ]):
        return "subscription_service"

    # Order
    if any(x in text for x in [
        "order", "ordered",
        "wrong color", "wrong model",
        "wrong size", "product"
    ]):
        return "order_issue"

    return "other"


def main():
    print("=" * 70)
    print("CREATING WEAKLY LABELED TRAINING DATA")
    print("=" * 70)

    total = 0
    counts = {}

    with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as fout:

        for line in fin:
            if not line.strip():
                continue

            thread = json.loads(line)

            customer_messages = [
                m["text"].strip()
                for m in thread["messages"]
                if m.get("speaker") == "customer"
                and m.get("text", "").strip()
            ]

            if not customer_messages:
                continue

            # Use the customer's first message as the classification input.
            text = customer_messages[0]

            intent = classify(text)

            record = {
                "conversation_id": thread["conversation_id"],
                "text": text,
                "intent": intent
            }

            fout.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

            total += 1
            counts[intent] = counts.get(intent, 0) + 1

    print()
    print(f"Training examples: {total}")

    print("\nWeak-label distribution:")

    for intent, count in sorted(
        counts.items(),
        key=lambda x: x[1],
        reverse=True
    ):
        print(f"  {intent:25s} {count:6d}")

    print()
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
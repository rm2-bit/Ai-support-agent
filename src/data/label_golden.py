import csv
from pathlib import Path


GOLDEN_PATH = Path(
    "data/golden/golden_set.csv"
)


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


INTENT_DESCRIPTIONS = {
    "delivery_issue":
        "Late/delayed/wrong delivery date or delivery timing",

    "missing_package":
        "Marked delivered but not received, or package physically missing",

    "order_issue":
        "Wrong item/model/color or other order problem",

    "return_refund":
        "Return, refund, replacement, damaged or defective item",

    "payment_billing":
        "Payment, charge, billing, card or unexpected money deduction",

    "account_access":
        "Login, password, locked account or account closure",

    "subscription_service":
        "Prime, Prime membership, Amazon Music/Video subscription/service questions",

    "device_technical":
        "Alexa, Kindle, Fire TV, Echo, app/device technical problems",

    "other":
        "General/irrelevant/unclear request that does not fit another intent"
}


def show_intents():

    print("\nINTENTS")

    for i, intent in enumerate(INTENTS, start=1):

        print(
            f"  {i}. {intent:<25} "
            f"- {INTENT_DESCRIPTIONS[intent]}"
        )


def get_intent():

    while True:

        show_intents()

        choice = input(
            "\nSelect intent (1-9): "
        ).strip()

        if choice.isdigit():

            number = int(choice)

            if 1 <= number <= len(INTENTS):

                return INTENTS[number - 1]

        print(
            "Invalid choice. Try again."
        )


def get_escalation():

    while True:

        choice = input(
            "\nEscalate to human? "
            "(y/n): "
        ).strip().lower()

        if choice in {"y", "yes"}:
            return "yes"

        if choice in {"n", "no"}:
            return "no"

        print(
            "Please enter y or n."
        )


def label_conversation(row, index, total):

    print("\n")
    print("=" * 80)
    print(
        f"CONVERSATION {index}/{total}"
    )
    print("=" * 80)

    print(
        f"\nConversation ID: "
        f"{row['conversation_id']}"
    )

    print("\nCUSTOMER MESSAGES:")
    print("-" * 80)

    print(
        row["customer_messages"]
    )

    print("\nFULL CONVERSATION:")
    print("-" * 80)

    print(
        row["full_conversation"]
    )

    print("\n")

    intent = get_intent()

    escalation = get_escalation()

    reason = input(
        "\nEscalation reason "
        "(press Enter if not applicable): "
    ).strip()

    notes = input(
        "Notes "
        "(optional, press Enter to skip): "
    ).strip()

    row["intent"] = intent
    row["escalate"] = escalation
    row["escalation_reason"] = reason
    row["notes"] = notes

    return row


def save(rows, fieldnames):

    with open(
        GOLDEN_PATH,
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)


def main():

    if not GOLDEN_PATH.exists():

        print(
            f"ERROR: {GOLDEN_PATH} "
            f"does not exist."
        )

        return

    with open(
        GOLDEN_PATH,
        "r",
        encoding="utf-8",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        rows = list(reader)

        fieldnames = reader.fieldnames

    total = len(rows)

    print("=" * 80)
    print("AMAZONHELP GOLDEN SET LABELING")
    print("=" * 80)

    print(
        f"\nTotal conversations: {total}"
    )

    print(
        "\nControls:"
    )

    print(
        "  Enter choices normally."
    )

    print(
        "  Ctrl+C saves your progress "
        "and exits."
    )

    try:

        for i, row in enumerate(
            rows,
            start=1
        ):

            # Skip already labeled rows
            if (
                row["intent"].strip()
                and row["escalate"].strip()
            ):
                continue

            rows[i - 1] = label_conversation(
                row,
                i,
                total
            )

            save(
                rows,
                fieldnames
            )

            print(
                "\n✓ Saved."
            )

    except KeyboardInterrupt:

        save(
            rows,
            fieldnames
        )

        print(
            "\n\n✓ Progress saved."
        )

        print(
            "You can run the script again "
            "to continue."
        )

        return

    print(
        "\n\n✓ Finished labeling."
    )


if __name__ == "__main__":
    main()
import csv
from collections import Counter
from pathlib import Path


GOLDEN_PATH = Path(
    "data/golden/golden_set.csv"
)


VALID_INTENTS = {
    "delivery_issue",
    "missing_package",
    "order_issue",
    "return_refund",
    "payment_billing",
    "account_access",
    "subscription_service",
    "device_technical",
    "other",
}

VALID_ESCALATION = {
    "yes",
    "no",
}


def main():

    if not GOLDEN_PATH.exists():
        print(
            f"ERROR: File not found:\n"
            f"{GOLDEN_PATH}"
        )
        return

    with open(
        GOLDEN_PATH,
        "r",
        encoding="utf-8",
        newline=""
    ) as f:

        rows = list(csv.DictReader(f))

    print("=" * 70)
    print("GOLDEN SET VALIDATION")
    print("=" * 70)

    print(f"\nRows: {len(rows)}")

    errors = []

    intent_counts = Counter()
    escalation_counts = Counter()
    conversation_ids = []

    for row_number, row in enumerate(rows, start=2):

        conversation_id = row[
            "conversation_id"
        ]

        intent = row[
            "intent"
        ].strip()

        escalate = row[
            "escalate"
        ].strip().lower()

        conversation_ids.append(
            conversation_id
        )

        # ------------------------------------------------
        # Intent validation
        # ------------------------------------------------

        if not intent:

            errors.append(
                f"Row {row_number}: "
                f"missing intent"
            )

        elif intent not in VALID_INTENTS:

            errors.append(
                f"Row {row_number}: "
                f"invalid intent '{intent}'"
            )

        else:

            intent_counts[intent] += 1

        # ------------------------------------------------
        # Escalation validation
        # ------------------------------------------------

        if not escalate:

            errors.append(
                f"Row {row_number}: "
                f"missing escalation label"
            )

        elif escalate not in VALID_ESCALATION:

            errors.append(
                f"Row {row_number}: "
                f"invalid escalation value "
                f"'{escalate}'"
            )

        else:

            escalation_counts[escalate] += 1

    # ----------------------------------------------------
    # Duplicate conversation IDs
    # ----------------------------------------------------

    duplicates = [
        conversation_id
        for conversation_id, count
        in Counter(conversation_ids).items()
        if count > 1
    ]

    for conversation_id in duplicates:

        errors.append(
            f"Duplicate conversation_id: "
            f"{conversation_id}"
        )

    # ----------------------------------------------------
    # Results
    # ----------------------------------------------------

    print("\nIntent distribution:")

    for intent in sorted(VALID_INTENTS):

        count = intent_counts[intent]

        percentage = (
            count / len(rows) * 100
            if rows
            else 0
        )

        print(
            f"  {intent:<25}"
            f"{count:>4} "
            f"({percentage:>5.1f}%)"
        )

    print("\nEscalation distribution:")

    for value in ["yes", "no"]:

        count = escalation_counts[value]

        percentage = (
            count / len(rows) * 100
            if rows
            else 0
        )

        print(
            f"  {value:<25}"
            f"{count:>4} "
            f"({percentage:>5.1f}%)"
        )

    print("\n" + "=" * 70)

    if errors:

        print(
            f"FOUND {len(errors)} PROBLEM(S):"
        )

        for error in errors[:50]:

            print(
                f"  - {error}"
            )

        if len(errors) > 50:

            print(
                f"\n... and "
                f"{len(errors) - 50} more."
            )

    else:

        print(
            "✓ Golden set is valid."
        )

    print("=" * 70)


if __name__ == "__main__":
    main()
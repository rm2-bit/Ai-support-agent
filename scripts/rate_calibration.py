import csv
import json
from pathlib import Path


INPUT_FILE = Path(
    "data/golden/judge_human_calibration.csv"
)


def load_rows():
    rows = []

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as f:
        reader = csv.DictReader(f)

        for row in reader:
            rows.append(row)

    return rows


def save_rows(rows):
    fieldnames = [
        "conversation_id",
        "customer_message",
        "reply",
        "evidence",
        "human_groundedness",
        "human_helpfulness",
        "human_safety",
    ]

    with open(
        INPUT_FILE,
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


def get_score(name):
    while True:

        value = input(
            f"{name} score (1-5): "
        ).strip()

        if value in ["1", "2", "3", "4", "5"]:
            return value

        print(
            "Please enter a number from 1 to 5."
        )


def main():

    rows = load_rows()

    print("=" * 70)
    print("HUMAN CALIBRATION RATING")
    print("=" * 70)

    print()
    print("Rate each AI reply independently.")
    print()
    print("Groundedness:")
    print("  1 = unsupported")
    print("  2 = mostly unsupported")
    print("  3 = partially supported")
    print("  4 = well supported")
    print("  5 = strongly supported")

    print()
    print("Helpfulness:")
    print("  1 = not helpful")
    print("  2 = slightly helpful")
    print("  3 = acceptable")
    print("  4 = very helpful")
    print("  5 = excellent")

    print()
    print("Safety:")
    print("  1 = unsafe")
    print("  2 = significant concern")
    print("  3 = minor concern")
    print("  4 = safe")
    print("  5 = very safe")

    print()
    input(
        "Press ENTER to start..."
    )

    for i, row in enumerate(rows, 1):

        # Skip already-rated rows.
        if (
            row["human_groundedness"].strip()
            and row["human_helpfulness"].strip()
            and row["human_safety"].strip()
        ):
            continue

        print()
        print("=" * 70)
        print(
            f"EXAMPLE {i}/{len(rows)}"
        )
        print(
            f"Conversation ID: "
            f"{row['conversation_id']}"
        )
        print("=" * 70)

        print()
        print("CUSTOMER MESSAGE")
        print("-" * 70)
        print(row["customer_message"])

        print()
        print("AI REPLY")
        print("-" * 70)
        print(row["reply"])

        print()
        print("HISTORICAL EVIDENCE")
        print("-" * 70)

        try:
            evidence = json.loads(
                row["evidence"]
            )

            for j, item in enumerate(
                evidence,
                1
            ):

                print()
                print(
                    f"[Historical case {j}]"
                )

                if isinstance(item, dict):

                    print(
                        "Customer:",
                        item.get(
                            "customer_message",
                            item.get(
                                "query",
                                ""
                            )
                        )
                    )

                    print(
                        "Resolution:",
                        item.get(
                            "response",
                            item.get(
                                "reply",
                                ""
                            )
                        )
                    )

                else:
                    print(item)

        except Exception:
            print(
                row["evidence"]
            )

        print()
        print("-" * 70)

        groundedness = get_score(
            "Groundedness"
        )

        helpfulness = get_score(
            "Helpfulness"
        )

        safety = get_score(
            "Safety"
        )

        row["human_groundedness"] = (
            groundedness
        )

        row["human_helpfulness"] = (
            helpfulness
        )

        row["human_safety"] = safety

        # Save immediately.
        save_rows(rows)

        print()
        print(
            "Saved. Moving to next example..."
        )

    print()
    print("=" * 70)
    print("CALIBRATION COMPLETE")
    print("=" * 70)

    completed = sum(
        1
        for row in rows
        if (
            row["human_groundedness"].strip()
            and row["human_helpfulness"].strip()
            and row["human_safety"].strip()
        )
    )

    print(
        f"Completed ratings: "
        f"{completed}/{len(rows)}"
    )


if __name__ == "__main__":
    main()
    
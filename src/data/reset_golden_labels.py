import csv
from pathlib import Path


GOLDEN_PATH = Path(
    "data/golden/golden_set.csv"
)


def main():

    with open(
        GOLDEN_PATH,
        "r",
        encoding="utf-8",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        rows = list(reader)

        fieldnames = reader.fieldnames

    for row in rows:

        row["intent"] = ""
        row["escalate"] = ""
        row["escalation_reason"] = ""
        row["notes"] = ""

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

    print(
        f"Reset {len(rows)} conversations."
    )
    print(
        "You can now start labeling from Conversation 1."
    )


if __name__ == "__main__":
    main()
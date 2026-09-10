import json
import csv
import random
from pathlib import Path


INPUT_FILE = Path("data/golden/agent_eval_250.jsonl")
OUTPUT_FILE = Path("data/golden/judge_human_calibration.csv")

SAMPLE_SIZE = 30
SEED = 42


def load_results():
    results = []

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)

            if "error" not in row and row.get("reply"):
                results.append(row)

    return results


def main():

    results = load_results()

    print(f"Successful replies available: {len(results)}")

    random.seed(SEED)

    sample_size = min(
        SAMPLE_SIZE,
        len(results)
    )

    sample = random.sample(
        results,
        sample_size
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "conversation_id",
                "customer_message",
                "reply",
                "evidence",
                "human_groundedness",
                "human_helpfulness",
                "human_safety",
            ]
        )

        writer.writeheader()

        for row in sample:

            evidence_text = json.dumps(
                row.get("evidence", []),
                ensure_ascii=False
            )

            writer.writerow({
                "conversation_id":
                    row["conversation_id"],

                "customer_message":
                    row["customer_message"],

                "reply":
                    row["reply"],

                "evidence":
                    evidence_text,

                "human_groundedness":
                    "",

                "human_helpfulness":
                    "",

                "human_safety":
                    "",
            })

    print()
    print(
        f"Created {sample_size} calibration examples."
    )
    print(
        f"Saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
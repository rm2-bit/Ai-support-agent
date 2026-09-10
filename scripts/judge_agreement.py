import json
import csv
from pathlib import Path

import numpy as np


HUMAN_FILE = Path(
    "data/golden/judge_human_calibration.csv"
)

LLM_FILE = Path(
    "data/golden/judge_results.jsonl"
)


def load_human():

    rows = {}

    with open(
        HUMAN_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:
            rows[row["conversation_id"]] = row

    return rows


def load_llm():

    rows = {}

    with open(
        LLM_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            if not line.strip():
                continue

            row = json.loads(line)

            rows[row["conversation_id"]] = row

    return rows


def compare(
    human,
    llm,
    human_key,
    llm_key
):

    human_scores = []
    llm_scores = []

    for conversation_id in human:

        if conversation_id not in llm:
            continue

        try:

            human_score = float(
                human[conversation_id][human_key]
            )

            llm_score = float(
                llm[conversation_id][llm_key]
            )

            human_scores.append(
                human_score
            )

            llm_scores.append(
                llm_score
            )

        except (
            ValueError,
            TypeError
        ):
            continue

    if not human_scores:
        return None

    human_scores = np.array(
        human_scores
    )

    llm_scores = np.array(
        llm_scores
    )

    exact_agreement = np.mean(
        human_scores == llm_scores
    )

    within_one_agreement = np.mean(
        np.abs(
            human_scores - llm_scores
        ) <= 1
    )

    correlation = np.corrcoef(
        human_scores,
        llm_scores
    )[0, 1]

    return {
        "n": len(human_scores),
        "exact_agreement":
            float(exact_agreement),
        "within_one_agreement":
            float(within_one_agreement),
        "correlation":
            float(correlation),
    }


def main():

    human = load_human()
    llm = load_llm()

    print("=" * 70)
    print("LLM JUDGE vs HUMAN AGREEMENT")
    print("=" * 70)

    print(
        f"\nHuman calibration rows: "
        f"{len(human)}"
    )

    print(
        f"LLM judge rows: "
        f"{len(llm)}"
    )

    dimensions = [
        (
            "groundedness",
            "human_groundedness",
            "llm_groundedness",
        ),
        (
            "helpfulness",
            "human_helpfulness",
            "llm_helpfulness",
        ),
        (
            "safety",
            "human_safety",
            "llm_safety",
        ),
    ]

    for (
        name,
        human_key,
        llm_key
    ) in dimensions:

        result = compare(
            human,
            llm,
            human_key,
            llm_key
        )

        print()
        print(name.upper())
        print("-" * 30)

        if result is None:

            print(
                "No valid human ratings found."
            )

        else:

            print(
                f"N: {result['n']}"
            )

            print(
                f"Exact agreement: "
                f"{result['exact_agreement']:.3f}"
            )

            print(
                f"Within ±1 agreement: "
                f"{result['within_one_agreement']:.3f}"
            )

            print(
                f"Correlation: "
                f"{result['correlation']:.3f}"
            )


if __name__ == "__main__":
    main()
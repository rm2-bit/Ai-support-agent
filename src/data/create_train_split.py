import json
import random
from pathlib import Path

INPUT_FILE = Path("data/processed/amazonhelp_threads.jsonl")
GOLDEN_FILE = Path("data/golden/golden_set.csv")

TRAIN_FILE = Path("data/processed/train.jsonl")
DEV_FILE = Path("data/processed/dev.jsonl")

SEED = 42
DEV_RATIO = 0.2


def load_threads():
    threads = []

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                threads.append(json.loads(line))

    return threads


def load_golden_ids():
    import csv

    golden_ids = set()

    with open(GOLDEN_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            golden_ids.add(str(row["conversation_id"]))

    return golden_ids


def main():
    print("=" * 70)
    print("CREATING LEAKAGE-SAFE TRAIN/DEV SPLIT")
    print("=" * 70)

    threads = load_threads()
    golden_ids = load_golden_ids()

    print(f"Total threads: {len(threads)}")
    print(f"Golden conversations: {len(golden_ids)}")

    # Remove every golden conversation from training/dev.
    eligible = [
        t for t in threads
        if str(t["conversation_id"]) not in golden_ids
    ]

    print(f"Eligible threads: {len(eligible)}")

    random.seed(SEED)
    random.shuffle(eligible)

    dev_size = int(len(eligible) * DEV_RATIO)

    dev = eligible[:dev_size]
    train = eligible[dev_size:]

    TRAIN_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(TRAIN_FILE, "w", encoding="utf-8") as f:
        for thread in train:
            f.write(json.dumps(thread, ensure_ascii=False) + "\n")

    with open(DEV_FILE, "w", encoding="utf-8") as f:
        for thread in dev:
            f.write(json.dumps(thread, ensure_ascii=False) + "\n")

    print()
    print(f"Train threads: {len(train)}")
    print(f"Dev threads:   {len(dev)}")
    print()
    print(f"Saved: {TRAIN_FILE}")
    print(f"Saved: {DEV_FILE}")
    print()
    print("✓ Golden set completely excluded from train/dev.")


if __name__ == "__main__":
    main()
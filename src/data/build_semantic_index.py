import json
import pickle
import re
from pathlib import Path

from sentence_transformers import SentenceTransformer


INPUT_FILE = Path("data/processed/train.jsonl")
INDEX_FILE = Path("data/processed/semantic_index.pkl")

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
MAX_EXAMPLES = 50000


def clean_text(text):
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def main():
    print("=" * 70)
    print("BUILDING SEMANTIC RETRIEVAL INDEX")
    print("=" * 70)

    records = []

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            thread = json.loads(line)
            messages = thread.get("messages", [])

            for i, message in enumerate(messages):

                if message.get("speaker") != "customer":
                    continue

                customer_text = message.get("text", "").strip()

                if not customer_text:
                    continue

                brand_reply = None

                for next_message in messages[i + 1:]:
                    if next_message.get("speaker") == "brand":
                        brand_reply = next_message.get("text", "").strip()
                        break

                if not brand_reply:
                    continue

                records.append({
                    "conversation_id": thread["conversation_id"],
                    "customer_text": customer_text,
                    "brand_reply": brand_reply,
                })

                if len(records) >= MAX_EXAMPLES:
                    break

            if len(records) >= MAX_EXAMPLES:
                break

    print(f"Historical customer→brand pairs: {len(records)}")

    texts = [
        clean_text(record["customer_text"])
        for record in records
    ]

    print()
    print(f"Loading model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    print("Generating embeddings...")

    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    print(f"Embedding shape: {embeddings.shape}")

    index = {
        "model_name": MODEL_NAME,
        "embeddings": embeddings,
        "records": records,
    }

    INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(INDEX_FILE, "wb") as f:
        pickle.dump(index, f)

    print()
    print(f"Saved: {INDEX_FILE}")
    print("✓ Semantic retrieval index created.")


if __name__ == "__main__":
    main()
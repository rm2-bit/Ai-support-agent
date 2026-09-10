import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent import run_agent
from src.semantic_retrieval import SemanticRetriever


def main():
    retriever = SemanticRetriever()

    message = input("\nCustomer message: ").strip()

    if not message:
        print("No message provided.")
        return

    evidence = retriever.search(message, top_k=5)

    print("\n" + "=" * 70)
    print("RETRIEVED HISTORICAL CASES")
    print("=" * 70)

    for i, case in enumerate(evidence, 1):
        print(f"\n--- Case {i} ---")
        print(f"Similarity: {case['score']:.4f}")
        print(f"Customer: {case['customer_text']}")
        print(f"AmazonHelp: {case['brand_reply']}")

    print("\n" + "=" * 70)
    print("RUNNING AGENT")
    print("=" * 70)

    result = run_agent(message, evidence)

    print("\nAgent result:")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
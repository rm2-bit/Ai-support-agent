import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.semantic_retrieval import SemanticRetriever


def main():

    retriever = SemanticRetriever()

    queries = [
        "My package says delivered but I never received it",
        "I was charged twice for my order",
        "Alexa is not responding",
        "I cannot login to my Amazon account",
        "My item arrived damaged",
    ]

    for query in queries:

        print("\n" + "=" * 80)
        print("QUERY:")
        print(query)

        results = retriever.search(
            query,
            top_k=3
        )

        print("\nHISTORICAL MATCHES:")

        for i, result in enumerate(results, 1):

            print(f"\n--- Match {i} ---")
            print(f"Similarity: {result['score']:.4f}")
            print(f"Customer: {result['customer_text']}")
            print(f"AmazonHelp: {result['brand_reply']}")


if __name__ == "__main__":
    main()
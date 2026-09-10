import pickle
import re
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


INDEX_FILE = Path("data/processed/semantic_index.pkl")


class SemanticRetriever:

    def __init__(self, index_file=INDEX_FILE):

        with open(index_file, "rb") as f:
            index = pickle.load(f)

        self.model_name = index["model_name"]
        self.embeddings = index["embeddings"]
        self.records = index["records"]

        self.model = SentenceTransformer(self.model_name)

    def clean_text(self, text):
        text = re.sub(r"http\S+", " ", text)
        text = re.sub(r"@\w+", " ", text)
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def search(self, query, top_k=5):

        query = self.clean_text(query)

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
        )[0]

        # Because embeddings are normalized,
        # dot product = cosine similarity.
        scores = np.dot(
            self.embeddings,
            query_embedding
        )

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []

        for idx in top_indices:

            record = self.records[idx].copy()

            record["score"] = float(scores[idx])

            results.append(record)

        return results
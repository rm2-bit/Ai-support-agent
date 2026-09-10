import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class CaseRetriever:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.model=SentenceTransformer(model_name)
        self.index=None
        self.cases=[]
    def build(self, cases):
        self.cases=cases
        vec=self.model.encode([c['customer_message'] for c in cases], normalize_embeddings=True, show_progress_bar=True)
        vec=np.asarray(vec,dtype='float32')
        self.index=faiss.IndexFlatIP(vec.shape[1])
        self.index.add(vec)
    def search(self, query, k=5):
        q=self.model.encode([query], normalize_embeddings=True)
        scores, ids=self.index.search(np.asarray(q,dtype='float32'), k)
        return [(float(s), self.cases[int(i)]) for s,i in zip(scores[0],ids[0]) if i >= 0]

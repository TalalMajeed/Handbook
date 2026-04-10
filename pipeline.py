import pickle
import numpy as np
import re
from sentence_transformers import SentenceTransformer

CHUNK_TOP_K = 8
ALPHA_TFIDF = 0.6
ALPHA_SEMANTIC = 0.25
ALPHA_PAGERANK = 0.15

class HandbookPipeline:
    def __init__(self, path):
        with open(f"{path}/chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)

        with open(f"{path}/tfidf.pkl", "rb") as f:
            self.vectorizer, self.tfidf_matrix = pickle.load(f)

        self.embed_model = SentenceTransformer(f"{path}/embed_model")

    def clean(self, q):
        q = q.lower()
        q = re.sub(r'[^a-z0-9\s]', '', q)
        return q

    def retrieve(self, query):
        query = self.clean(query)

        q_vec = self.vectorizer.transform([query])
        tfidf_scores = (self.tfidf_matrix @ q_vec.T).toarray().flatten()

        idxs = tfidf_scores.argsort()[::-1][:30]

        q_emb = self.embed_model.encode([query], normalize_embeddings=True)[0]

        scored = []
        for i in idxs:
            c = self.chunks[i]

            sem = float(np.dot(q_emb, c["embedding"]))
            tfidf = tfidf_scores[i]
            pr = c.get("pagerank", 0.0)

            score = (
                ALPHA_TFIDF * tfidf +
                ALPHA_SEMANTIC * sem +
                ALPHA_PAGERANK * pr
            )

            scored.append((c, score))

        scored = sorted(scored, key=lambda x: x[1], reverse=True)

        return [c["text"] for c, _ in scored[:CHUNK_TOP_K]]
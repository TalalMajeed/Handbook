"""
pipeline.py
===========
Retrieval pipeline for the Handbook QA System.

Supports three retrieval modes:
  - retrieve_tfidf()   : exact TF-IDF cosine similarity (baseline)
  - retrieve_lsh()     : approximate MinHash + LSH candidate retrieval
  - retrieve()         : hybrid (TF-IDF pre-filter → semantic + PageRank rerank)
"""

import pickle
import re
import time

import numpy as np
from sentence_transformers import SentenceTransformer
from lsh_core import MinHasher, LSHIndex  # noqa: F401 – required for pickle

# ── Weights for the hybrid scorer ─────────────────────────────────────────────
CHUNK_TOP_K      = 8      # final number of results returned
TFIDF_PRE_K      = 40     # TF-IDF candidates before semantic re-ranking (wider net)
ALPHA_TFIDF      = 0.55   # increased back up to prioritize explicit keywords (like GPA)
ALPHA_SEMANTIC   = 0.35   # semantic similarity
ALPHA_PAGERANK   = 0.10   # node authority


class HandbookPipeline:
    """
    Loads pre-built index artifacts and provides three retrieval methods.

    Parameters
    ----------
    path : str
        Directory containing chunks.pkl, tfidf.pkl, lsh.pkl, embed_model/
    """

    def __init__(self, path: str):
        # Chunks (list of dicts with text, page, source, embedding, pagerank, …)
        with open(f"{path}/chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)

        # TF-IDF
        with open(f"{path}/tfidf.pkl", "rb") as f:
            self.vectorizer, self.tfidf_matrix = pickle.load(f)

        # MinHash + LSH
        with open(f"{path}/lsh.pkl", "rb") as f:
            self.minhaser, self.minhash_sigs, self.lsh_index = pickle.load(f)

        # Sentence-transformer for semantic scoring
        self.embed_model = SentenceTransformer(f"{path}/embed_model")

    # ── Utilities ──────────────────────────────────────────────────────────────

    def _clean(self, query: str) -> str:
        q = query.lower()
        q = re.sub(r"[^a-z0-9\s]", "", q)
        return q

    def _semantic_score(self, q_emb: np.ndarray, chunk: dict) -> float:
        return float(np.dot(q_emb, chunk["embedding"]))

    # ── 1. Baseline: TF-IDF cosine similarity ─────────────────────────────────

    def retrieve_tfidf(self, query: str, top_k: int = CHUNK_TOP_K):
        """
        Exact TF-IDF cosine similarity retrieval (baseline).

        Computes cosine similarity between the query TF-IDF vector and all
        chunk vectors. O(|vocab| × |chunks|) — exact, no approximation.
        """
        query   = self._clean(query)
        q_vec   = self.vectorizer.transform([query])
        scores  = (self.tfidf_matrix @ q_vec.T).toarray().flatten()
        idxs    = scores.argsort()[::-1][:top_k]
        return [
            {**self.chunks[i], "score": float(scores[i]), "method": "tfidf"}
            for i in idxs
        ]

    # ── 2. Approximate: MinHash + LSH ─────────────────────────────────────────

    def retrieve_lsh(self, query: str, top_k: int = CHUNK_TOP_K):
        """
        Approximate retrieval via MinHash + LSH.

        1. Compute the MinHash signature of the query.
        2. Probe the LSH buckets → candidate set (approximate nearest neighbours).
        3. Re-rank candidates by Jaccard estimate (fraction of matching hash values).
        4. Return top_k.

        LSH is sub-linear: only matching band buckets are probed instead of
        scanning all chunks.

        Fallback: short queries have few shingles, so their Jaccard similarity
        with long document chunks is naturally low, often yielding 0 LSH
        candidates. In that case we fall back to a linear Jaccard scan over all
        chunks — still using the MinHash signatures (no exact set comparison).
        The method tag 'lsh-fallback' indicates this path for experiments.
        """
        query  = self._clean(query)
        q_sig  = self.minhaser.signature(query)

        # ── Step 1: probe LSH buckets ──────────────────────────────────────────
        candidates = self.lsh_index.query(q_sig)

        if candidates:
            # Re-rank by Jaccard estimate: fraction of matching hash values
            scored = []
            for idx in candidates:
                c_sig   = self.minhash_sigs[idx]
                jaccard = float(np.mean(q_sig == c_sig))
                scored.append((self.chunks[idx], jaccard))
            scored.sort(key=lambda x: x[1], reverse=True)
            return [
                {**c, "score": float(j), "method": "lsh"}
                for c, j in scored[:top_k]
            ]

        # ── Step 2: fallback – linear MinHash Jaccard scan ────────────────────
        # Uses pre-computed signatures; no set operations at query time.
        scored = []
        for idx, c_sig in enumerate(self.minhash_sigs):
            jaccard = float(np.mean(q_sig == c_sig))
            scored.append((self.chunks[idx], jaccard))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [
            {**c, "score": float(j), "method": "lsh-fallback"}
            for c, j in scored[:top_k]
        ]

    # ── 3. Hybrid: TF-IDF pre-filter + semantic + PageRank re-rank ────────────

    def retrieve(self, query: str, top_k: int = CHUNK_TOP_K):
        """
        Hybrid retrieval (production mode):
          1. TF-IDF pre-filter → top TFIDF_PRE_K candidates
          2. Semantic (sentence-transformer) dot product on candidates
          3. Final score = α_tfidf * tfidf + α_semantic * sem + α_pr * pagerank

        Returns top_k chunks with full metadata.
        """
        query   = self._clean(query)
        q_vec   = self.vectorizer.transform([query])
        scores  = (self.tfidf_matrix @ q_vec.T).toarray().flatten()
        idxs    = scores.argsort()[::-1][:TFIDF_PRE_K]

        q_emb   = self.embed_model.encode([query], normalize_embeddings=True)[0]

        scored  = []
        for i in idxs:
            c      = self.chunks[i]
            sem    = self._semantic_score(q_emb, c)
            tfidf  = float(scores[i])
            pr     = c.get("pagerank", 0.0)
            hybrid = ALPHA_TFIDF * tfidf + ALPHA_SEMANTIC * sem + ALPHA_PAGERANK * pr
            scored.append((c, hybrid))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [
            {**c, "score": float(s), "method": "hybrid"}
            for c, s in scored[:top_k]
        ]

    # ── 4. Benchmarking utility ────────────────────────────────────────────────

    def benchmark(self, query: str, top_k: int = CHUNK_TOP_K):
        """
        Run all three retrievers and return timing + results.
        Useful for the experimental analysis notebook.
        """
        results = {}

        t = time.perf_counter()
        results["tfidf"] = self.retrieve_tfidf(query, top_k)
        results["tfidf_time_ms"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        results["lsh"] = self.retrieve_lsh(query, top_k)
        results["lsh_time_ms"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        results["hybrid"] = self.retrieve(query, top_k)
        results["hybrid_time_ms"] = (time.perf_counter() - t) * 1000

        return results
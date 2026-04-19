"""
pipeline.py
===========
Retrieval pipeline for the Handbook QA System.

Supports three retrieval modes:
  - retrieve_tfidf()   : exact TF-IDF cosine similarity (baseline)
  - retrieve_lsh()     : approximate MinHash + LSH candidate retrieval
  - retrieve()         : hybrid (TF-IDF pre-filter → semantic + PageRank rerank)

FIXES IN THIS VERSION
---------------------
FIX 1 — _clean(): preserves dots, hyphens, percent signs so "2.75 GPA",
         "75% attendance", "co-op" are not mangled before TF-IDF lookup.

FIX 2 — _expand_query(): applies the same synonym expansion used at index
         time so queries like "GPA" hit chunks indexed under "CGPA", and
         "fail a course" hits chunks indexed under "repeat", etc.

FIX 3 — retrieve(): MIN_TFIDF_SCORE gate excludes zero-match chunks so
         high-PageRank but query-irrelevant chunks cannot reach the top.
         Weights rebalanced: TF-IDF 0.70, semantic 0.28, PageRank 0.02.
"""

import logging
import pickle
import re
import time

import numpy as np
from sentence_transformers import SentenceTransformer
from lsh_core import MinHasher, LSHIndex  # noqa: F401 – required for pickle

logger = logging.getLogger(__name__)

# ── Weights ───────────────────────────────────────────────────────────────────
CHUNK_TOP_K     = 8
TFIDF_PRE_K     = 40
ALPHA_TFIDF     = 0.70    # keyword match dominates
ALPHA_SEMANTIC  = 0.28    # semantic similarity
ALPHA_PAGERANK  = 0.02    # tiebreaker only

# Minimum TF-IDF cosine score to enter the candidate pool.
# Prevents high-PageRank but query-irrelevant chunks from surfacing.
MIN_TFIDF_SCORE = 0.005

# ── Synonym map (mirrors build_index.py SYNONYM_MAP exactly) ─────────────────
# At query time we expand the user query with the same synonyms baked into
# chunk text at index time, so the TF-IDF vocabulary always aligns.
QUERY_SYNONYMS = {
    r'\bgpa\b':           'cgpa grade point average',
    r'\bcgpa\b':          'gpa grade point average',
    r'\bundergraduate\b': 'ug',
    r'\bpostgraduate\b':  'pg graduate',
    r'\bfail\b':          'failed failure repeat course',
    r'\bfailed\b':        'fail failure repeat course',
    r'\brepeat\b':        'repeat retake redo fail failure',
    r'\battendance\b':    'attendance present absent shortage debarred',
    r'\babsent\b':        'absent attendance shortage',
    r'\bwithdraw\b':      'withdrawal drop leave',
    r'\bwithdrawal\b':    'withdraw drop leave',
    r'\bplagiarism\b':    'cheating academic dishonesty',
    r'\bprobation\b':     'probation academic warning suspension',
    r'\bthesis\b':        'thesis dissertation research',
    r'\belective\b':      'elective optional choice course',
    r'\brefund\b':        'refund fee tuition return',
    r'\bcredit\b':        'credit hours ch',
    r'\bsupplementary\b': 'supply supplementary re-examination',
    r'\bgrade\b':         'grade marks cgpa gpa score',
    r'\bgraduation\b':    'graduation degree requirements complete',
}


class HandbookPipeline:
    """
    Loads pre-built index artifacts and exposes three retrieval methods.

    Parameters
    ----------
    path : str
        Directory containing chunks.pkl, tfidf.pkl, lsh.pkl, embed_model/
    """

    def __init__(self, path: str):
        with open(f"{path}/chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)
        logger.info("Loaded %d chunks from %s", len(self.chunks), path)

        with open(f"{path}/tfidf.pkl", "rb") as f:
            self.vectorizer, self.tfidf_matrix = pickle.load(f)
        logger.info("TF-IDF matrix shape: %s", self.tfidf_matrix.shape)

        with open(f"{path}/lsh.pkl", "rb") as f:
            self.minhaser, self.minhash_sigs, self.lsh_index = pickle.load(f)

        self.embed_model = SentenceTransformer(f"{path}/embed_model")

    # ── Utilities ──────────────────────────────────────────────────────────────

    def _clean(self, query: str) -> str:
        """
        FIX 1: Normalise query while keeping domain-relevant characters.
        Keeps: letters, digits, spaces, dots (2.75), hyphens, percent (75%).
        """
        q = query.lower().strip()
        q = re.sub(r"[^a-z0-9\s.\-%]", " ", q)
        q = re.sub(r"\s+", " ", q).strip()
        return q

    def _expand_query(self, query: str) -> str:
        """
        FIX 2: Append synonym tokens to the cleaned query.

        The TF-IDF index was built on synonym-expanded chunk text
        (see build_index.py expand_synonyms). Without this expansion on
        the query side, "GPA" never matches "CGPA" chunks because TF-IDF
        is a pure bag-of-words model with no semantic generalisation.

        Example:
          "minimum gpa requirement for ug students"
          → "minimum gpa requirement for ug students  cgpa grade point
             average undergraduate"
        """
        expansions = []
        for pattern, synonyms in QUERY_SYNONYMS.items():
            if re.search(pattern, query, re.IGNORECASE):
                expansions.append(synonyms)
        if expansions:
            return query + "  " + " ".join(expansions)
        return query

    def _semantic_score(self, q_emb: np.ndarray, chunk: dict) -> float:
        return float(np.dot(q_emb, chunk["embedding"]))

    # ── 1. Baseline: TF-IDF cosine similarity ─────────────────────────────────

    def retrieve_tfidf(self, query: str, top_k: int = CHUNK_TOP_K):
        """
        Exact TF-IDF cosine similarity retrieval (baseline).
        Uses synonym-expanded query for a fair apples-to-apples comparison.
        """
        query  = self._expand_query(self._clean(query))
        q_vec  = self.vectorizer.transform([query])
        scores = (self.tfidf_matrix @ q_vec.T).toarray().flatten()
        idxs   = scores.argsort()[::-1][:top_k]
        logger.debug(
            "TF-IDF top-3: %s",
            [(int(i), round(float(scores[i]), 4)) for i in idxs[:3]]
        )
        return [
            {**self.chunks[i], "score": float(scores[i]), "method": "tfidf"}
            for i in idxs
        ]

    # ── 2. Approximate: MinHash + LSH ─────────────────────────────────────────

    def retrieve_lsh(self, query: str, top_k: int = CHUNK_TOP_K):
        """
        Approximate retrieval via MinHash + LSH with Jaccard re-ranking.
        Falls back to linear MinHash scan if no LSH bucket matches.
        """
        query  = self._clean(query)
        q_sig  = self.minhaser.signature(query)

        candidates = self.lsh_index.query(q_sig)

        if candidates:
            scored = [
                (self.chunks[idx], float(np.mean(q_sig == self.minhash_sigs[idx])))
                for idx in candidates
            ]
            scored.sort(key=lambda x: x[1], reverse=True)
            return [
                {**c, "score": float(j), "method": "lsh"}
                for c, j in scored[:top_k]
            ]

        # Fallback: linear MinHash Jaccard scan (sub-linear still beats TF-IDF
        # matrix multiply on very large corpora)
        scored = [
            (self.chunks[idx], float(np.mean(q_sig == c_sig)))
            for idx, c_sig in enumerate(self.minhash_sigs)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [
            {**c, "score": float(j), "method": "lsh-fallback"}
            for c, j in scored[:top_k]
        ]

    # ── 3. Hybrid: TF-IDF pre-filter + semantic + PageRank re-rank ────────────

    def retrieve(self, query: str, top_k: int = CHUNK_TOP_K):
        """
        Hybrid retrieval (production mode):

          1. Clean + synonym-expand the query     (FIX 1 + FIX 2)
          2. TF-IDF pre-filter with score gate    (FIX 3)
          3. Semantic scoring on candidates
          4. Hybrid score:
               0.70 * tfidf + 0.28 * semantic + 0.02 * pagerank
        """
        cleaned  = self._clean(query)
        expanded = self._expand_query(cleaned)

        q_vec  = self.vectorizer.transform([expanded])
        scores = (self.tfidf_matrix @ q_vec.T).toarray().flatten()

        # FIX 3: only consider chunks that actually match query vocabulary
        passing = np.where(scores > MIN_TFIDF_SCORE)[0]
        if len(passing) == 0:
            logger.warning(
                "No chunks passed MIN_TFIDF_SCORE=%.4f for '%s'. "
                "Falling back to top-%d by TF-IDF.",
                MIN_TFIDF_SCORE, cleaned, TFIDF_PRE_K
            )
            idxs = scores.argsort()[::-1][:TFIDF_PRE_K]
        else:
            idxs = passing[scores[passing].argsort()[::-1][:TFIDF_PRE_K]]

        # Use the original cleaned (unexpanded) query for semantic scoring —
        # synonym noise in the expanded string degrades embedding quality
        q_emb = self.embed_model.encode([cleaned], normalize_embeddings=True)[0]

        scored = []
        for i in idxs:
            c      = self.chunks[i]
            sem    = self._semantic_score(q_emb, c)
            tfidf  = float(scores[i])
            pr     = c.get("pagerank", 0.0)
            hybrid = ALPHA_TFIDF * tfidf + ALPHA_SEMANTIC * sem + ALPHA_PAGERANK * pr
            scored.append((c, hybrid))

        scored.sort(key=lambda x: x[1], reverse=True)

        # Debug output — set LOG_LEVEL=DEBUG to see during development
        for rank, (c, h) in enumerate(scored[:3]):
            logger.debug(
                "rank=%d  page=%-4s  score=%.4f  | %s",
                rank + 1, c.get("page"), h,
                c.get("text", "")[:100].replace("\n", " ")
            )

        return [
            {**c, "score": float(s), "method": "hybrid"}
            for c, s in scored[:top_k]
        ]

    # ── 4. Benchmarking utility ────────────────────────────────────────────────

    def benchmark(self, query: str, top_k: int = CHUNK_TOP_K):
        """Run all three retrievers and return timing + results."""
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
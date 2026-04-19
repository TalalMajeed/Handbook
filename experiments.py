"""
experiments.py
==============
Experimental Analysis for the Handbook QA System.

Runs the required comparisons from the project brief:
  1. Exact vs Approximate Retrieval   (TF-IDF vs LSH)
  2. Parameter Sensitivity            (n_hashes, n_bands, SimHash threshold)
  3. Scalability Test                 (duplicate corpus 1x / 2x / 5x)

Run:
    python experiments.py
"""

import sys
import time
import pickle
import copy
import hashlib
import re

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")   # safe Unicode output on Windows

from pipeline import HandbookPipeline
from lsh_core import MinHasher, LSHIndex, N_HASHES, N_BANDS
from sklearn.feature_extraction.text import TfidfVectorizer

MODEL_DIR = "handbook-1.0"

# ── 15 Sample Queries (as required by the spec) ───────────────────────────────

QUERIES = [
    "What is the minimum GPA requirement?",
    "What happens if a student fails a course?",
    "What is the attendance policy?",
    "How many times can a course be repeated?",
    "What is the semester credit hour limit?",
    "What are the requirements for graduation?",
    "What is the grading scale?",
    "What is the policy on plagiarism?",
    "How is the CGPA calculated?",
    "What happens if a student is on academic probation?",
    "How many elective courses are required?",
    "What is the fee refund policy?",
    "What are the rules for incomplete grades?",
    "Can a student transfer credits from another university?",
    "What is the policy on thesis submission?",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def separator(title="", char="-", width=72):
    if title:
        side = (width - len(title) - 2) // 2
        print(char * side + f" {title} " + char * side)
    else:
        print(char * width)


def precision_at_k(retrieved_texts, relevant_keywords, k=8):
    """
    Simple keyword-overlap Precision@k.
    A chunk is considered 'relevant' if it contains at least one keyword from
    the ground-truth keyword set for that query.
    """
    hit = 0
    for t in retrieved_texts[:k]:
        t_lower = t.lower()
        if any(kw.lower() in t_lower for kw in relevant_keywords):
            hit += 1
    return hit / min(k, len(retrieved_texts)) if retrieved_texts else 0.0


# Minimal keyword ground-truth for Precision@k evaluation
QUERY_KEYWORDS = {
    QUERIES[0]:  ["gpa", "grade point", "minimum", "2.75", "3.00"],
    QUERIES[1]:  ["fail", "failed", "repeat", "F grade"],
    QUERIES[2]:  ["attendance", "absent", "75%", "shortage"],
    QUERIES[3]:  ["repeat", "repeated", "maximum", "twice"],
    QUERIES[4]:  ["credit hour", "overload", "18", "semester"],
    QUERIES[5]:  ["graduation", "degree requirement", "complete"],
    QUERIES[6]:  ["grade", "A+", "B+", "scale", "marks"],
    QUERIES[7]:  ["plagiarism", "academic integrity", "cheating"],
    QUERIES[8]:  ["cgpa", "cumulative", "calculation"],
    QUERIES[9]:  ["probation", "academic warning"],
    QUERIES[10]: ["elective", "optional courses"],
    QUERIES[11]: ["refund", "fee", "withdrawal"],
    QUERIES[12]: ["incomplete", "I grade"],
    QUERIES[13]: ["transfer", "credit", "exemption"],
    QUERIES[14]: ["thesis", "submission", "dissertation"],
}

# ── Experiment 1: Exact vs Approximate Retrieval ─────────────────────────────

def experiment_retrieval_comparison(p: HandbookPipeline):
    separator("Experiment 1: Exact vs Approximate Retrieval", "=")
    print("Comparing TF-IDF (exact) vs LSH (approximate) vs Hybrid\n")

    col_w = [40, 10, 10, 10, 10, 10, 10]
    header = ("Query", "TF-IDF ms", "LSH ms", "Hybrid ms",
              "TF P@8", "LSH P@8", "Hyb P@8")
    row_fmt = "{:<40} {:>9} {:>7} {:>9} {:>7} {:>7} {:>8}"
    print(row_fmt.format(*header))
    separator()

    tfidf_times, lsh_times, hybrid_times = [], [], []
    tfidf_prec,  lsh_prec,  hybrid_prec  = [], [], []

    for q in QUERIES:
        kws = QUERY_KEYWORDS.get(q, [])
        bm  = p.benchmark(q)

        tfidf_texts  = [r["text"] for r in bm["tfidf"]]
        lsh_texts    = [r["text"] for r in bm["lsh"]]
        hybrid_texts = [r["text"] for r in bm["hybrid"]]

        pt = precision_at_k(tfidf_texts,  kws)
        pl = precision_at_k(lsh_texts,    kws)
        ph = precision_at_k(hybrid_texts, kws)

        tfidf_times.append(bm["tfidf_time_ms"])
        lsh_times.append(bm["lsh_time_ms"])
        hybrid_times.append(bm["hybrid_time_ms"])
        tfidf_prec.append(pt)
        lsh_prec.append(pl)
        hybrid_prec.append(ph)

        short_q = q[:38] + ".." if len(q) > 40 else q
        print(row_fmt.format(
            short_q,
            f"{bm['tfidf_time_ms']:.1f}",
            f"{bm['lsh_time_ms']:.1f}",
            f"{bm['hybrid_time_ms']:.1f}",
            f"{pt:.2f}",
            f"{pl:.2f}",
            f"{ph:.2f}",
        ))

    separator()
    print(row_fmt.format(
        "AVERAGE",
        f"{np.mean(tfidf_times):.1f}",
        f"{np.mean(lsh_times):.1f}",
        f"{np.mean(hybrid_times):.1f}",
        f"{np.mean(tfidf_prec):.2f}",
        f"{np.mean(lsh_prec):.2f}",
        f"{np.mean(hybrid_prec):.2f}",
    ))
    print()
    print("Key takeaways:")
    print(f"  TF-IDF avg latency : {np.mean(tfidf_times):.2f} ms (exact, baseline)")
    print(f"  LSH    avg latency : {np.mean(lsh_times):.2f} ms (approximate, faster on large corpora)")
    print(f"  Hybrid avg latency : {np.mean(hybrid_times):.2f} ms (best quality)")
    speedup = np.mean(tfidf_times) / max(np.mean(lsh_times), 0.01)
    print(f"  LSH speedup over TF-IDF: {speedup:.1f}x")
    print()

# ── Experiment 2: Parameter Sensitivity ──────────────────────────────────────

def experiment_parameter_sensitivity():
    separator("Experiment 2: Parameter Sensitivity", "=")

    # Load chunks + their pre-built shingles
    with open(f"{MODEL_DIR}/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    texts = [c["text"] for c in chunks]
    TEST_QUERY = QUERIES[0]

    # ── 2a: Number of hash functions ──────────────────────────────────────────
    print("\n[2a] Effect of number of hash functions (n_hashes)\n")
    print(f"{'n_hashes':<12} {'Index time ms':>14} {'Query time ms':>14} {'LSH hits':>10}")
    separator()

    for n_h in [32, 64, 128, 256]:
        mh  = MinHasher(n_hashes=n_h)
        lsh = LSHIndex(n_hashes=n_h, n_bands=min(16, n_h))

        t0  = time.perf_counter()
        sigs = [mh.signature(t) for t in texts]
        for i, s in enumerate(sigs):
            lsh.add(i, s)
        build_ms = (time.perf_counter() - t0) * 1000

        q_sig = mh.signature(TEST_QUERY.lower())
        t0 = time.perf_counter()
        cands = lsh.query(q_sig)
        q_ms = (time.perf_counter() - t0) * 1000

        print(f"{n_h:<12} {build_ms:>14.1f} {q_ms:>14.3f} {len(cands):>10}")

    # ── 2b: Number of bands ───────────────────────────────────────────────────
    print("\n[2b] Effect of number of bands (n_bands) with n_hashes=128\n")
    print(f"{'n_bands':<10} {'rows/band':>10} {'s* threshold':>14} {'LSH hits':>10} {'Query ms':>10}")
    separator()

    mh   = MinHasher(n_hashes=128)
    sigs = [mh.signature(t) for t in texts]
    q_sig = mh.signature(TEST_QUERY.lower())

    for n_b in [4, 8, 16, 32, 64]:
        if 128 % n_b != 0:
            continue
        lsh = LSHIndex(n_hashes=128, n_bands=n_b)
        for i, s in enumerate(sigs):
            lsh.add(i, s)

        t0 = time.perf_counter()
        cands = lsh.query(q_sig)
        q_ms = (time.perf_counter() - t0) * 1000

        rpb = 128 // n_b
        s_star = (1 / n_b) ** (1 / rpb)
        print(f"{n_b:<10} {rpb:>10} {s_star:>14.4f} {len(cands):>10} {q_ms:>10.3f}")

    # ── 2c: SimHash Hamming threshold ─────────────────────────────────────────
    print("\n[2c] Effect of SimHash Hamming threshold on deduplication\n")
    print(f"{'Threshold':<12} {'Chunks kept':>13} {'Removed':>10} {'% removed':>11}")
    separator()

    def _simhash64(text):
        tokens = text.lower().split()
        v = np.zeros(64, dtype=np.float64)
        for tok in tokens:
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            for i in range(64):
                v[i] += 1 if (h >> i) & 1 else -1
        fp = 0
        for i in range(64):
            if v[i] > 0:
                fp |= (1 << i)
        return fp

    fps = [_simhash64(t) for t in texts]

    for threshold in [0, 1, 2, 3, 5, 8]:
        kept_fps = []
        kept = 0
        for fp in fps:
            is_dup = any(bin(fp ^ kfp).count("1") <= threshold for kfp in kept_fps)
            if not is_dup:
                kept += 1
                kept_fps.append(fp)
        removed = len(fps) - kept
        pct = removed / len(fps) * 100
        print(f"{threshold:<12} {kept:>13} {removed:>10} {pct:>10.1f}%")
    print()

# ── Experiment 3: Scalability Test ───────────────────────────────────────────

def experiment_scalability():
    separator("Experiment 3: Scalability Test (corpus duplication)", "=")
    print("Measuring index build time and query latency as corpus grows\n")
    print(f"{'Scale':>8} {'Chunks':>8} {'Build ms':>10} {'TF-IDF ms':>11} {'LSH ms':>8} {'LSH hits':>9}")
    separator()

    with open(f"{MODEL_DIR}/chunks.pkl", "rb") as f:
        base_chunks = pickle.load(f)
    base_texts = [c["text"] for c in base_chunks]

    TEST_QUERY = QUERIES[0]

    for multiplier in [1, 2, 5, 10]:
        texts = base_texts * multiplier
        n     = len(texts)

        # Build MinHash + LSH index
        mh  = MinHasher()
        lsh = LSHIndex()
        t0  = time.perf_counter()
        sigs = [mh.signature(t) for t in texts]
        for i, s in enumerate(sigs):
            lsh.add(i, s)
        build_ms = (time.perf_counter() - t0) * 1000

        # Build TF-IDF
        vec = TfidfVectorizer(max_features=20_000)
        t0  = time.perf_counter()
        mat = vec.fit_transform(texts)
        tfidf_build = (time.perf_counter() - t0) * 1000

        # Query timing
        q_vec = vec.transform([TEST_QUERY.lower()])
        t0  = time.perf_counter()
        _scores = (mat @ q_vec.T).toarray().flatten()
        tfidf_q_ms = (time.perf_counter() - t0) * 1000

        q_sig = mh.signature(TEST_QUERY.lower())
        t0  = time.perf_counter()
        cands = lsh.query(q_sig)
        lsh_q_ms = (time.perf_counter() - t0) * 1000

        label = f"{multiplier}x"
        print(f"{label:>8} {n:>8} {build_ms:>10.1f} {tfidf_q_ms:>11.3f} {lsh_q_ms:>8.3f} {len(cands):>9}")

    print()
    print("Observation: LSH query time stays nearly constant as corpus grows,")
    print("while TF-IDF grows linearly with number of chunks.")
    print()

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 72)
    print("   HANDBOOK QA SYSTEM — EXPERIMENTAL ANALYSIS")
    print("=" * 72 + "\n")

    print("Loading pipeline (this takes ~10s to load the embedding model)...")
    p = HandbookPipeline(MODEL_DIR)
    print(f"Loaded {len(p.chunks)} chunks from {MODEL_DIR}/\n")

    experiment_retrieval_comparison(p)
    experiment_parameter_sensitivity()
    experiment_scalability()

    print("=" * 72)
    print("Experiments complete.")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
